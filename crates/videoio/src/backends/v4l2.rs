//! Video4Linux2 capture backend

use crate::{Result, VideoCapture, VideoError};
use image::GrayImage;
use v4l::buffer::Type;
use v4l::format::FourCC;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

pub struct V4L2Capture {
    device: Device,
    stream: Option<MmapStream<'static>>,
}

impl std::fmt::Debug for V4L2Capture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4L2Capture")
            .field("device", &"v4l::Device")
            .field("stream_active", &self.stream.is_some())
            .finish()
    }
}

impl V4L2Capture {
    pub fn new(path: &str) -> Result<Self> {
        let device = Device::with_path(path)
            .map_err(|e| VideoError::Backend(format!("Failed to open device: {}", e)))?;

        Ok(Self {
            device,
            stream: None,
        })
    }

    /// Requests `width`x`height` YUYV and starts streaming.
    ///
    /// `set_format` is a request, not a command: a driver may substitute a
    /// different pixel format *or size* (this is the normal outcome for a size
    /// the device does not advertise — the kernel picks the nearest supported
    /// mode). Both are checked here, so a caller never receives frames in a
    /// format or size it did not ask for: `retrieve` indexes the buffer
    /// according to what was negotiated, and `open_camera` promises the
    /// requested resolution.
    pub fn start_stream(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(VideoError::InvalidParameters(format!(
                "Camera resolution must be non-zero, got {}x{}",
                width, height
            )));
        }

        let mut fmt = self
            .device
            .format()
            .map_err(|e| VideoError::Backend(format!("Failed to get format: {}", e)))?;

        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"YUYV"); // Common format, we'll convert to Gray

        self.device
            .set_format(&fmt)
            .map_err(|e| VideoError::Backend(format!("Failed to set format: {}", e)))?;

        // set_format is a request: drivers may substitute a different FourCC.
        // Verify before relying on YUYV layout in retrieve().
        let negotiated = self
            .device
            .format()
            .map_err(|e| VideoError::Backend(format!("Failed to get format: {}", e)))?;
        if negotiated.fourcc != FourCC::new(b"YUYV") {
            return Err(VideoError::Backend(format!(
                "Driver negotiated {:?} instead of YUYV; unsupported",
                negotiated.fourcc
            )));
        }
        if (negotiated.width, negotiated.height) != (width, height) {
            return Err(VideoError::Backend(format!(
                "Driver negotiated {}x{} instead of the requested {}x{}; \
                 request a size the device advertises (e.g. `v4l2-ctl --list-formats-ext`)",
                negotiated.width, negotiated.height, width, height
            )));
        }

        let stream = MmapStream::with_buffers(&self.device, Type::VideoCapture, 4)
            .map_err(|e| VideoError::Backend(format!("Failed to create stream: {}", e)))?;

        self.stream = Some(stream);
        Ok(())
    }
}

/// Converts one YUYV (a.k.a. YUY2) buffer to 8-bit grayscale by taking its Y
/// samples: `Y0 U0 Y1 V0 Y2 U1 Y3 V1 …`, so pixel `x` of a row starts at byte
/// `2 * x`.
///
/// `stride` is the driver's bytes-per-line (`v4l::format::Format::stride`),
/// which V4L2 does *not* require to be `2 * width`: drivers are free to pad
/// rows. Indexing with `2 * width` reads each row from progressively further
/// into the buffer, so every row after the first is skewed and the last rows
/// fall off the end — silently, since the data is still in bounds.
fn yuyv_to_gray(data: &[u8], width: u32, height: u32, stride: usize) -> Result<GrayImage> {
    if width == 0 || height == 0 {
        return Err(VideoError::CaptureFailed(format!(
            "Empty YUYV frame: {}x{}",
            width, height
        )));
    }

    let row_bytes = width as usize * 2;
    if stride < row_bytes {
        return Err(VideoError::CaptureFailed(format!(
            "Driver reported a stride of {} bytes for {}-pixel YUYV rows ({} bytes)",
            stride, width, row_bytes
        )));
    }

    // Only the byte range that carries pixels has to be present: a driver may
    // report a `sizeimage` that omits the last row's padding.
    let needed = (height as usize - 1) * stride + row_bytes;
    if data.len() < needed {
        return Err(VideoError::CaptureFailed(format!(
            "Frame buffer too small: got {} bytes, need {} for {}x{} YUYV with stride {}",
            data.len(),
            needed,
            width,
            height,
            stride
        )));
    }

    let width = width as usize;
    let mut gray = GrayImage::new(width as u32, height);
    for y in 0..height as usize {
        let row = &data[y * stride..y * stride + row_bytes];
        let out = &mut gray.as_mut()[y * width..(y + 1) * width];
        for (x, pixel) in out.iter_mut().enumerate() {
            *pixel = row[x * 2];
        }
    }
    Ok(gray)
}

impl VideoCapture for V4L2Capture {
    fn is_opened(&self) -> bool {
        self.stream.is_some()
    }

    fn grab(&mut self) -> Result<()> {
        // v4l-rust grab is essentially next() on the stream
        Ok(())
    }

    fn retrieve(&mut self) -> Result<GrayImage> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| VideoError::CaptureFailed("Stream not started".to_string()))?;

        let (data, _metadata) = stream
            .next()
            .map_err(|e| VideoError::CaptureFailed(format!("Failed to grab frame: {}", e)))?;

        let fmt = self
            .device
            .format()
            .map_err(|e| VideoError::Backend(format!("Failed to get format: {}", e)))?;

        // YUYV to grayscale. The layout is driven by what the driver reports
        // (bytes per line, size), never by what was requested, so a negotiation
        // this backend did not anticipate cannot make it read past the buffer
        // or skew rows.
        let stride = if fmt.stride == 0 {
            2 * fmt.width as usize
        } else {
            fmt.stride as usize
        };
        yuyv_to_gray(data, fmt.width, fmt.height, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one YUYV frame of `width`x`height` pixels with `stride` bytes per
    /// line. The Y sample at `(x, y)` is `y * 10 + x`, the chroma bytes are
    /// filled in but must be ignored, and row padding is filled with `0xFF` so
    /// that any read of it shows up as 255 in the output.
    fn frame(width: usize, height: usize, stride: usize) -> (Vec<u8>, Vec<u8>) {
        let mut data = vec![0xFFu8; stride * height];
        let mut expected = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let luma = (y * 10 + x) as u8;
                data[y * stride + 2 * x] = luma;
                data[y * stride + 2 * x + 1] = 128;
                expected.push(luma);
            }
        }
        (data, expected)
    }

    fn naive(data: &[u8], pixels: usize) -> Vec<u8> {
        (0..pixels).map(|i| data[i * 2]).collect()
    }

    #[test]
    fn yuyv_reads_rows_at_the_driver_stride() {
        let (width, height) = (4usize, 3usize);
        let stride = 2 * width + 6; // rows padded, which V4L2 permits
        let (data, expected) = frame(width, height, stride);

        let gray = yuyv_to_gray(&data, width as u32, height as u32, stride).unwrap();
        assert_eq!(gray.as_raw()[..], expected[..]);

        // The formula this replaced — `data[i * 2]`, ignoring the stride —
        // reads the padding as pixels and shifts every row after the first.
        let naive = naive(&data, width * height);
        assert_eq!(
            naive,
            vec![0, 1, 2, 3, 255, 255, 255, 10, 11, 12, 13, 255],
            "pin the defect: with a padded stride the naive read returns padding as pixels \
             and never reaches the last row"
        );
    }

    #[test]
    fn yuyv_tightly_packed_rows_are_unchanged() {
        // Control: equally sized rows (what a UVC webcam reports) read
        // identically before and after, so this change cannot alter real frames.
        let (width, height) = (4usize, 3usize);
        let stride = 2 * width;
        let (data, expected) = frame(width, height, stride);

        let gray = yuyv_to_gray(&data, width as u32, height as u32, stride).unwrap();
        assert_eq!(gray.as_raw()[..], expected[..]);
        assert_eq!(naive(&data, width * height), expected);
    }

    #[test]
    fn yuyv_rejects_impossible_layouts() {
        // A stride narrower than a row cannot address the pixels.
        assert!(yuyv_to_gray(&[0u8; 64], 4, 2, 4).is_err());
        // A buffer missing its last row.
        assert!(yuyv_to_gray(&[0u8; 23], 4, 3, 8).is_err());
        // ...but a buffer that stops after the last row's pixels is complete.
        assert!(yuyv_to_gray(&[0u8; 24], 4, 3, 8).is_ok());
        // Zero-sized frames must error rather than underflow `height - 1`.
        assert!(yuyv_to_gray(&[], 0, 0, 0).is_err());
        assert!(yuyv_to_gray(&[], 4, 0, 8).is_err());
    }
}
