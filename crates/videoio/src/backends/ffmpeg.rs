//! Native FFmpeg video capture backend using ffmpeg-next

use crate::{Result, VideoCapture, VideoError};
use ffmpeg_next as ffmpeg;
use image::GrayImage;
use std::path::Path;

pub struct NativeFfmpegCapture {
    ictx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    width: u32,
    height: u32,
    current_frame: Option<GrayImage>,
    /// `send_eof` has been issued; the decoder is draining.
    flushed: bool,
}

impl std::fmt::Debug for NativeFfmpegCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeFfmpegCapture")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl NativeFfmpegCapture {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        ffmpeg::init().map_err(|e| VideoError::Backend(format!("FFmpeg init failed: {}", e)))?;

        let ictx = ffmpeg::format::input(&path)
            .map_err(|e| VideoError::Backend(format!("Failed to open input: {}", e)))?;

        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| VideoError::Backend("No video stream found".to_string()))?;

        let stream_index = input.index();
        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())
            .map_err(|e| VideoError::Backend(format!("Failed to get codec context: {}", e)))?;

        let decoder = context
            .decoder()
            .video()
            .map_err(|e| VideoError::Backend(format!("Failed to get decoder: {}", e)))?;

        let width = decoder.width();
        let height = decoder.height();

        Ok(Self {
            ictx,
            decoder,
            stream_index,
            width,
            height,
            current_frame: None,
            flushed: false,
        })
    }

    /// Next packet of the video stream, or `None` at end of input.
    fn next_packet(&mut self) -> Option<ffmpeg::Packet> {
        let index = self.stream_index;
        self.ictx
            .packets()
            .find(|(stream, _)| stream.index() == index)
            .map(|(_, packet)| packet)
    }

    /// Convert a decoded frame to an unpadded GRAY8 image.
    fn to_gray(&mut self, decoded: &ffmpeg::util::frame::Video) -> Result<GrayImage> {
        // The swscale context is built per frame, from the frame's own format.
        // ffmpeg-next 9's `scaling::Context` is not `Send` (it wraps a raw
        // `*mut SwsContext`), `VideoCapture` must be, and this crate forbids
        // `unsafe` - so it cannot be cached in the capture. Building it per
        // frame also follows a mid-stream format change.
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            decoded.format(),
            decoded.width(),
            decoded.height(),
            ffmpeg::format::Pixel::GRAY8,
            self.width,
            self.height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .map_err(|e| VideoError::CaptureFailed(format!("Failed to initialize scaler: {}", e)))?;
        let mut gray_frame =
            ffmpeg::util::frame::Video::new(ffmpeg::format::Pixel::GRAY8, self.width, self.height);
        scaler
            .run(decoded, &mut gray_frame)
            .map_err(|e| VideoError::CaptureFailed(format!("Scaling failed: {}", e)))?;
        let data = gray_frame.data(0);
        let stride = gray_frame.stride(0);
        let (width, height) = (self.width as usize, self.height as usize);
        let mut unpadded = Vec::with_capacity(width * height);
        for row in 0..height {
            unpadded.extend_from_slice(&data[row * stride..row * stride + width]);
        }
        GrayImage::from_raw(self.width, self.height, unpadded)
            .ok_or_else(|| VideoError::CaptureFailed("Image creation failed".to_string()))
    }
}

impl VideoCapture for NativeFfmpegCapture {
    fn is_opened(&self) -> bool {
        true
    }

    /// FFmpeg's send/receive loop: return any frame the decoder already holds,
    /// and feed it a packet only when it asks for input (`EAGAIN`).
    ///
    /// This used to send one packet and receive at most one frame per packet.
    /// A decoder with reordering delay (B-frames) or a packet holding several
    /// frames (packed MPEG-4 in AVI) still has output pending when the next
    /// packet is sent; `send_packet` then fails with `EAGAIN`, which was
    /// reported as `CaptureFailed` and ended the stream mid-video. Frames still
    /// buffered at end of input were also only partly drained.
    fn grab(&mut self) -> Result<()> {
        loop {
            let mut decoded = ffmpeg::util::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    self.current_frame = Some(self.to_gray(&decoded)?);
                    return Ok(());
                }
                Err(ffmpeg::Error::Eof) => {
                    return Err(VideoError::CaptureFailed("End of stream".to_string()));
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {}
                Err(e) => {
                    return Err(VideoError::CaptureFailed(format!(
                        "Receive frame failed: {}",
                        e
                    )));
                }
            }
            // The decoder wants input.
            if self.flushed {
                return Err(VideoError::CaptureFailed("End of stream".to_string()));
            }
            match self.next_packet() {
                Some(packet) => self
                    .decoder
                    .send_packet(&packet)
                    .map_err(|e| VideoError::CaptureFailed(format!("Send packet failed: {}", e)))?,
                None => {
                    self.decoder.send_eof().map_err(|e| {
                        VideoError::CaptureFailed(format!("Flushing decoder failed: {}", e))
                    })?;
                    self.flushed = true;
                }
            }
        }
    }

    fn retrieve(&mut self) -> Result<GrayImage> {
        self.current_frame
            .clone()
            .ok_or_else(|| VideoError::CaptureFailed("No frame grabbed".to_string()))
    }
}
