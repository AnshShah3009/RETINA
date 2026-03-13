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

    pub fn start_stream(&mut self, width: u32, height: u32) -> Result<()> {
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

        let stream = MmapStream::with_buffers(&self.device, Type::VideoCapture, 4)
            .map_err(|e| VideoError::Backend(format!("Failed to create stream: {}", e)))?;

        self.stream = Some(stream);
        Ok(())
    }
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

        // Simplified YUYV to Grayscale conversion
        let mut gray = GrayImage::new(fmt.width, fmt.height);
        for i in 0..(fmt.width * fmt.height) as usize {
            // YUYV: Y0 U0 Y1 V0 ...
            gray.as_mut()[i] = data[i * 2];
        }

        Ok(gray)
    }
}
