//! Video input/output and camera capture
//!
//! This crate provides a unified interface for capturing video from cameras
//! and reading/writing video files.

use image::GrayImage;
use std::fmt::Debug;

pub type Result<T> = std::result::Result<T, VideoError>;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Capture failed: {0}")]
    CaptureFailed(String),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),
}

/// Generic interface for video capture devices
pub trait VideoCapture: Send + Debug {
    fn is_opened(&self) -> bool;
    fn grab(&mut self) -> Result<()>;
    fn retrieve(&mut self) -> Result<GrayImage>;
    fn read(&mut self) -> Result<GrayImage> {
        self.grab()?;
        self.retrieve()
    }
}

/// Generic interface for video writing
pub trait VideoWriter: Send + Debug {
    fn write(&mut self, frame: &GrayImage) -> Result<()>;
}

pub mod backends;

/// Open a video file
pub fn open_video(path: &str) -> Result<Box<dyn VideoCapture>> {
    let p = std::path::Path::new(path);

    // 1. Try Pure Rust GIF
    if let Some(ext) = p.extension() {
        if ext.to_string_lossy().to_lowercase() == "gif" {
            let cap = backends::GifCapture::new(path)?;
            return Ok(Box::new(cap));
        }
    }

    // 2. Try Pure Rust Image Sequence (Directory)
    if p.is_dir() {
        let cap = backends::PngSequenceCapture::new(path)?;
        return Ok(Box::new(cap));
    }

    // 3. Try FFmpeg if enabled
    #[cfg(feature = "ffmpeg")]
    {
        if let Ok(cap) = backends::NativeFfmpegCapture::new(path) {
            return Ok(Box::new(cap));
        }
    }

    // 4. Failure
    Err(VideoError::Backend("Could not open video. Supported Pure Rust formats: GIF, Image Sequence (Dir). Enable 'ffmpeg' feature for MP4/AVI support.".to_string()))
}

/// Open a camera device
pub fn open_camera(_path: &str, _width: u32, _height: u32) -> Result<Box<dyn VideoCapture>> {
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    {
        let mut cap = backends::V4L2Capture::new(_path)?;
        cap.start_stream(_width, _height)?;
        Ok(Box::new(cap))
    }
    #[cfg(not(all(target_os = "linux", feature = "v4l2")))]
    {
        Err(VideoError::Backend(
            "Camera backend disabled or not supported on this platform.".to_string(),
        ))
    }
}
