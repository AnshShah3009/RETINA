//! Camera capture backends

#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub mod v4l2;

pub mod gif;
pub mod png_sequence;

#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;

#[cfg(feature = "ffmpeg")]
pub use ffmpeg::NativeFfmpegCapture;

pub use gif::GifCapture;
pub use png_sequence::{PngSequenceCapture, PngSequenceWriter};

#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub use v4l2::V4L2Capture;
