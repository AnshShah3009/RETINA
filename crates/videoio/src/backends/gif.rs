use crate::{Result, VideoCapture, VideoError};
use image::{AnimationDecoder, GrayImage};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct GifCapture {
    frames: Vec<image::Frame>,
    /// Index of the next frame [`VideoCapture::grab`] will decode.
    next_idx: usize,
    /// Frame produced by the last successful grab. `retrieve` returns it and
    /// does not advance.
    grabbed: Option<GrayImage>,
}

impl std::fmt::Debug for GifCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GifCapture")
            .field("frame_count", &self.frames.len())
            .field("next_idx", &self.next_idx)
            .field("has_grabbed", &self.grabbed.is_some())
            .finish()
    }
}

impl GifCapture {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(VideoError::Io)?;
        let reader = BufReader::new(file);
        let decoder = image::codecs::gif::GifDecoder::new(reader)
            .map_err(|e| VideoError::Backend(format!("Failed to decode GIF: {}", e)))?;

        let frames = decoder
            .into_frames()
            .collect_frames()
            .map_err(|e| VideoError::Backend(format!("Failed to collect frames: {}", e)))?;

        if frames.is_empty() {
            return Err(VideoError::Backend("GIF contains no frames".to_string()));
        }

        Ok(Self {
            frames,
            next_idx: 0,
            grabbed: None,
        })
    }

    fn frame_gray(frame: &image::Frame) -> GrayImage {
        image::DynamicImage::ImageRgba8(frame.buffer().clone()).into_luma8()
    }
}

impl VideoCapture for GifCapture {
    fn is_opened(&self) -> bool {
        !self.frames.is_empty()
    }

    /// Advance to the next frame and keep it. A second grab without
    /// `retrieve` replaces that frame, so two grabs then one retrieve skip
    /// a frame. OpenCV's contract, matching the FFmpeg backend.
    fn grab(&mut self) -> Result<()> {
        if self.next_idx >= self.frames.len() {
            return Err(VideoError::CaptureFailed("End of stream".to_string()));
        }
        let img = Self::frame_gray(&self.frames[self.next_idx]);
        self.next_idx += 1;
        self.grabbed = Some(img);
        Ok(())
    }

    /// Return the frame from the last successful `grab`. Does not advance.
    fn retrieve(&mut self) -> Result<GrayImage> {
        self.grabbed
            .clone()
            .ok_or_else(|| VideoError::CaptureFailed("No frame grabbed".to_string()))
    }
}
