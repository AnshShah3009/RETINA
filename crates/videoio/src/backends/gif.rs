use crate::{Result, VideoCapture, VideoError};
use image::{AnimationDecoder, GrayImage};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct GifCapture {
    frames: Vec<image::Frame>,
    current_idx: usize,
}

impl std::fmt::Debug for GifCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GifCapture")
            .field("frame_count", &self.frames.len())
            .field("current_idx", &self.current_idx)
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
            current_idx: 0,
        })
    }
}

impl VideoCapture for GifCapture {
    fn is_opened(&self) -> bool {
        !self.frames.is_empty()
    }

    fn grab(&mut self) -> Result<()> {
        if self.current_idx < self.frames.len() {
            Ok(())
        } else {
            Err(VideoError::CaptureFailed("End of stream".to_string()))
        }
    }

    fn retrieve(&mut self) -> Result<GrayImage> {
        if self.current_idx >= self.frames.len() {
            return Err(VideoError::CaptureFailed("End of stream".to_string()));
        }

        let frame = &self.frames[self.current_idx];
        self.current_idx += 1;

        // Convert to GrayImage
        let img = image::DynamicImage::ImageRgba8(frame.buffer().clone());
        Ok(img.into_luma8())
    }
}
