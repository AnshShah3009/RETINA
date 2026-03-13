use crate::Result;
use image::GrayImage;

pub struct Stitcher {
    // panorama state
}

impl Default for Stitcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Stitcher {
    pub fn new() -> Self {
        Self {}
    }

    pub fn stitch(&mut self, images: &[GrayImage]) -> Result<GrayImage> {
        if images.is_empty() {
            return Ok(GrayImage::new(0, 0));
        }

        // Return first image as placeholder for Phase 4
        Ok(images[0].clone())
    }
}
