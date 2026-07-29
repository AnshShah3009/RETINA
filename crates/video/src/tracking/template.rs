use crate::Result;
use cv_core::Error;
use image::{GrayImage, Luma};

use super::Tracker;

/// Simple template matching tracker
pub struct TemplateTracker {
    template: Option<GrayImage>,
    last_position: Option<(u32, u32)>,
    search_radius: u32,
}

impl TemplateTracker {
    pub fn new(search_radius: u32) -> Self {
        Self {
            template: None,
            last_position: None,
            search_radius,
        }
    }

    fn extract_template(&self, frame: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
        let mut template = GrayImage::new(w, h);

        for dy in 0..h {
            for dx in 0..w {
                let px = (x + dx).min(frame.width() - 1);
                let py = (y + dy).min(frame.height() - 1);
                let val = frame.get_pixel(px, py)[0];
                template.put_pixel(dx, dy, Luma([val]));
            }
        }

        template
    }

    fn find_best_match(&self, frame: &GrayImage) -> Option<(u32, u32)> {
        let template = self.template.as_ref()?;
        let last_pos = self.last_position?;

        let (tw, th) = (template.width(), template.height());
        let mut best_pos = last_pos;
        let mut best_score = f32::INFINITY;

        // Search in neighborhood
        let search_min_x = last_pos.0.saturating_sub(self.search_radius);
        let search_max_x = (last_pos.0 + self.search_radius).min(frame.width() - tw);
        let search_min_y = last_pos.1.saturating_sub(self.search_radius);
        let search_max_y = (last_pos.1 + self.search_radius).min(frame.height() - th);

        for y in search_min_y..=search_max_y {
            for x in search_min_x..=search_max_x {
                let score = self.compute_match_score(frame, template, x, y);
                if score < best_score {
                    best_score = score;
                    best_pos = (x, y);
                }
            }
        }

        Some(best_pos)
    }

    fn compute_match_score(&self, frame: &GrayImage, template: &GrayImage, x: u32, y: u32) -> f32 {
        let mut sum_squared_diff = 0.0f32;
        let mut count = 0;

        for ty in 0..template.height() {
            for tx in 0..template.width() {
                let fx = (x + tx).min(frame.width() - 1);
                let fy = (y + ty).min(frame.height() - 1);

                let frame_val = frame.get_pixel(fx, fy)[0] as f32;
                let template_val = template.get_pixel(tx, ty)[0] as f32;

                let diff = frame_val - template_val;
                sum_squared_diff += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            sum_squared_diff / count as f32
        } else {
            f32::INFINITY
        }
    }
}

impl Tracker for TemplateTracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()> {
        let (x, y, w, h) = bbox;
        self.template = Some(self.extract_template(frame, x, y, w, h));
        self.last_position = Some((x, y));
        Ok(())
    }

    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)> {
        if let Some(new_pos) = self.find_best_match(frame) {
            self.last_position = Some(new_pos);

            if let Some(ref template) = self.template {
                return Ok((new_pos.0, new_pos.1, template.width(), template.height()));
            }
        }

        Err(Error::RuntimeError(
            "Failed to track object".to_string(),
        ))
    }
}
