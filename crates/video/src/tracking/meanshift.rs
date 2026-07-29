use crate::Result;
use cv_core::Error;
use image::GrayImage;

use super::Tracker;

/// Mean-shift tracker
pub struct MeanShiftTracker {
    target_model: Option<Vec<f32>>,
    last_position: Option<(f64, f64)>,
    window_size: (u32, u32),
    max_iterations: usize,
    epsilon: f64,
}

impl MeanShiftTracker {
    pub fn new(window_width: u32, window_height: u32) -> Self {
        Self {
            target_model: None,
            last_position: None,
            window_size: (window_width, window_height),
            max_iterations: 10,
            epsilon: 0.1,
        }
    }

    fn compute_color_histogram(&self, frame: &GrayImage, cx: f64, cy: f64) -> Vec<f32> {
        let mut histogram = vec![0.0f32; 256];
        let (half_w, half_h) = (
            self.window_size.0 as f64 / 2.0,
            self.window_size.1 as f64 / 2.0,
        );

        let min_x = (cx - half_w).max(0.0) as u32;
        let max_x = (cx + half_w).min(frame.width() as f64 - 1.0) as u32;
        let min_y = (cy - half_h).max(0.0) as u32;
        let max_y = (cy + half_h).min(frame.height() as f64 - 1.0) as u32;

        let mut total_weight = 0.0;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // Epanechnikov kernel weight
                let dx = (x as f64 - cx) / half_w;
                let dy = (y as f64 - cy) / half_h;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1.0 {
                    let weight = 1.0 - dist_sq;
                    let intensity = frame.get_pixel(x, y)[0] as usize;
                    histogram[intensity] += weight as f32;
                    total_weight += weight;
                }
            }
        }

        // Normalize
        if total_weight > 0.0 {
            for val in &mut histogram {
                *val /= total_weight as f32;
            }
        }

        histogram
    }

    fn compute_mean_shift(&self, frame: &GrayImage, cx: f64, cy: f64) -> (f64, f64) {
        let target = match self.target_model.as_ref() {
            Some(t) => t,
            None => return (cx, cy),
        };
        let (half_w, half_h) = (
            self.window_size.0 as f64 / 2.0,
            self.window_size.1 as f64 / 2.0,
        );

        // Compute the candidate histogram at the current position
        let candidate = self.compute_color_histogram(frame, cx, cy);

        let mut numerator_x = 0.0;
        let mut numerator_y = 0.0;
        let mut denominator = 0.0;

        let min_x = (cx - half_w).max(0.0) as u32;
        let max_x = (cx + half_w).min(frame.width() as f64 - 1.0) as u32;
        let min_y = (cy - half_h).max(0.0) as u32;
        let max_y = (cy + half_h).min(frame.height() as f64 - 1.0) as u32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = (x as f64 - cx) / half_w;
                let dy = (y as f64 - cy) / half_h;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1.0 {
                    let intensity = frame.get_pixel(x, y)[0] as usize;
                    let weight = (target[intensity] / (candidate[intensity] + 1e-6)).sqrt();

                    numerator_x += x as f64 * weight as f64;
                    numerator_y += y as f64 * weight as f64;
                    denominator += weight as f64;
                }
            }
        }

        if denominator > 0.0 {
            (numerator_x / denominator, numerator_y / denominator)
        } else {
            (cx, cy)
        }
    }
}

impl Tracker for MeanShiftTracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()> {
        let (x, y, w, h) = bbox;
        self.window_size = (w, h);
        let cx = x as f64 + w as f64 / 2.0;
        let cy = y as f64 + h as f64 / 2.0;

        self.target_model = Some(self.compute_color_histogram(frame, cx, cy));
        self.last_position = Some((cx, cy));

        Ok(())
    }

    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)> {
        let (mut cx, mut cy) = self
            .last_position
            .ok_or_else(|| Error::RuntimeError("Tracker not initialized".to_string()))?;

        for _ in 0..self.max_iterations {
            let (new_cx, new_cy) = self.compute_mean_shift(frame, cx, cy);

            let dist = ((new_cx - cx).powi(2) + (new_cy - cy).powi(2)).sqrt();

            cx = new_cx;
            cy = new_cy;

            if dist < self.epsilon {
                break;
            }
        }

        self.last_position = Some((cx, cy));

        let x = (cx - self.window_size.0 as f64 / 2.0) as u32;
        let y = (cy - self.window_size.1 as f64 / 2.0) as u32;

        Ok((x, y, self.window_size.0, self.window_size.1))
    }
}
