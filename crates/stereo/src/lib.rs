//! Stereo vision and depth estimation
//!
//! This module provides stereo matching algorithms for computing disparity maps
//! and depth estimation from stereo image pairs.

use image::GrayImage;

pub mod block_matching;
pub mod depth;
pub mod rectification;
pub mod sgm;
pub use cv_calib3d as calib3d;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use block_matching::*;
pub use calib3d::*;
pub use depth::*;
pub use rectification::*;
pub use sgm::*;

#[cfg(feature = "gpu")]
pub use gpu::*;

pub use cv_core::{Error, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type StereoError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type StereoResult<T> = cv_core::Result<T>;

use cv_runtime::orchestrator::RuntimeRunner;

/// Stereo matching algorithm trait
pub trait StereoMatcher {
    fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap>;
}

/// Stereo matching algorithm trait with explicit context
pub trait StereoMatcherCtx {
    fn compute_ctx(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        group: &RuntimeRunner,
    ) -> Result<DisparityMap>;
}

/// Disparity map representation
#[derive(Debug, Clone)]
pub struct DisparityMap {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub min_disparity: i32,
    pub max_disparity: i32,
}

impl DisparityMap {
    pub fn new(width: u32, height: u32, min_d: i32, max_d: i32) -> Self {
        let size = (width * height) as usize;
        Self {
            data: vec![0.0; size],
            width,
            height,
            min_disparity: min_d,
            max_disparity: max_d,
        }
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        let idx = (y * self.width + x) as usize;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        let idx = (y * self.width + x) as usize;
        if let Some(cell) = self.data.get_mut(idx) {
            *cell = value;
        }
    }

    pub fn is_valid(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    /// Convert to a grayscale image for visualization
    pub fn to_image(&self) -> GrayImage {
        let mut img = GrayImage::new(self.width, self.height);

        // Find min/max for normalization
        let min_val = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        for y in 0..self.height {
            for x in 0..self.width {
                let disparity = self.get(x, y);
                let normalized = if range > 0.0 {
                    ((disparity - min_val) / range * 255.0) as u8
                } else {
                    0
                };
                img.put_pixel(x, y, image::Luma([normalized]));
            }
        }

        img
    }
}

/// Stereo camera parameters
#[derive(Debug, Clone, Copy)]
pub struct StereoParams {
    pub focal_length: f64,
    pub baseline: f64,
    pub cx: f64,
    pub cy: f64,
}

impl StereoParams {
    pub fn new(focal_length: f64, baseline: f64, cx: f64, cy: f64) -> Self {
        Self {
            focal_length,
            baseline,
            cx,
            cy,
        }
    }

    /// Compute depth from disparity
    pub fn disparity_to_depth(&self, disparity: f64) -> Option<f64> {
        if disparity.abs() < 1e-6 {
            None
        } else {
            Some((self.focal_length * self.baseline) / disparity)
        }
    }
}

/// Compute disparity validity mask
pub fn compute_validity_mask(disparity: &DisparityMap, threshold: f32) -> Vec<bool> {
    disparity
        .data
        .iter()
        .map(|&d| d >= threshold && d < (disparity.max_disparity as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disparity_map() {
        let mut disp = DisparityMap::new(10, 10, 0, 64);

        disp.set(5, 5, 32.0);
        assert_eq!(disp.get(5, 5), 32.0);

        let img = disp.to_image();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
    }

    #[test]
    fn test_stereo_params() {
        let params = StereoParams::new(500.0, 0.1, 320.0, 240.0);

        // Test depth computation: depth = (f * B) / disparity
        let disparity = 50.0;
        let expected_depth = (500.0 * 0.1) / 50.0;

        assert_eq!(params.disparity_to_depth(disparity), Some(expected_depth));
        assert_eq!(params.disparity_to_depth(0.0), None);
    }
}
