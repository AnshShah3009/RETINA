//! Video processing algorithms for motion estimation and object tracking
//!
//! This crate provides comprehensive video processing capabilities:
//! - **Optical Flow**: Motion estimation (Lucas-Kanade, dense methods)
//! - **Background Subtraction**: Foreground/background segmentation (MOG2)
//! - **Tracking**: Single and multi-object tracking
//! - **Kalman Filtering**: State estimation and prediction
//!
//! # Key Types
//!
//! - [`VideoFrame`]: Container for video frames with timing information
//! - [`MotionField`]: Dense optical flow representation
//! - [`mog2::Mog2`]: Mixture of Gaussians background model
//!
//! # Example: Background Subtraction
//!
//! ```no_run
//! # use cv_video::mog2::Mog2;
//! # use cv_hal::cpu::CpuBackend;
//! # use cv_hal::compute::ComputeDevice;
//! let cpu = CpuBackend::new().unwrap();
//! let mut mog2 = Mog2::new(100, 16.0, false);
//! // Process video frames to extract foreground
//! ```

use image::GrayImage;

/// Mixture of Gaussians background subtraction
pub mod mog2;
/// Optical flow computation algorithms
pub mod optical_flow;
/// Object tracking algorithms
pub mod tracking;

pub use cv_core::{Error, Result};
pub use optical_flow::*;
pub use tracking::*;

// Re-export from cv-core (Kalman is a general estimation primitive)
/// Backwards-compatible alias
pub use cv_core::kalman;
pub use cv_core::kalman::{
    DynamicKalmanFilter, ExtendedKalmanFilter, KalmanFilter, KalmanFilterState,
};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type VideoError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type VideoResult<T> = cv_core::Result<T>;

/// Single video frame with timing information
///
/// Container for a video frame with associated metadata
/// including timestamp and sequence number for temporal tracking.
///
/// # Fields
///
/// * `image` - Grayscale frame data (8-bit unsigned)
/// * `timestamp` - Frame capture time in seconds (e.g., from video codec)
/// * `frame_number` - Sequence number (0-indexed)
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Grayscale frame image data
    pub image: GrayImage,
    /// Frame timestamp in seconds (from video source)
    pub timestamp: f64,
    /// Sequence frame number (0-indexed from stream start)
    pub frame_number: usize,
}

impl VideoFrame {
    /// Create a new video frame
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale frame image
    /// * `timestamp` - Frame time in seconds
    /// * `frame_number` - Sequence index
    pub fn new(image: GrayImage, timestamp: f64, frame_number: usize) -> Self {
        Self {
            image,
            timestamp,
            frame_number,
        }
    }
}

/// Dense motion field representation
///
/// Stores horizontal (u) and vertical (v) motion components at each pixel
/// as computed by optical flow algorithms. Provides efficient lookup and
/// visualization utilities.
///
/// # Storage Format
///
/// Motion vectors are stored in row-major order: `index = y * width + x`
/// Both u and v components use the same indexing scheme.
///
/// # Example
///
/// ```
/// # use cv_video::MotionField;
/// let mut field = MotionField::new(640, 480);
/// field.set_motion(100, 100, 1.5, -2.0);  // (u=1.5, v=-2.0) at (100,100)
/// let (u, v) = field.get_motion(100, 100);
/// assert_eq!(u, 1.5);
/// assert_eq!(v, -2.0);
/// ```
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Horizontal motion component (u) at each pixel
    pub u: Vec<f32>,
    /// Vertical motion component (v) at each pixel
    pub v: Vec<f32>,
    /// Motion field width in pixels
    pub width: u32,
    /// Motion field height in pixels
    pub height: u32,
}

impl MotionField {
    /// Create a new motion field
    ///
    /// Initializes u and v components to zero.
    ///
    /// # Arguments
    ///
    /// * `width` - Field width in pixels
    /// * `height` - Field height in pixels
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            u: vec![0.0; size],
            v: vec![0.0; size],
            width,
            height,
        }
    }

    /// Get motion vector at pixel location
    ///
    /// # Arguments
    ///
    /// * `x` - Pixel x-coordinate (0 to width-1)
    /// * `y` - Pixel y-coordinate (0 to height-1)
    ///
    /// # Returns
    ///
    /// (u, v) horizontal and vertical motion components (in pixels per frame)
    ///
    /// # Panics
    ///
    /// If coordinates are out of bounds
    pub fn get_motion(&self, x: u32, y: u32) -> (f32, f32) {
        let idx = (y * self.width + x) as usize;
        (self.u[idx], self.v[idx])
    }

    /// Set motion vector at pixel location
    ///
    /// # Arguments
    ///
    /// * `x`, `y` - Pixel coordinates
    /// * `u` - Horizontal motion component
    /// * `v` - Vertical motion component
    ///
    /// # Panics
    ///
    /// If coordinates are out of bounds
    pub fn set_motion(&mut self, x: u32, y: u32, u: f32, v: f32) {
        let idx = (y * self.width + x) as usize;
        self.u[idx] = u;
        self.v[idx] = v;
    }

    /// Compute motion magnitude (speed) at each pixel
    ///
    /// # Returns
    ///
    /// Vector of magnitude values: `sqrt(u^2 + v^2)` at each pixel
    pub fn magnitude(&self) -> Vec<f32> {
        self.u
            .iter()
            .zip(self.v.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect()
    }

    /// Visualize motion field as RGB image
    ///
    /// Creates a color-coded visualization where:
    /// - **Hue**: Direction of motion (0-360 degrees)
    /// - **Saturation**: Always maximum (fully saturated color)
    /// - **Value**: Magnitude of motion (normalized to max)
    ///
    /// # Returns
    ///
    /// RGB image with motion encoded in HSV color space
    ///
    /// # Visualization
    ///
    /// - Red/Yellow: Rightward motion
    /// - Green/Cyan: Upward motion
    /// - Blue/Magenta: Leftward/downward motion
    /// - Brightness: Strength of motion
    pub fn visualize(&self) -> image::RgbImage {
        use image::Rgb;

        let mut img = image::RgbImage::new(self.width, self.height);

        // Find max magnitude for normalization
        let max_mag = self.magnitude().into_iter().fold(0.0f32, f32::max).max(1.0);

        for y in 0..self.height {
            for x in 0..self.width {
                let (u, v) = self.get_motion(x, y);

                // Compute angle and magnitude
                let angle = v.atan2(u); // -PI to PI
                let magnitude = (u * u + v * v).sqrt();

                // Convert to HSV: Hue = angle, Saturation = 1.0, Value = magnitude
                let hue =
                    ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 255.0) as u8;
                let saturation = 255u8;
                let value = ((magnitude / max_mag) * 255.0) as u8;

                // Simple HSV to RGB conversion
                let rgb = hsv_to_rgb(hue, saturation, value);
                img.put_pixel(x, y, Rgb(rgb));
            }
        }

        img
    }
}

/// Simple HSV to RGB conversion
fn hsv_to_rgb(h: u8, s: u8, v: u8) -> [u8; 3] {
    let h = h as f32 / 255.0 * 360.0;
    let s = s as f32 / 255.0;
    let v = v as f32 / 255.0;

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_field() {
        let mut field = MotionField::new(10, 10);

        field.set_motion(5, 5, 3.0, 4.0);
        let (u, v) = field.get_motion(5, 5);

        assert_eq!(u, 3.0);
        assert_eq!(v, 4.0);

        let mag = field.magnitude();
        assert_eq!(mag[55], 5.0); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_video_frame() {
        let img = GrayImage::new(100, 100);
        let frame = VideoFrame::new(img, 0.0, 0);

        assert_eq!(frame.frame_number, 0);
        assert_eq!(frame.timestamp, 0.0);
    }
}
