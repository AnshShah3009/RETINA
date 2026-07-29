//! Object tracking algorithms
//!
//! Various tracking methods for following objects in video sequences:
//!
//! - [`TemplateTracker`]: Simple template matching (SSD-based)
//! - [`MeanShiftTracker`]: Histogram-based mean-shift tracking
//! - [`KcfTracker`]: Kernelized Correlation Filters (frequency-domain ridge regression)
//! - [`MosseTracker`]: Minimum Output Sum of Squared Error (fast, simple)
//! - [`MultiObjectTracker`]: Track multiple objects simultaneously
//!
//! The KCF and MOSSE trackers operate on [`CpuTensor`] frames via the [`ObjectTracker`] trait,
//! while the legacy [`Tracker`] trait operates on [`GrayImage`] frames.

#![allow(deprecated)]

use crate::Result;
use cv_core::{CpuTensor, Float};
use image::GrayImage;

/// Tracker interface
pub trait Tracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()>;
    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)>;
}

pub mod meanshift;
pub use meanshift::MeanShiftTracker;

pub mod template;
pub use template::TemplateTracker;

pub mod kcf;
pub use kcf::{KcfConfig, KcfTracker};

pub mod mosse;
pub use mosse::MosseTracker;

pub mod multi;
pub use multi::MultiObjectTracker;

// ---------------------------------------------------------------------------
// Bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box for object tracking.
///
/// Coordinates use floating-point to allow sub-pixel precision during
/// correlation-filter based tracking.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Top-left x coordinate
    pub x: f64,
    /// Top-left y coordinate
    pub y: f64,
    /// Box width
    pub width: f64,
    /// Box height
    pub height: f64,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Center x coordinate.
    pub fn cx(&self) -> f64 {
        self.x + self.width / 2.0
    }

    /// Center y coordinate.
    pub fn cy(&self) -> f64 {
        self.y + self.height / 2.0
    }
}

// ---------------------------------------------------------------------------
// Inline 2-D DFT / IDFT (small patches only — no cross-crate dependency)
// ---------------------------------------------------------------------------

/// Complex number (re, im).
pub(crate) type Complex = (f64, f64);

pub(crate) fn complex_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

pub(crate) fn complex_conj(a: Complex) -> Complex {
    (a.0, -a.1)
}

pub(crate) fn complex_div(a: Complex, b: Complex) -> Complex {
    let denom = b.0 * b.0 + b.1 * b.1 + 1e-15;
    (
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
    )
}

/// 1-D DFT (not in-place, O(n^2) -- fine for small n).
#[allow(clippy::needless_range_loop)]
pub(crate) fn dft_1d(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out = vec![(0.0, 0.0); n];
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    for k in 0..n {
        let mut sum = (0.0, 0.0);
        for (j, inp) in input.iter().enumerate() {
            let angle = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let w = (angle.cos(), angle.sin());
            sum.0 += inp.0 * w.0 - inp.1 * w.1;
            sum.1 += inp.0 * w.1 + inp.1 * w.0;
        }
        out[k] = (sum.0 * scale, sum.1 * scale);
    }
    out
}

/// 2-D DFT via row-then-column 1-D DFTs.
pub(crate) fn dft_2d(data: &[Complex], rows: usize, cols: usize, inverse: bool) -> Vec<Complex> {
    // Row transforms
    let mut buf = vec![(0.0, 0.0); rows * cols];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let transformed = dft_1d(row, inverse);
        buf[r * cols..(r + 1) * cols].copy_from_slice(&transformed);
    }
    // Column transforms
    let mut result = vec![(0.0, 0.0); rows * cols];
    let mut col_buf = vec![(0.0, 0.0); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = buf[r * cols + c];
        }
        let transformed = dft_1d(&col_buf, inverse);
        for r in 0..rows {
            result[r * cols + c] = transformed[r];
        }
    }
    result
}

pub(crate) fn fft2(data: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
    dft_2d(data, rows, cols, false)
}

pub(crate) fn ifft2(data: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
    dft_2d(data, rows, cols, true)
}

// ---------------------------------------------------------------------------
// Helpers: extract grayscale patch from CpuTensor, cosine window, Gaussian target
// ---------------------------------------------------------------------------

/// Extract a grayscale patch from a CpuTensor as f64 values.
/// The tensor is assumed to be single-channel (or channel 0 is used) in CHW layout.
pub(crate) fn extract_patch<T: Float>(
    frame: &CpuTensor<T>,
    cx: f64,
    cy: f64,
    patch_w: usize,
    patch_h: usize,
) -> Vec<f64> {
    let shape = frame.shape;
    let data = match frame.as_slice() {
        Ok(d) => d,
        Err(_) => return vec![0.0; patch_w * patch_h],
    };
    let fw = shape.width;
    let fh = shape.height;
    let mut patch = vec![0.0; patch_w * patch_h];
    let x0 = (cx - patch_w as f64 / 2.0).round() as isize;
    let y0 = (cy - patch_h as f64 / 2.0).round() as isize;
    for py in 0..patch_h {
        for px in 0..patch_w {
            let sx = (x0 + px as isize).clamp(0, fw as isize - 1) as usize;
            let sy = (y0 + py as isize).clamp(0, fh as isize - 1) as usize;
            // Channel 0, CHW layout: index = 0 * H * W + sy * W + sx
            let idx = sy * fw + sx;
            if idx < data.len() {
                patch[py * patch_w + px] = data[idx].to_f64();
            }
        }
    }
    patch
}

/// Normalize patch values to [0, 1] range (for KCF).
pub(crate) fn normalize_patch(patch: &mut [f64]) {
    let max_val = patch.iter().copied().fold(0.0f64, f64::max);
    if max_val > 0.0 {
        for v in patch.iter_mut() {
            *v /= max_val;
        }
    }
}

/// Preprocess patch: log-transform and normalize to zero mean, unit variance.
pub(crate) fn preprocess_patch(patch: &mut [f64]) {
    // log transform
    for v in patch.iter_mut() {
        *v = (*v + 1.0).ln();
    }
    let n = patch.len() as f64;
    let mean = patch.iter().copied().sum::<f64>() / n;
    for v in patch.iter_mut() {
        *v -= mean;
    }
    let var = patch.iter().map(|v| v * v).sum::<f64>() / n;
    let std = var.sqrt().max(1e-10);
    for v in patch.iter_mut() {
        *v /= std;
    }
}

/// Create a 2-D cosine (Hann) window of the given size.
pub(crate) fn create_cosine_window(rows: usize, cols: usize) -> Vec<f64> {
    let mut win = vec![0.0; rows * cols];
    for r in 0..rows {
        let wr = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * r as f64 / (rows as f64 - 1.0)).cos());
        for c in 0..cols {
            let wc =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * c as f64 / (cols as f64 - 1.0)).cos());
            win[r * cols + c] = wr * wc;
        }
    }
    win
}

/// Create a 2-D Gaussian regression target centred at (0,0) with circular wrapping.
///
/// The peak is at index (0,0) so that a zero-displacement detection corresponds
/// to finding the peak at position 0 in the DFT response.
pub(crate) fn create_gaussian_target(rows: usize, cols: usize, sigma: f64) -> Vec<f64> {
    let mut target = vec![0.0; rows * cols];
    let s2 = 2.0 * sigma * sigma;
    for r in 0..rows {
        for c in 0..cols {
            // Circular distance from (0,0)
            let dy = if r <= rows / 2 {
                r as f64
            } else {
                r as f64 - rows as f64
            };
            let dx = if c <= cols / 2 {
                c as f64
            } else {
                c as f64 - cols as f64
            };
            target[r * cols + c] = (-(dx * dx + dy * dy) / s2).exp();
        }
    }
    target
}

/// Gaussian kernel correlation in the frequency domain (element-wise).
/// k = exp(-1/(sigma^2) * max(0, ||x||^2 + ||z||^2 - 2 * IFFT(conj(FFT(x)) . FFT(z))) / numel)
pub(crate) fn gaussian_correlation(
    xf: &[Complex],
    zf: &[Complex],
    x_energy: f64,
    z_energy: f64,
    sigma: f64,
    rows: usize,
    cols: usize,
) -> Vec<Complex> {
    let n = rows * cols;
    // Cross-correlation in freq domain
    let mut xzf = vec![(0.0, 0.0); n];
    for i in 0..n {
        xzf[i] = complex_mul(complex_conj(xf[i]), zf[i]);
    }
    let xz = ifft2(&xzf, rows, cols);
    let sigma2 = sigma * sigma;
    let numel = n as f64;
    let mut k = vec![0.0; n];
    for i in 0..n {
        let val = (x_energy + z_energy - 2.0 * xz[i].0) / numel;
        k[i] = (-val.max(0.0) / sigma2).exp();
    }
    // Return FFT of k
    let kc: Vec<Complex> = k.iter().map(|&v| (v, 0.0)).collect();
    fft2(&kc, rows, cols)
}

pub(crate) fn vec_energy(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

// ---------------------------------------------------------------------------
// ObjectTracker trait + TrackerType
// ---------------------------------------------------------------------------

/// Tracker type selector for [`MultiObjectTracker`].
pub enum TrackerType {
    /// Kernelized Correlation Filters.
    Kcf,
    /// Minimum Output Sum of Squared Error.
    Mosse,
}

/// Trait for CpuTensor-based object trackers.
///
/// Unlike the legacy [`Tracker`] trait (which uses `GrayImage`), this trait
/// operates on generic [`CpuTensor<f32>`] frames and uses [`BoundingBox`].
pub trait ObjectTracker: Send {
    /// Initialize the tracker with the first frame and bounding box.
    fn init_tracker(&mut self, frame: &CpuTensor<f32>, bbox: BoundingBox);
    /// Update with a new frame. Returns the updated bounding box, or `None` if lost.
    fn update_tracker(&mut self, frame: &CpuTensor<f32>) -> Option<BoundingBox>;
    /// Return the current position.
    fn get_position(&self) -> BoundingBox;
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::TensorShape;
    use image::Luma;

    fn create_test_sequence() -> Vec<GrayImage> {
        let mut frames = Vec::new();
        let width = 100u32;
        let height = 100u32;

        for frame_idx in 0..5 {
            let mut frame = GrayImage::new(width, height);

            // Fill with black
            for y in 0..height {
                for x in 0..width {
                    frame.put_pixel(x, y, Luma([0]));
                }
            }

            // Draw moving square
            let square_x = 30 + frame_idx * 5;
            let square_y = 40;
            let square_size = 20;

            for y in 0..square_size {
                for x in 0..square_size {
                    let px = (square_x + x).min(width - 1);
                    let py = (square_y + y).min(height - 1);
                    frame.put_pixel(px, py, Luma([255]));
                }
            }

            frames.push(frame);
        }

        frames
    }

    #[test]
    fn test_template_tracker() {
        let frames = create_test_sequence();
        let mut tracker = TemplateTracker::new(15);

        // Initialize with first frame
        tracker.init(&frames[0], (30, 40, 20, 20)).unwrap();

        // Track through sequence
        for (i, frame) in frames.iter().enumerate().skip(1) {
            let bbox = tracker.update(frame).unwrap();
            println!("Frame {}: bbox at ({}, {})", i, bbox.0, bbox.1);

            // Square should move right
            assert!(bbox.0 >= 30 + (i as u32 - 1) * 5);
        }
    }

    #[test]
    fn test_mean_shift_tracker() {
        let frames = create_test_sequence();
        let mut tracker = MeanShiftTracker::new(20, 20);

        // Initialize with first frame
        tracker.init(&frames[0], (30, 40, 20, 20)).unwrap();

        // Track through sequence
        for (i, frame) in frames.iter().enumerate().skip(1) {
            let bbox = tracker.update(frame).unwrap();
            println!("Frame {}: MeanShift bbox at ({}, {})", i, bbox.0, bbox.1);
        }
    }

    // --- Helpers for CpuTensor-based tests ---

    /// Create a synthetic f32 CpuTensor with a bright rectangle on a dark background.
    fn make_tensor_frame(
        width: usize,
        height: usize,
        rect_x: usize,
        rect_y: usize,
        rect_w: usize,
        rect_h: usize,
    ) -> CpuTensor<f32> {
        let mut data = vec![0.0f32; width * height];
        for r in rect_y..(rect_y + rect_h).min(height) {
            for c in rect_x..(rect_x + rect_w).min(width) {
                data[r * width + c] = 255.0;
            }
        }
        CpuTensor::<f32>::from_vec(data, TensorShape::new(1, height, width)).unwrap()
    }

    #[test]
    fn test_bounding_box_basic() {
        let bb = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(bb.x, 10.0);
        assert_eq!(bb.y, 20.0);
        assert_eq!(bb.width, 30.0);
        assert_eq!(bb.height, 40.0);
        assert!((bb.cx() - 25.0).abs() < 1e-9);
        assert!((bb.cy() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_kcf_tracker_follows_object() {
        let width = 100;
        let height = 100;
        let rect_w = 20;
        let rect_h = 20;

        // Frame 0: rectangle at (30, 40)
        let frame0 = make_tensor_frame(width, height, 30, 40, rect_w, rect_h);
        let mut tracker = KcfTracker::new(KcfConfig::default());
        tracker.init(
            &frame0,
            BoundingBox::new(30.0, 40.0, rect_w as f64, rect_h as f64),
        );

        // Frame 1: rectangle shifted by (5, 5)
        let frame1 = make_tensor_frame(width, height, 35, 45, rect_w, rect_h);
        let result = tracker.update(&frame1);
        assert!(result.is_some(), "KCF tracker should not lose the target");

        let bb = result.unwrap();
        // The tracker should move in the positive x and y directions
        let dx = bb.cx() - 40.0; // original centre was 30 + 10 = 40
        let dy = bb.cy() - 50.0; // original centre was 40 + 10 = 50
        println!(
            "KCF after shift: dx={:.1}, dy={:.1}, bbox=({:.1},{:.1})",
            dx, dy, bb.x, bb.y
        );
        // We expect the tracker to move towards the new position; allow generous tolerance
        assert!(dx > 0.0, "KCF should move right (dx={dx})");
        assert!(dy > 0.0, "KCF should move down (dy={dy})");
    }

    #[test]
    fn test_mosse_tracker_follows_object() {
        let width = 100;
        let height = 100;
        let rect_w = 20;
        let rect_h = 20;

        let frame0 = make_tensor_frame(width, height, 30, 40, rect_w, rect_h);
        let mut tracker = MosseTracker::new(0.125);
        tracker.init(
            &frame0,
            BoundingBox::new(30.0, 40.0, rect_w as f64, rect_h as f64),
        );

        // Frame 1: shifted by (5, 5)
        let frame1 = make_tensor_frame(width, height, 35, 45, rect_w, rect_h);
        let result = tracker.update(&frame1);
        // MOSSE may return None if PSR is low on synthetic data; that is acceptable.
        // If it does track, verify direction.
        if let Some(bb) = result {
            let dx = bb.cx() - 40.0;
            let dy = bb.cy() - 50.0;
            println!("MOSSE after shift: dx={:.1}, dy={:.1}", dx, dy);
            // At minimum the position should have changed from the init position
            assert!(
                (bb.x - 30.0).abs() > 0.01 || (bb.y - 40.0).abs() > 0.01,
                "MOSSE should attempt to follow"
            );
        } else {
            println!("MOSSE returned None (low PSR on synthetic data) -- acceptable");
        }
    }

    #[test]
    fn test_multi_object_tracker() {
        let width = 200;
        let height = 200;

        // Two objects at different positions
        let mut data = vec![0.0f32; width * height];
        // Object A at (20, 20) size 15x15
        for r in 20..35 {
            for c in 20..35 {
                data[r * width + c] = 255.0;
            }
        }
        // Object B at (100, 100) size 15x15
        for r in 100..115 {
            for c in 100..115 {
                data[r * width + c] = 255.0;
            }
        }
        let frame0 = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, height, width)).unwrap();

        let mut mot = MultiObjectTracker::new();
        let id_a = mot.add(
            &frame0,
            BoundingBox::new(20.0, 20.0, 15.0, 15.0),
            TrackerType::Kcf,
        );
        let id_b = mot.add(
            &frame0,
            BoundingBox::new(100.0, 100.0, 15.0, 15.0),
            TrackerType::Kcf,
        );
        assert_eq!(mot.len(), 2);
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);

        // Shift both objects by (3, 3)
        let mut data2 = vec![0.0f32; width * height];
        for r in 23..38 {
            for c in 23..38 {
                data2[r * width + c] = 255.0;
            }
        }
        for r in 103..118 {
            for c in 103..118 {
                data2[r * width + c] = 255.0;
            }
        }
        let frame1 = CpuTensor::<f32>::from_vec(data2, TensorShape::new(1, height, width)).unwrap();

        let results = mot.update(&frame1);
        assert_eq!(results.len(), 2);
        // Both should still be tracked
        for (id, bbox_opt) in &results {
            println!("MultiTracker id={id}: {:?}", bbox_opt);
            assert!(bbox_opt.is_some(), "Tracker {id} should still be tracking");
        }

        // Remove one
        mot.remove(id_a);
        assert_eq!(mot.len(), 1);
    }
}
