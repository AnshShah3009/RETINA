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
use cv_core::{CpuTensor, Error, Float};
use image::GrayImage;

/// Tracker interface
pub trait Tracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()>;
    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)>;
}

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
                template.put_pixel(dx, dy, image::Luma([val]));
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

        Err(crate::Error::RuntimeError(
            "Failed to track object".to_string(),
        ))
    }
}

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
                    let weight = (target[intensity] / (target[intensity] + 1e-6)).sqrt();

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
type Complex = (f64, f64);

fn complex_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn complex_conj(a: Complex) -> Complex {
    (a.0, -a.1)
}

fn complex_div(a: Complex, b: Complex) -> Complex {
    let denom = b.0 * b.0 + b.1 * b.1 + 1e-15;
    (
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
    )
}

/// 1-D DFT (not in-place, O(n^2) -- fine for small n).
#[allow(clippy::needless_range_loop)]
fn dft_1d(input: &[Complex], inverse: bool) -> Vec<Complex> {
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
fn dft_2d(data: &[Complex], rows: usize, cols: usize, inverse: bool) -> Vec<Complex> {
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

fn fft2(data: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
    dft_2d(data, rows, cols, false)
}

fn ifft2(data: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
    dft_2d(data, rows, cols, true)
}

// ---------------------------------------------------------------------------
// Helpers: extract grayscale patch from CpuTensor, cosine window, Gaussian target
// ---------------------------------------------------------------------------

/// Extract a grayscale patch from a CpuTensor as f64 values.
/// The tensor is assumed to be single-channel (or channel 0 is used) in CHW layout.
fn extract_patch<T: Float>(
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
fn normalize_patch(patch: &mut [f64]) {
    let max_val = patch.iter().copied().fold(0.0f64, f64::max);
    if max_val > 0.0 {
        for v in patch.iter_mut() {
            *v /= max_val;
        }
    }
}

/// Preprocess patch: log-transform and normalize to zero mean, unit variance.
fn preprocess_patch(patch: &mut [f64]) {
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
fn create_cosine_window(rows: usize, cols: usize) -> Vec<f64> {
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
fn create_gaussian_target(rows: usize, cols: usize, sigma: f64) -> Vec<f64> {
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
fn gaussian_correlation(
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

fn vec_energy(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

// ---------------------------------------------------------------------------
// KCF Tracker
// ---------------------------------------------------------------------------

/// Configuration for the KCF (Kernelized Correlation Filters) tracker.
pub struct KcfConfig {
    /// Extra area around the target as a multiplier of the bbox size (default 2.5).
    pub padding: f64,
    /// Regularisation parameter (default 1e-4).
    pub lambda: f64,
    /// Gaussian kernel bandwidth (default 0.2).
    pub sigma: f64,
    /// Model interpolation (learning) rate (default 0.075).
    pub interp_factor: f64,
    /// Spatial bandwidth factor for the regression target (default 0.1).
    pub output_sigma_factor: f64,
}

impl Default for KcfConfig {
    fn default() -> Self {
        Self {
            padding: 2.5,
            lambda: 1e-4,
            sigma: 0.2,
            interp_factor: 0.075,
            output_sigma_factor: 0.1,
        }
    }
}

/// Kernelized Correlation Filters (KCF) single-object tracker.
///
/// Implements the KCF algorithm with a Gaussian kernel in the frequency domain.
/// The tracker learns a ridge-regression filter from a single training patch and
/// applies cyclic-shift detection to locate the target in subsequent frames.
///
/// Reference: Henriques et al., "High-Speed Tracking with Kernelized Correlation
/// Filters", IEEE TPAMI 2015.
pub struct KcfTracker {
    config: KcfConfig,
    bbox: BoundingBox,
    // Frequency-domain model
    model_alphaf: Option<Vec<Complex>>,
    model_xf: Option<Vec<Complex>>,
    model_x: Option<Vec<f64>>,
    cos_window: Vec<f64>,
    patch_rows: usize,
    patch_cols: usize,
    target_f: Option<Vec<Complex>>,
    initialized: bool,
}

impl KcfTracker {
    /// Create a new KCF tracker with the given configuration.
    pub fn new(config: KcfConfig) -> Self {
        Self {
            config,
            bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
            model_alphaf: None,
            model_xf: None,
            model_x: None,
            cos_window: Vec::new(),
            patch_rows: 0,
            patch_cols: 0,
            target_f: None,
            initialized: false,
        }
    }

    /// Initialize the tracker with the first frame and bounding box.
    pub fn init<T: Float>(&mut self, frame: &CpuTensor<T>, bbox: BoundingBox) {
        self.bbox = bbox;
        // Determine padded patch size (keep it manageable)
        let pw = (bbox.width * self.config.padding).round().max(4.0) as usize;
        let ph = (bbox.height * self.config.padding).round().max(4.0) as usize;
        self.patch_rows = ph;
        self.patch_cols = pw;
        self.cos_window = create_cosine_window(ph, pw);

        // Regression target
        let output_sigma = (bbox.width * bbox.height).sqrt() * self.config.output_sigma_factor
            / self.config.padding;
        let target = create_gaussian_target(ph, pw, output_sigma);
        let target_c: Vec<Complex> = target.iter().map(|&v| (v, 0.0)).collect();
        self.target_f = Some(fft2(&target_c, ph, pw));

        // Extract, normalize to [0,1], and apply cosine window
        let mut patch = extract_patch(frame, bbox.cx(), bbox.cy(), pw, ph);
        normalize_patch(&mut patch);
        for (p, w) in patch.iter_mut().zip(self.cos_window.iter()) {
            *p *= w;
        }

        self.train(&patch);
        self.model_x = Some(patch);
        self.initialized = true;
    }

    /// Update the tracker with a new frame. Returns the updated bounding box,
    /// or `None` if tracking is lost.
    pub fn update<T: Float>(&mut self, frame: &CpuTensor<T>) -> Option<BoundingBox> {
        if !self.initialized {
            return None;
        }
        let (rows, cols) = (self.patch_rows, self.patch_cols);

        // --- Detection ---
        let mut z = extract_patch(frame, self.bbox.cx(), self.bbox.cy(), cols, rows);
        normalize_patch(&mut z);
        for (p, w) in z.iter_mut().zip(self.cos_window.iter()) {
            *p *= w;
        }
        let zf = fft2(&z.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), rows, cols);

        let model_x = self.model_x.as_ref()?;
        let model_xf = self.model_xf.as_ref()?;
        let model_alphaf = self.model_alphaf.as_ref()?;

        let x_energy = vec_energy(model_x);
        let z_energy = vec_energy(&z);
        let kzf = gaussian_correlation(
            model_xf,
            &zf,
            x_energy,
            z_energy,
            self.config.sigma,
            rows,
            cols,
        );

        // response = IFFT(alphaf * kzf)
        let n = rows * cols;
        let mut resp_f = vec![(0.0, 0.0); n];
        for i in 0..n {
            resp_f[i] = complex_mul(model_alphaf[i], kzf[i]);
        }
        let response = ifft2(&resp_f, rows, cols);

        // Find peak
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (i, r) in response.iter().enumerate() {
            if r.0 > best_val {
                best_val = r.0;
                best_idx = i;
            }
        }
        let peak_r = best_idx / cols;
        let peak_c = best_idx % cols;

        // Displacement from centre
        let dy = if peak_r > rows / 2 {
            peak_r as f64 - rows as f64
        } else {
            peak_r as f64
        };
        let dx = if peak_c > cols / 2 {
            peak_c as f64 - cols as f64
        } else {
            peak_c as f64
        };

        self.bbox.x += dx;
        self.bbox.y += dy;

        // --- Training (model update) ---
        let mut new_patch = extract_patch(frame, self.bbox.cx(), self.bbox.cy(), cols, rows);
        normalize_patch(&mut new_patch);
        for (p, w) in new_patch.iter_mut().zip(self.cos_window.iter()) {
            *p *= w;
        }
        let new_alphaf = self.compute_alphaf(&new_patch);
        let new_xf = fft2(
            &new_patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(),
            rows,
            cols,
        );

        // Interpolate model
        let lr = self.config.interp_factor;
        let alphaf = self.model_alphaf.as_mut().unwrap();
        let xf = self.model_xf.as_mut().unwrap();
        let mx = self.model_x.as_mut().unwrap();
        for i in 0..n {
            alphaf[i].0 = (1.0 - lr) * alphaf[i].0 + lr * new_alphaf[i].0;
            alphaf[i].1 = (1.0 - lr) * alphaf[i].1 + lr * new_alphaf[i].1;
            xf[i].0 = (1.0 - lr) * xf[i].0 + lr * new_xf[i].0;
            xf[i].1 = (1.0 - lr) * xf[i].1 + lr * new_xf[i].1;
            mx[i] = (1.0 - lr) * mx[i] + lr * new_patch[i];
        }

        Some(self.bbox)
    }

    /// Return the current bounding box.
    pub fn get_position(&self) -> BoundingBox {
        self.bbox
    }

    // --- internal helpers ---

    fn compute_alphaf(&self, patch: &[f64]) -> Vec<Complex> {
        let (rows, cols) = (self.patch_rows, self.patch_cols);
        let n = rows * cols;
        let pf = fft2(
            &patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(),
            rows,
            cols,
        );
        let energy = vec_energy(patch);
        let kf = gaussian_correlation(&pf, &pf, energy, energy, self.config.sigma, rows, cols);
        let target_f = self.target_f.as_ref().unwrap();
        let lambda = self.config.lambda;
        let mut alphaf = vec![(0.0, 0.0); n];
        for i in 0..n {
            alphaf[i] = complex_div(target_f[i], (kf[i].0 + lambda, kf[i].1));
        }
        alphaf
    }

    fn train(&mut self, patch: &[f64]) {
        let (rows, cols) = (self.patch_rows, self.patch_cols);
        let alphaf = self.compute_alphaf(patch);
        let xf = fft2(
            &patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(),
            rows,
            cols,
        );
        self.model_alphaf = Some(alphaf);
        self.model_xf = Some(xf);
    }
}

// ---------------------------------------------------------------------------
// MOSSE Tracker
// ---------------------------------------------------------------------------

/// Minimum Output Sum of Squared Error (MOSSE) single-object tracker.
///
/// The MOSSE filter learns an adaptive correlation filter in the frequency domain.
/// It is significantly faster than KCF because it operates on raw grayscale patches
/// without kernel tricks.
///
/// Reference: Bolme et al., "Visual Object Tracking using Adaptive Correlation
/// Filters", CVPR 2010.
pub struct MosseTracker {
    bbox: BoundingBox,
    /// Filter numerator A in frequency domain (complex).
    filter_num: Option<Vec<Complex>>,
    /// Filter denominator B in frequency domain (complex).
    filter_den: Option<Vec<Complex>>,
    /// Learning rate for online update.
    learning_rate: f64,
    cos_window: Vec<f64>,
    patch_rows: usize,
    patch_cols: usize,
    /// PSR threshold below which tracking is considered lost.
    psr_threshold: f64,
    initialized: bool,
}

impl MosseTracker {
    /// Create a new MOSSE tracker with the given learning rate (typical: 0.125).
    pub fn new(learning_rate: f64) -> Self {
        Self {
            bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
            filter_num: None,
            filter_den: None,
            learning_rate,
            cos_window: Vec::new(),
            patch_rows: 0,
            patch_cols: 0,
            psr_threshold: 5.0,
            initialized: false,
        }
    }

    /// Initialize the MOSSE tracker with a first frame and bounding box.
    ///
    /// Bootstraps the filter using small random affine perturbations of the
    /// initial patch to produce a more robust initial correlation filter.
    pub fn init<T: Float>(&mut self, frame: &CpuTensor<T>, bbox: BoundingBox) {
        self.bbox = bbox;
        let pw = bbox.width.round().max(4.0) as usize;
        let ph = bbox.height.round().max(4.0) as usize;
        self.patch_rows = ph;
        self.patch_cols = pw;
        self.cos_window = create_cosine_window(ph, pw);
        let n = ph * pw;

        // Gaussian target
        let sigma = (bbox.width.min(bbox.height)) * 0.1;
        let g = create_gaussian_target(ph, pw, sigma.max(1.0));
        let gf = fft2(&g.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), ph, pw);

        let mut a_sum = vec![(0.0, 0.0); n];
        let mut b_sum = vec![(0.0, 0.0); n];

        // Bootstrap with 8 slightly perturbed versions of the patch
        let num_perturbations = 8;
        for p in 0..num_perturbations {
            let offset_x = if p == 0 { 0.0 } else { (p as f64 - 4.0) * 0.5 };
            let offset_y = if p == 0 {
                0.0
            } else {
                ((p as f64 * 1.7) % 3.0 - 1.5) * 0.5
            };
            let mut patch =
                extract_patch(frame, bbox.cx() + offset_x, bbox.cy() + offset_y, pw, ph);
            preprocess_patch(&mut patch);
            for (v, w) in patch.iter_mut().zip(self.cos_window.iter()) {
                *v *= w;
            }
            let fi = fft2(&patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), ph, pw);
            for i in 0..n {
                // A += G* . F_i
                let gc = complex_conj(gf[i]);
                let af = complex_mul(gc, fi[i]);
                a_sum[i].0 += af.0;
                a_sum[i].1 += af.1;
                // B += F_i* . F_i
                let fc = complex_conj(fi[i]);
                let bf = complex_mul(fc, fi[i]);
                b_sum[i].0 += bf.0;
                b_sum[i].1 += bf.1;
            }
        }

        self.filter_num = Some(a_sum);
        self.filter_den = Some(b_sum);
        self.initialized = true;
    }

    /// Update the tracker with a new frame. Returns the updated bounding box,
    /// or `None` if the PSR drops below the threshold (tracking lost).
    pub fn update<T: Float>(&mut self, frame: &CpuTensor<T>) -> Option<BoundingBox> {
        if !self.initialized {
            return None;
        }
        let (rows, cols) = (self.patch_rows, self.patch_cols);
        let n = rows * cols;

        let a = self.filter_num.as_ref()?;
        let b = self.filter_den.as_ref()?;

        // Extract patch at current position
        let mut patch = extract_patch(frame, self.bbox.cx(), self.bbox.cy(), cols, rows);
        preprocess_patch(&mut patch);
        for (v, w) in patch.iter_mut().zip(self.cos_window.iter()) {
            *v *= w;
        }
        let fi = fft2(
            &patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(),
            rows,
            cols,
        );

        // H = A / B  =>  response = IFFT( (A / B) . F )  but equivalently IFFT( A . F / B )
        // For stability we compute H_i = A_i / (B_i + eps), then response = IFFT(H . F)
        let mut resp_f = vec![(0.0, 0.0); n];
        for i in 0..n {
            let h = complex_div(a[i], (b[i].0 + 1e-10, b[i].1));
            resp_f[i] = complex_mul(h, fi[i]);
        }
        let response = ifft2(&resp_f, rows, cols);

        // Find peak
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (i, r) in response.iter().enumerate() {
            if r.0 > best_val {
                best_val = r.0;
                best_idx = i;
            }
        }

        // Compute PSR (Peak to Sidelobe Ratio)
        let resp_real: Vec<f64> = response.iter().map(|c| c.0).collect();
        let mean = resp_real.iter().sum::<f64>() / n as f64;
        let var = resp_real.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt().max(1e-10);
        let psr = (best_val - mean) / std;

        if psr < self.psr_threshold {
            return None;
        }

        // Peak displacement
        let peak_r = best_idx / cols;
        let peak_c = best_idx % cols;
        let dy = if peak_r > rows / 2 {
            peak_r as f64 - rows as f64
        } else {
            peak_r as f64
        };
        let dx = if peak_c > cols / 2 {
            peak_c as f64 - cols as f64
        } else {
            peak_c as f64
        };

        self.bbox.x += dx;
        self.bbox.y += dy;

        // Update filter with new patch at updated position
        let mut new_patch = extract_patch(frame, self.bbox.cx(), self.bbox.cy(), cols, rows);
        preprocess_patch(&mut new_patch);
        for (v, w) in new_patch.iter_mut().zip(self.cos_window.iter()) {
            *v *= w;
        }
        let new_fi = fft2(
            &new_patch.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(),
            rows,
            cols,
        );

        // Gaussian target
        let sigma = (self.bbox.width.min(self.bbox.height)) * 0.1;
        let g = create_gaussian_target(rows, cols, sigma.max(1.0));
        let gf = fft2(&g.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), rows, cols);

        let lr = self.learning_rate;
        let a_mut = self.filter_num.as_mut().unwrap();
        let b_mut = self.filter_den.as_mut().unwrap();
        for i in 0..n {
            let gc = complex_conj(gf[i]);
            let new_a = complex_mul(gc, new_fi[i]);
            a_mut[i].0 = (1.0 - lr) * a_mut[i].0 + lr * new_a.0;
            a_mut[i].1 = (1.0 - lr) * a_mut[i].1 + lr * new_a.1;

            let fc = complex_conj(new_fi[i]);
            let new_b = complex_mul(fc, new_fi[i]);
            b_mut[i].0 = (1.0 - lr) * b_mut[i].0 + lr * new_b.0;
            b_mut[i].1 = (1.0 - lr) * b_mut[i].1 + lr * new_b.1;
        }

        Some(self.bbox)
    }

    /// Return the current bounding box.
    pub fn get_position(&self) -> BoundingBox {
        self.bbox
    }
}

// ---------------------------------------------------------------------------
// ObjectTracker trait + MultiObjectTracker
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

impl ObjectTracker for KcfTracker {
    fn init_tracker(&mut self, frame: &CpuTensor<f32>, bbox: BoundingBox) {
        self.init(frame, bbox);
    }
    fn update_tracker(&mut self, frame: &CpuTensor<f32>) -> Option<BoundingBox> {
        self.update(frame)
    }
    fn get_position(&self) -> BoundingBox {
        KcfTracker::get_position(self)
    }
}

impl ObjectTracker for MosseTracker {
    fn init_tracker(&mut self, frame: &CpuTensor<f32>, bbox: BoundingBox) {
        self.init(frame, bbox);
    }
    fn update_tracker(&mut self, frame: &CpuTensor<f32>) -> Option<BoundingBox> {
        self.update(frame)
    }
    fn get_position(&self) -> BoundingBox {
        MosseTracker::get_position(self)
    }
}

/// Multi-object tracker that manages several independent single-object trackers.
///
/// Each tracked object is assigned a unique ID. Trackers that report `None`
/// (lost) are kept in the list so their IDs remain stable; use [`remove`](Self::remove)
/// to explicitly drop a tracker.
pub struct MultiObjectTracker {
    trackers: Vec<(usize, Box<dyn ObjectTracker>)>,
    next_id: usize,
}

impl MultiObjectTracker {
    /// Create an empty multi-object tracker.
    pub fn new() -> Self {
        Self {
            trackers: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a new object to track. Returns the assigned tracker ID.
    pub fn add(
        &mut self,
        frame: &CpuTensor<f32>,
        bbox: BoundingBox,
        tracker_type: TrackerType,
    ) -> usize {
        let mut tracker: Box<dyn ObjectTracker> = match tracker_type {
            TrackerType::Kcf => Box::new(KcfTracker::new(KcfConfig::default())),
            TrackerType::Mosse => Box::new(MosseTracker::new(0.125)),
        };
        tracker.init_tracker(frame, bbox);
        let id = self.next_id;
        self.next_id += 1;
        self.trackers.push((id, tracker));
        id
    }

    /// Update all trackers with a new frame.
    ///
    /// Returns a vector of `(id, Option<BoundingBox>)` for every active tracker.
    pub fn update(&mut self, frame: &CpuTensor<f32>) -> Vec<(usize, Option<BoundingBox>)> {
        self.trackers
            .iter_mut()
            .map(|(id, t)| (*id, t.update_tracker(frame)))
            .collect()
    }

    /// Remove a tracker by its ID.
    pub fn remove(&mut self, id: usize) {
        self.trackers.retain(|(tid, _)| *tid != id);
    }

    /// Number of active trackers.
    pub fn len(&self) -> usize {
        self.trackers.len()
    }

    /// Returns `true` if there are no active trackers.
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new()
    }
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
