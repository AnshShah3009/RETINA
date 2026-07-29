use cv_core::{CpuTensor, Float};

use super::{
    complex_div, complex_mul, create_cosine_window, create_gaussian_target, extract_patch, fft2,
    gaussian_correlation, ifft2, normalize_patch, vec_energy, BoundingBox, Complex, ObjectTracker,
};

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
