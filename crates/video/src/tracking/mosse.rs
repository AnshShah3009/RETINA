use cv_core::{CpuTensor, Float};

use super::{
    complex_conj, complex_div, complex_mul, create_cosine_window, create_gaussian_target,
    extract_patch, fft2, ifft2, preprocess_patch, BoundingBox, Complex, ObjectTracker,
};

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
