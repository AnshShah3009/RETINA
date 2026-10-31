//! Standard Hough lines as the HAL backends compute it, shared by the CPU
//! backend and the GPU host code so both produce the same lines.
//!
//! Layout is OpenCV's `HoughLinesStandard`: `round(((w + h) * 2 + 1) / rho)`
//! rho bins, bin `k` centred on `(k - (num_rho - 1) / 2) * rho` (integer
//! division), `floor(pi / theta) + 1` angles minus a duplicate of theta = 0.
//! Votes are stored theta-major (`t * num_rho + r`).

/// Accumulator shape for one image / resolution pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoughLayout {
    pub num_rho: usize,
    pub num_theta: usize,
    /// `(num_rho - 1) / 2` (integer division), the bin of rho = 0.
    pub rho_center: usize,
    /// Resolutions after the `1e-6` floor the backends apply.
    pub rho: f32,
    pub theta: f32,
}

impl HoughLayout {
    pub fn new(w: usize, h: usize, rho: f32, theta: f32) -> Self {
        let rho = rho.max(1e-6);
        let theta = theta.max(1e-6);
        let num_rho = (((w + h) * 2 + 1) as f32 / rho).round() as usize;
        let mut num_theta = (std::f32::consts::PI / theta).floor() as usize + 1;
        if num_theta > 1
            && (std::f32::consts::PI - (num_theta - 1) as f32 * theta).abs() < theta / 2.0
        {
            num_theta -= 1;
        }
        Self {
            num_rho,
            num_theta,
            rho_center: num_rho.saturating_sub(1) / 2,
            rho,
            theta,
        }
    }

    /// `cos(t * theta)`, `sin(t * theta)` in f32.
    pub fn trig_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let cos = (0..self.num_theta)
            .map(|t| (t as f32 * self.theta).cos())
            .collect();
        let sin = (0..self.num_theta)
            .map(|t| (t as f32 * self.theta).sin())
            .collect();
        (cos, sin)
    }

    /// Rho bin of pixel (x, y) at angle t, relative to `rho_center`:
    /// `round((x cos + y sin) / rho)`, ties away from zero, all in f32.
    #[inline]
    pub fn bin(&self, x: usize, y: usize, cos: f32, sin: f32) -> f32 {
        let r = x as f32 * cos + y as f32 * sin;
        (r / self.rho).round()
    }

    /// Accumulator index for a bin from [`Self::bin`], if in range.
    #[inline]
    pub fn index(&self, t: usize, k: f32) -> Option<usize> {
        let r_idx = k + self.rho_center as f32;
        if r_idx >= 0.0 && (r_idx as usize) < self.num_rho {
            Some(t * self.num_rho + r_idx as usize)
        } else {
            None
        }
    }

    /// Lines from a filled accumulator: OpenCV's `findLocalMaximums` (a bin
    /// beats its rho and theta neighbours, `>` on the lower side and `>=` on
    /// the upper, out-of-range neighbours 0) for bins with at least
    /// `threshold` votes, strongest first (stable on theta-major order).
    pub fn peaks(&self, accumulator: &[u32], threshold: u32) -> Vec<cv_core::HoughLine> {
        let (num_rho, num_theta) = (self.num_rho, self.num_theta);
        let at = |t: usize, r: usize| accumulator[t * num_rho + r];
        let mut lines = Vec::new();
        for t in 0..num_theta {
            for r in 0..num_rho {
                let votes = at(t, r);
                if votes < threshold {
                    continue;
                }
                let r_lo = if r > 0 { at(t, r - 1) } else { 0 };
                let r_hi = if r + 1 < num_rho { at(t, r + 1) } else { 0 };
                let t_lo = if t > 0 { at(t - 1, r) } else { 0 };
                let t_hi = if t + 1 < num_theta { at(t + 1, r) } else { 0 };
                if votes > r_lo && votes >= r_hi && votes > t_lo && votes >= t_hi {
                    lines.push(cv_core::HoughLine {
                        // Bin r holds rho = (r - (num_rho - 1) / 2) * rho with
                        // *integer* division - what cv2 4.13 reports, also when
                        // num_rho is even.
                        rho: (r as f32 - self.rho_center as f32) * self.rho,
                        theta: t as f32 * self.theta,
                        score: votes,
                    });
                }
            }
        }
        lines.sort_by_key(|l| std::cmp::Reverse(l.score));
        lines
    }
}
