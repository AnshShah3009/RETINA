//! Optical Flow Algorithms
//!
//! Implements various optical flow estimation methods:
//! - Lucas-Kanade: Sparse optical flow for corner tracking
//! - Farneback: Dense optical flow using polynomial expansion
//! - Horn-Schunck: Global smoothness constraint

use crate::{MotionField, Result};
use cv_core::{Error, KeyPoint};
use image::GrayImage;
use nalgebra::{Matrix2, Vector2};

/// Lucas-Kanade optical flow tracker
///
/// Sparse optical flow that tracks feature points between frames
pub struct LucasKanade {
    pub window_size: usize,
    pub max_iterations: usize,
    pub epsilon: f32,
    pub pyramid_levels: usize,
}

impl Default for LucasKanade {
    fn default() -> Self {
        Self {
            window_size: 21,
            max_iterations: 30,
            epsilon: 0.01,
            pyramid_levels: 3,
        }
    }
}

impl LucasKanade {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    /// Track a single point using pyramidal iterative Lucas-Kanade (Bouguet's
    /// algorithm, as in OpenCV's `calcOpticalFlowPyrLK`).
    ///
    /// Builds Gaussian image pyramids for both frames and performs coarse-to-fine
    /// tracking with iterative refinement at each pyramid level. To track many
    /// points use [`Self::track_points`], which builds the pyramids once.
    ///
    /// Returns `None` when the frames are not the same size. Level `L` of the
    /// pyramid is `frame / 2^L`, so frames of different sizes give two pyramids
    /// that are *different scalings* of the scene: the coordinate `p / 2^L`
    /// names one point in the previous frame and a different point in the next,
    /// and the answer is meaningless rather than merely imprecise. Frames of
    /// unequal size reach this from a capture whose resolution changes
    /// mid-stream (`cv-videoio`'s PNG-sequence backend reads every image in a
    /// directory, whatever its size), so the mismatch is declined the same way
    /// any other untrackable feature is.
    pub fn track_point(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        point: (f32, f32),
    ) -> Option<(f32, f32)> {
        if !frames_match(prev_frame, next_frame) {
            return None;
        }
        let prev_pyr = build_lk_pyramid(prev_frame, self.pyramid_levels, true);
        let next_pyr = build_lk_pyramid(next_frame, self.pyramid_levels, false);
        self.track_in_pyramids(&prev_pyr, &next_pyr, point)
    }

    /// Bouguet's pyramidal LK for one feature on prebuilt pyramids.
    ///
    /// At level `L` the feature sits at `p / 2^L`. The template window of the
    /// previous frame (intensity and gradients) is sampled **bilinearly** around
    /// that fractional point; the next frame is sampled bilinearly at
    /// `p / 2^L + g + v`, where `g` is the guess carried from the coarser level
    /// and `v` the in-level residual. Each solve updates `v`; the guess passes
    /// down as `g = 2 (g + v)`.
    ///
    /// The template used to be sampled at the integer `floor(p) + d` while the
    /// next frame was sampled at the fractional `p + flow + d`, so the
    /// fractional part of the point was lost: tracking against an identical
    /// frame moved (60.5, 60.5) and (60.75, 60.25) to (60, 60).
    fn track_in_pyramids(
        &self,
        prev_pyr: &[LkLevel],
        next_pyr: &[LkLevel],
        point: (f32, f32),
    ) -> Option<(f32, f32)> {
        let (px, py) = (point.0 as f64, point.1 as f64);
        let half_window = (self.window_size / 2) as i32;
        let r = half_window as f64;

        // The window must fit at the finest level.
        if !prev_pyr[0].window_fits(px, py, r) {
            return None;
        }

        let eps2 = (self.epsilon as f64) * (self.epsilon as f64);
        let n = (2 * half_window + 1) as usize;
        let mut tmpl: Vec<(f64, f64, f64)> = Vec::with_capacity(n * n);
        let mut g = Vector2::<f64>::zeros();

        for level in (0..prev_pyr.len()).rev() {
            let prev = &prev_pyr[level];
            let next = &next_pyr[level];
            let scale = 1.0 / (1u64 << level) as f64;
            let (lx, ly) = (px * scale, py * scale);

            let mut v = Vector2::<f64>::zeros();
            let tracked = 'level: {
                if !prev.window_fits(lx, ly, r) {
                    break 'level false;
                }

                // Template and structure tensor at the feature's own
                // (fractional) position.
                tmpl.clear();
                let mut gm: Matrix2<f64> = Matrix2::zeros();
                for dy in -half_window..=half_window {
                    for dx in -half_window..=half_window {
                        let sx = lx + dx as f64;
                        let sy = ly + dy as f64;
                        let i0 = bilinear(&prev.img, prev.w, prev.h, sx, sy);
                        let ix = bilinear(&prev.gx, prev.w, prev.h, sx, sy);
                        let iy = bilinear(&prev.gy, prev.w, prev.h, sx, sy);
                        gm[(0, 0)] += ix * ix;
                        gm[(0, 1)] += ix * iy;
                        gm[(1, 1)] += iy * iy;
                        tmpl.push((i0, ix, iy));
                    }
                }
                gm[(1, 0)] = gm[(0, 1)];

                // Singularity check via the smaller eigenvalue of G.
                let trace = gm[(0, 0)] + gm[(1, 1)];
                let det = gm[(0, 0)] * gm[(1, 1)] - gm[(0, 1)] * gm[(1, 0)];
                let discriminant = trace * trace - 4.0 * det;
                if discriminant < 0.0 || det < 1e-7 {
                    break 'level false;
                }
                let min_eigenvalue = (trace - discriminant.sqrt()) / 2.0;
                if min_eigenvalue < 1e-4 {
                    break 'level false;
                }
                let Some(g_inv) = gm.try_inverse() else {
                    break 'level false;
                };

                let mut prev_delta = Vector2::<f64>::zeros();
                for iter in 0..self.max_iterations {
                    let cx = lx + g[0] + v[0];
                    let cy = ly + g[1] + v[1];
                    if !(cx.is_finite() && cy.is_finite()) {
                        break;
                    }
                    let mut b: Vector2<f64> = Vector2::zeros();
                    let mut k = 0;
                    for dy in -half_window..=half_window {
                        for dx in -half_window..=half_window {
                            let j =
                                bilinear(&next.img, next.w, next.h, cx + dx as f64, cy + dy as f64);
                            let (i0, ix, iy) = tmpl[k];
                            let it = j - i0;
                            b[0] -= ix * it;
                            b[1] -= iy * it;
                            k += 1;
                        }
                    }
                    let delta = g_inv * b;
                    v += delta;
                    if delta.norm_squared() < eps2 {
                        break;
                    }
                    // OpenCV's oscillation guard: a step that undoes the
                    // previous one means the minimum lies between them.
                    if iter > 0
                        && (delta[0] + prev_delta[0]).abs() < 0.01
                        && (delta[1] + prev_delta[1]).abs() < 0.01
                    {
                        v -= delta * 0.5;
                        break;
                    }
                    prev_delta = delta;
                }
                true
            };

            if !tracked && level == 0 {
                return None;
            }
            g += v;
            if level > 0 {
                g *= 2.0;
            }
        }

        let new_x = px + g[0];
        let new_y = py + g[1];
        let (w, h) = (next_pyr[0].w as f64, next_pyr[0].h as f64);

        // Reject if the tracked point falls outside the image
        if !(new_x >= 0.0 && new_x < w && new_y >= 0.0 && new_y < h) {
            return None;
        }

        Some((new_x as f32, new_y as f32))
    }

    /// Single-pass (non-iterative, non-pyramidal) Lucas-Kanade tracking.
    ///
    /// This is the original simple implementation kept as a fallback.
    /// It computes a single gradient solve with no pyramid and no iteration.
    ///
    /// Returns `None` when the frames are not the same size: the solve indexes
    /// the next frame at the previous frame's coordinates, so a smaller next
    /// frame was read out of bounds (a panic), and a larger one was compared
    /// against unrelated pixels.
    pub fn track_point_single_pass(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        point: (f32, f32),
    ) -> Option<(f32, f32)> {
        if !frames_match(prev_frame, next_frame) {
            return None;
        }
        let (x, y) = point;
        let half_window = (self.window_size / 2) as i32;

        // Check if point is within valid region
        if x < half_window as f32
            || x >= (prev_frame.width() as i32 - half_window) as f32
            || y < half_window as f32
            || y >= (prev_frame.height() as i32 - half_window) as f32
        {
            return None;
        }

        // Compute spatial gradients in prev_frame
        let mut a: Matrix2<f64> = Matrix2::zeros();
        let mut b: Vector2<f64> = Vector2::zeros();

        for dy in -half_window..=half_window {
            for dx in -half_window..=half_window {
                let px = (x as i32 + dx) as u32;
                let py = (y as i32 + dy) as u32;

                // Compute gradients using central differences
                let ix = compute_gradient_x(prev_frame, px, py);
                let iy = compute_gradient_y(prev_frame, px, py);
                let it = compute_temporal_gradient(prev_frame, next_frame, px, py);

                // Build system: [ix^2   ix*iy] [u] = [-ix*it]
                //               [ix*iy  iy^2 ] [v]   [-iy*it]
                a[(0, 0)] += (ix * ix) as f64;
                a[(0, 1)] += (ix * iy) as f64;
                a[(1, 0)] += (ix * iy) as f64;
                a[(1, 1)] += (iy * iy) as f64;

                b[0] += -(ix * it) as f64;
                b[1] += -(iy * it) as f64;
            }
        }

        // Solve for optical flow: A * [u, v]^T = b
        let a_inv = a.try_inverse()?;
        let flow = a_inv * b;

        Some((x + flow[0] as f32, y + flow[1] as f32))
    }

    /// Track multiple points.
    ///
    /// Both pyramids (and the previous frame's gradients) are built once for the
    /// whole batch, not once per point.
    pub fn track_points(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        points: &[(f32, f32)],
    ) -> Vec<Option<(f32, f32)>> {
        if points.is_empty() {
            return Vec::new();
        }
        // Declined for the same reason as `track_point`: the two pyramids are
        // different scalings of the scene, so no point has a meaning across
        // them.
        if !frames_match(prev_frame, next_frame) {
            return vec![None; points.len()];
        }
        let prev_pyr = build_lk_pyramid(prev_frame, self.pyramid_levels, true);
        let next_pyr = build_lk_pyramid(next_frame, self.pyramid_levels, false);
        points
            .iter()
            .map(|&p| self.track_in_pyramids(&prev_pyr, &next_pyr, p))
            .collect()
    }

    /// Track keypoints (pyramids built once per call, as in [`Self::track_points`]).
    pub fn track_keypoints(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        keypoints: &[KeyPoint],
    ) -> Vec<Option<KeyPoint>> {
        let points: Vec<(f32, f32)> = keypoints
            .iter()
            .map(|kp| (kp.x as f32, kp.y as f32))
            .collect();
        self.track_points(prev_frame, next_frame, &points)
            .into_iter()
            .zip(keypoints)
            .map(|(t, kp)| {
                t.map(|(x, y)| {
                    KeyPoint::new(x as f64, y as f64)
                        .with_size(kp.size)
                        .with_angle(kp.angle)
                })
            })
            .collect()
    }
}

/// Compute x gradient using central differences
fn compute_gradient_x(img: &GrayImage, x: u32, y: u32) -> f32 {
    let width = img.width();
    let xp = (x + 1).min(width - 1);
    let xm = x.saturating_sub(1);

    (img.get_pixel(xp, y)[0] as f32 - img.get_pixel(xm, y)[0] as f32) / 2.0
}

/// Compute y gradient using central differences
fn compute_gradient_y(img: &GrayImage, x: u32, y: u32) -> f32 {
    let height = img.height();
    let yp = (y + 1).min(height - 1);
    let ym = y.saturating_sub(1);

    (img.get_pixel(x, yp)[0] as f32 - img.get_pixel(x, ym)[0] as f32) / 2.0
}

/// Compute temporal gradient (frame difference)
fn compute_temporal_gradient(prev: &GrayImage, next: &GrayImage, x: u32, y: u32) -> f32 {
    next.get_pixel(x, y)[0] as f32 - prev.get_pixel(x, y)[0] as f32
}

/// One pyramid level in f32: intensities plus (for the previous frame) the
/// central-difference gradients, computed once per call and sampled bilinearly.
struct LkLevel {
    w: usize,
    h: usize,
    img: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl LkLevel {
    fn new(w: usize, h: usize, img: Vec<f32>, with_gradients: bool) -> Self {
        let (mut gx, mut gy) = (Vec::new(), Vec::new());
        if with_gradients {
            gx = vec![0.0; w * h];
            gy = vec![0.0; w * h];
            for y in 0..h {
                let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
                for x in 0..w {
                    let (xm, xp) = (x.saturating_sub(1), (x + 1).min(w - 1));
                    gx[y * w + x] = (img[y * w + xp] - img[y * w + xm]) * 0.5;
                    gy[y * w + x] = (img[yp * w + x] - img[ym * w + x]) * 0.5;
                }
            }
        }
        Self { w, h, img, gx, gy }
    }

    /// Whether a `(2r+1)^2` window centred at `(x, y)` lies inside the level.
    fn window_fits(&self, x: f64, y: f64, r: f64) -> bool {
        x - r >= 0.0 && x + r <= (self.w - 1) as f64 && y - r >= 0.0 && y + r <= (self.h - 1) as f64
    }
}

/// Bilinear interpolation with replicated border.
fn bilinear(buf: &[f32], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let x0 = (x.floor() as usize).min(w - 1);
    let y0 = (y.floor() as usize).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let ax = x - x0 as f64;
    let ay = y - y0 as f64;
    let top = buf[y0 * w + x0] as f64 * (1.0 - ax) + buf[y0 * w + x1] as f64 * ax;
    let bot = buf[y1 * w + x0] as f64 * (1.0 - ax) + buf[y1 * w + x1] as f64 * ax;
    top * (1.0 - ay) + bot * ay
}

/// OpenCV-style `pyrDown`: 5-tap binomial `[1 4 6 4 1] / 16` blur with
/// reflect-101 border, keeping even pixels, size `ceil(w/2) x ceil(h/2)`.
/// Pixel `i` of level `L+1` is pixel `2i` of level `L`, so a point `p` at level
/// 0 is exactly `p / 2^L` at level `L`. Kept in f32 to avoid requantising.
fn pyr_down(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let reflect = |i: isize, n: usize| -> usize {
        if n == 1 {
            return 0;
        }
        let n = n as isize;
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * (n - 1) - i;
        }
        i.clamp(0, n - 1) as usize
    };
    let nw = w.div_ceil(2);
    let nh = h.div_ceil(2);
    // Horizontal pass at even columns only.
    let mut tmp = vec![0.0f32; nw * h];
    for y in 0..h {
        for ox in 0..nw {
            let cx = (2 * ox) as isize;
            let mut acc = 0.0;
            for (k, wk) in K.iter().enumerate() {
                acc += wk * src[y * w + reflect(cx + k as isize - 2, w)];
            }
            tmp[y * nw + ox] = acc;
        }
    }
    let mut out = vec![0.0f32; nw * nh];
    for oy in 0..nh {
        let cy = (2 * oy) as isize;
        for x in 0..nw {
            let mut acc = 0.0;
            for (k, wk) in K.iter().enumerate() {
                acc += wk * tmp[reflect(cy + k as isize - 2, h) * nw + x];
            }
            out[oy * nw + x] = acc;
        }
    }
    (out, nw, nh)
}

/// Whether two frames describe a scene at the same scale.
///
/// Sparse LK correlates a window of the previous frame with the next frame at
/// the *same* coordinates, and every pyramid level is a division of the frame
/// by `2^L`. Frames of different sizes therefore give pyramids that are
/// different scalings of the scene: the coordinate `p / 2^L` names one point in
/// the previous frame and another in the next. `Farneback::compute` rejects such
/// a pair with `DimensionMismatch`; sparse LK reports the feature as
/// untrackable, which is the same outcome as any other unusable feature.
///
/// This is reachable from a real capture: `cv-videoio`'s PNG-sequence backend
/// reads every image in a directory, so a directory holding two recordings at
/// different resolutions yields a stream whose frame size changes mid-stream.
fn frames_match(prev_frame: &GrayImage, next_frame: &GrayImage) -> bool {
    prev_frame.dimensions() == next_frame.dimensions()
}

/// Build an f32 Gaussian pyramid, level 0 = finest.
fn build_lk_pyramid(img: &GrayImage, levels: usize, with_gradients: bool) -> Vec<LkLevel> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let base: Vec<f32> = img.as_raw().iter().map(|&v| v as f32).collect();
    let mut pyr = vec![LkLevel::new(w, h, base, with_gradients)];
    for _ in 1..levels.max(1) {
        let last = pyr.last().unwrap();
        let (img, nw, nh) = pyr_down(&last.img, last.w, last.h);
        pyr.push(LkLevel::new(nw, nh, img, with_gradients));
    }
    pyr
}

/// Farneback dense optical flow
///
/// Estimates motion at every pixel using polynomial expansion
pub struct Farneback {
    pub pyramid_scale: f32,
    pub pyramid_levels: usize,
    pub window_size: usize,
    pub iterations: usize,
    pub poly_n: usize,
    pub poly_sigma: f32,
}

impl Default for Farneback {
    fn default() -> Self {
        Self {
            pyramid_scale: 0.5,
            pyramid_levels: 3,
            window_size: 15,
            iterations: 3,
            poly_n: 5,
            poly_sigma: 1.2,
        }
    }
}

impl Farneback {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the Farneback parameters that must not be degenerate.
    ///
    /// * `pyramid_scale` must be finite and in `(0, 1)`. `0.0` collapses every
    ///   pyramid level onto the full-resolution image (no multiscale search);
    ///   `>= 1.0` (or a non-finite value) scales levels *up*, and
    ///   `scale_image` computes a 0-sized image for scale `0.0`.
    /// * `poly_sigma` must be finite and `> 0`. `sigma == 0.0` makes
    ///   `gaussian_weight` evaluate `exp(-r^2 / (2 * 0))`, i.e. `0/0` at the
    ///   `(0, 0)` tap -> `NaN`, which poisons every 6x6 polynomial solve (all
    ///   coefficients fall back to zero) and every windowed least-squares solve,
    ///   yielding an all-zero motion field.
    /// * `pyramid_levels`, `window_size` and `iterations` must be non-zero, and
    ///   `poly_n` must be odd and at least 3 so the polynomial expansion has a
    ///   solvable, centred neighbourhood.
    pub fn validate(&self) -> Result<()> {
        if !(self.pyramid_scale.is_finite() && self.pyramid_scale > 0.0 && self.pyramid_scale < 1.0)
        {
            return Err(Error::InvalidInput(format!(
                "Farneback: pyramid_scale must be finite and in (0, 1), got {}. \
                 0.0 collapses the pyramid and >= 1.0 grows the levels instead of coarsening them",
                self.pyramid_scale
            )));
        }
        if self.pyramid_levels == 0 {
            return Err(Error::InvalidInput(
                "Farneback: pyramid_levels must be greater than 0".to_string(),
            ));
        }
        if self.window_size == 0 {
            return Err(Error::InvalidInput(
                "Farneback: window_size must be greater than 0".to_string(),
            ));
        }
        if self.iterations == 0 {
            return Err(Error::InvalidInput(
                "Farneback: iterations must be greater than 0".to_string(),
            ));
        }
        if self.poly_n < 3 || self.poly_n.is_multiple_of(2) {
            return Err(Error::InvalidInput(format!(
                "Farneback: poly_n must be odd and >= 3, got {}",
                self.poly_n
            )));
        }
        if !(self.poly_sigma.is_finite() && self.poly_sigma > 0.0) {
            return Err(Error::InvalidInput(format!(
                "Farneback: poly_sigma must be finite and > 0, got {}. \
                 sigma == 0 makes the Gaussian weight at the (0,0) tap evaluate 0/0 = NaN, \
                 which turns the whole motion field into zeros",
                self.poly_sigma
            )));
        }
        Ok(())
    }

    pub fn with_pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Compute dense optical flow.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when the parameters are degenerate (see
    /// [`Farneback::validate`]) or [`Error::DimensionMismatch`] when the frames
    /// differ in size.
    pub fn compute(&self, prev_frame: &GrayImage, next_frame: &GrayImage) -> Result<MotionField> {
        self.validate()?;

        if prev_frame.width() != next_frame.width() || prev_frame.height() != next_frame.height() {
            return Err(Error::DimensionMismatch(
                "Frames must have the same dimensions".to_string(),
            ));
        }

        let width = prev_frame.width();
        let height = prev_frame.height();

        // Start with zero motion field
        let mut flow = MotionField::new(width, height);

        // Build Gaussian pyramid
        let mut prev_pyramid = vec![prev_frame.clone()];
        let mut next_pyramid = vec![next_frame.clone()];

        for _ in 1..self.pyramid_levels {
            let prev_scaled = scale_image(prev_pyramid.last().unwrap(), self.pyramid_scale);
            let next_scaled = scale_image(next_pyramid.last().unwrap(), self.pyramid_scale);
            prev_pyramid.push(prev_scaled);
            next_pyramid.push(next_scaled);
        }

        // Process from coarse to fine
        for level in (0..self.pyramid_levels).rev() {
            let level_width = prev_pyramid[level].width();
            let level_height = prev_pyramid[level].height();

            // Scale flow to current level
            if level < self.pyramid_levels - 1 {
                flow = scale_motion_field(&flow, level_width, level_height);
            }

            // Compute polynomial expansion
            let prev_poly =
                polynomial_expansion(&prev_pyramid[level], self.poly_n, self.poly_sigma);
            let next_poly =
                polynomial_expansion(&next_pyramid[level], self.poly_n, self.poly_sigma);

            // Refine flow using iterative method. Each pass samples the
            // next frame at the displacement already estimated, so it adds
            // a residual instead of stacking another copy of the full motion.
            for _ in 0..self.iterations {
                flow = self.refine_flow(&prev_poly, &next_poly, &flow, level_width, level_height);
            }
        }

        Ok(flow)
    }

    fn refine_flow(
        &self,
        prev_poly: &PolynomialExpansion,
        next_poly: &PolynomialExpansion,
        flow: &MotionField,
        width: u32,
        height: u32,
    ) -> MotionField {
        let mut new_flow = MotionField::new(width, height);
        let half_window = (self.window_size / 2) as i32;
        // Weight across the flow window, not the polynomial neighbourhood.
        // `poly_sigma` (~1.2) zeros every tap past a couple of pixels, so
        // `window_size` never actually entered the fit.
        let win_sigma = (self.window_size as f32) * 0.3;
        let win_side = (2 * half_window + 1) as usize;
        let mut win_weight = vec![0.0f64; win_side * win_side];
        for dy in -half_window..=half_window {
            for dx in -half_window..=half_window {
                let i = ((dy + half_window) as usize) * win_side + (dx + half_window) as usize;
                win_weight[i] = gaussian_weight(dx as f32, dy as f32, win_sigma) as f64;
            }
        }

        for y in 0..height {
            for x in 0..width {
                let (u_prev, v_prev) = flow.get_motion(x, y);

                // Correct Farneback formulation:
                // For each pixel, build A from quadratic coefficients and b from linear terms.
                // A_k = [[2*coeff.0, coeff.1], [coeff.1, 2*coeff.2]]
                // b_k = [coeff.3, coeff.4]
                // Then: A_avg = (A1 + A2) / 2, delta_b = -(b2 - b1) / 2
                // displacement d = A_avg^{-1} * delta_b
                //
                // The next-frame expansion is sampled at the displacement already
                // estimated for the pixel being refined, so `d` is the residual
                // of the current field (Farnebäck / Bouguet). Sampling it at the
                // same pixel as the previous frame makes every iteration
                // recompute the full displacement, and adding that scales the
                // field by `iterations`.
                let mut a_sum: Matrix2<f64> = Matrix2::zeros();
                let mut db_sum: Vector2<f64> = Vector2::zeros();
                let mut w_sum: f64 = 0.0;

                for dy in -half_window..=half_window {
                    for dx in -half_window..=half_window {
                        let px = (x as i32 + dx) as u32;
                        let py = (y as i32 + dy) as u32;

                        if px >= width || py >= height {
                            continue;
                        }

                        let prev_c = prev_poly.get(px, py);
                        // One displacement for the window: the estimate at the
                        // pixel being refined. The next-frame expansion is read
                        // there, so the solve returns a residual.
                        let next_c = next_poly.sample(px as f32 + u_prev, py as f32 + v_prev);

                        let weight = win_weight[((dy + half_window) as usize) * win_side
                            + (dx + half_window) as usize];

                        // A from quadratic coefficients: [[2*A, B], [B, 2*C]]
                        let a1 = Matrix2::new(
                            2.0 * prev_c.0 as f64,
                            prev_c.1 as f64,
                            prev_c.1 as f64,
                            2.0 * prev_c.2 as f64,
                        );
                        let a2 = Matrix2::new(
                            2.0 * next_c.0 as f64,
                            next_c.1 as f64,
                            next_c.1 as f64,
                            2.0 * next_c.2 as f64,
                        );
                        let a_avg = (a1 + a2) * 0.5;

                        // b from linear terms.
                        // a = [[2A, B], [B, 2C]] = 2 A_paper, and the
                        // displacement identity is 2 A d = b1 - b2. The 1/2 on
                        // the mismatch is a damped step: the undamped step is
                        // unstable once the remaining motion exceeds the
                        // polynomial neighbourhood (~2 px).
                        let b1 = Vector2::new(prev_c.3 as f64, prev_c.4 as f64);
                        let b2 = Vector2::new(next_c.3 as f64, next_c.4 as f64);
                        let delta_b = -(b2 - b1) * 0.5;

                        // Accumulate weighted A^T * A and A^T * delta_b for least squares
                        let a_t = a_avg.transpose();
                        a_sum += a_t * a_avg * weight;
                        db_sum += a_t * delta_b * weight;
                        w_sum += weight;
                    }
                }

                if w_sum > 0.0 {
                    if let Some(a_inv) = a_sum.try_inverse() {
                        // `d` is only the residual against the displacement the
                        // next-frame polynomial was sampled at. Add it to the
                        // estimate this pixel already holds.
                        let d = a_inv * db_sum;
                        new_flow.set_motion(
                            x,
                            y,
                            (u_prev as f64 + d[0]) as f32,
                            (v_prev as f64 + d[1]) as f32,
                        );
                    } else {
                        new_flow.set_motion(x, y, u_prev, v_prev);
                    }
                } else {
                    new_flow.set_motion(x, y, u_prev, v_prev);
                }
            }
        }

        new_flow
    }
}

/// Polynomial expansion coefficients for f(x,y) ~ Ax^2 + Bxy + Cy^2 + Dx + Ey.
///
/// The constant term `F` of the full quadratic is produced by the 6x6 solve
/// but is never used by the flow (which only reads the gradients A..E that
/// feed the structure tensor), so it is dropped here rather than carried as a
/// dead field.
#[derive(Clone, Copy)]
struct PolyCoeffs(f32, f32, f32, f32, f32);

/// Polynomial expansion result
struct PolynomialExpansion {
    coeffs: Vec<PolyCoeffs>,
    width: u32,
    height: u32,
}

impl PolynomialExpansion {
    fn get(&self, x: u32, y: u32) -> PolyCoeffs {
        let idx = (y * self.width + x) as usize;
        self.coeffs
            .get(idx)
            .copied()
            .unwrap_or(PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0))
    }

    fn at(&self, x: i32, y: i32) -> PolyCoeffs {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            self.get(x as u32, y as u32)
        }
    }

    /// Expansion at a fractional pixel.
    ///
    /// Each neighbouring expansion is recentred onto `(x, y)` before the
    /// bilinear blend. Blending the raw coefficients mixes polynomials that
    /// are centred on different pixels, so a fractional displacement does not
    /// describe the image at that point. An integer coordinate inside the
    /// image is exactly [`PolynomialExpansion::get`].
    fn sample(&self, x: f32, y: f32) -> PolyCoeffs {
        if !x.is_finite() || !y.is_finite() {
            return PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0);
        }
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let c00 = recenter(self.at(x0, y0), fx, fy);
        let c10 = recenter(self.at(x0 + 1, y0), fx - 1.0, fy);
        let c01 = recenter(self.at(x0, y0 + 1), fx, fy - 1.0);
        let c11 = recenter(self.at(x0 + 1, y0 + 1), fx - 1.0, fy - 1.0);
        let top = lerp_poly(c00, c10, fx);
        let bot = lerp_poly(c01, c11, fx);
        lerp_poly(top, bot, fy)
    }
}

/// Re-express a quadratic centred at the origin as one centred at `(-dx, -dy)`,
/// i.e. shift the expansion point by `(dx, dy)`. Quadratic terms are unchanged;
/// linear terms pick up `2 A δ`.
fn recenter(c: PolyCoeffs, dx: f32, dy: f32) -> PolyCoeffs {
    PolyCoeffs(
        c.0,
        c.1,
        c.2,
        c.3 + 2.0 * c.0 * dx + c.1 * dy,
        c.4 + c.1 * dx + 2.0 * c.2 * dy,
    )
}

fn lerp_poly(a: PolyCoeffs, b: PolyCoeffs, t: f32) -> PolyCoeffs {
    PolyCoeffs(
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
        a.3 + (b.3 - a.3) * t,
        a.4 + (b.4 - a.4) * t,
    )
}

/// Solve 6x6 system A*x = b using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular (determinant ≈ 0).
///
/// This deliberately does *not* call `cv_math::linalg::solve_square`: that helper
/// routes through nalgebra's generic (const-generic) LU, which measured ~1.7x
/// slower than this hand-written loop for the 6x6 f32 case on the machine the
/// consolidation was done on (84-96 ns/solve here vs 140-165 ns/solve for
/// `nalgebra::linalg::LU` on the same 200k weighted normal-equations systems,
/// `--release`). This function runs once per pixel, so that is ~20 ms/frame at
/// VGA. The loop below is unchanged from the previous revision (same partial
/// pivoting + back substitution), so the coefficients are bit-identical.
fn solve_6x6_gauss(a: &mut [[f32; 6]; 6], b: &mut [f32; 6]) -> Option<[f32; 6]> {
    let n = 6;

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None; // Singular matrix
        }

        // Swap rows if needed
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = [0.0f32; 6];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    Some(x)
}

/// Compute polynomial expansion of image
fn polynomial_expansion(img: &GrayImage, poly_n: usize, sigma: f32) -> PolynomialExpansion {
    let width = img.width();
    let height = img.height();
    let mut coeffs = Vec::with_capacity((width * height) as usize);

    let half_n = (poly_n / 2) as i32;
    let side = (2 * half_n + 1) as usize;
    // The Gaussian depends only on the offset, so each tap's weight is one
    // number for the whole image. Interior pixels also share the Gram matrix:
    // every offset is inside the image, and the basis does not depend on the
    // pixel. The right-hand side still does.
    let mut weights = vec![0.0f32; side * side];
    for dy in -half_n..=half_n {
        for dx in -half_n..=half_n {
            let i = ((dy + half_n) as usize) * side + (dx + half_n) as usize;
            weights[i] = gaussian_weight(dx as f32, dy as f32, sigma);
        }
    }
    let mut a_full = [[0.0f32; 6]; 6];
    for dy in -half_n..=half_n {
        for dx in -half_n..=half_n {
            let xf = dx as f32;
            let yf = dy as f32;
            let weight = weights[((dy + half_n) as usize) * side + (dx + half_n) as usize];
            let basis = [xf * xf, xf * yf, yf * yf, xf, yf, 1.0];
            for i in 0..6 {
                for j in 0..6 {
                    a_full[i][j] += basis[i] * basis[j] * weight;
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            // Fit quadratic polynomial in neighborhood using weighted least squares
            let interior = x as i32 >= half_n
                && y as i32 >= half_n
                && (x as i32) + half_n < width as i32
                && (y as i32) + half_n < height as i32;
            let mut a = if interior { a_full } else { [[0.0f32; 6]; 6] };
            let mut b = [0.0f32; 6];

            for dy in -half_n..=half_n {
                for dx in -half_n..=half_n {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;

                    if px >= width || py >= height {
                        continue;
                    }

                    let val = img.get_pixel(px, py)[0] as f32;
                    let xf = dx as f32;
                    let yf = dy as f32;
                    let weight = weights[((dy + half_n) as usize) * side + (dx + half_n) as usize];

                    // Basis functions
                    let basis = [
                        xf * xf, // x^2
                        xf * yf, // xy
                        yf * yf, // y^2
                        xf,      // x
                        yf,      // y
                        1.0,     // 1
                    ];

                    if !interior {
                        for i in 0..6 {
                            for j in 0..6 {
                                a[i][j] += basis[i] * basis[j] * weight;
                            }
                        }
                    }
                    for i in 0..6 {
                        b[i] += basis[i] * val * weight;
                    }
                }
            }

            // Solve the full 6x6 system A*c = b using Gaussian elimination with partial pivoting
            // (avoids 614,400 heap allocations of DMatrix/DVector per VGA frame)
            let c = solve_6x6_gauss(&mut a, &mut b)
                .map(|sol| {
                    PolyCoeffs(
                        sol[0], // A: x^2
                        sol[1], // B: xy
                        sol[2], // C: y^2
                        sol[3], // D: x
                        sol[4], // E: y
                                // sol[5] is the constant term F; see PolyCoeffs.
                    )
                })
                .unwrap_or_else(|| PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0));

            coeffs.push(c);
        }
    }

    PolynomialExpansion {
        coeffs,
        width,
        height,
    }
}

/// Gaussian weight function.
///
/// `sigma` is validated by [`Farneback::validate`] before this is reached; the
/// guard below is a defence in depth so a `sigma == 0` can never evaluate
/// `0 / 0` and poison a whole frame with `NaN`.
fn gaussian_weight(x: f32, y: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        // Degenerate sigma: fall back to a uniform (box) weighting instead of
        // returning NaN.
        return 1.0;
    }
    let w = (-(x * x + y * y) / (2.0 * sigma_sq)).exp();
    if w.is_finite() { w } else { 1.0 }
}

/// Scale image down
fn scale_image(img: &GrayImage, scale: f32) -> GrayImage {
    let new_width = (img.width() as f32 * scale) as u32;
    let new_height = (img.height() as f32 * scale) as u32;

    image::imageops::resize(
        img,
        new_width,
        new_height,
        image::imageops::FilterType::Gaussian,
    )
}

/// Scale motion field
fn scale_motion_field(flow: &MotionField, new_width: u32, new_height: u32) -> MotionField {
    let scale_x = new_width as f32 / flow.width as f32;
    let scale_y = new_height as f32 / flow.height as f32;

    let mut new_flow = MotionField::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f32 / scale_x) as u32;
            let src_y = (y as f32 / scale_y) as u32;

            if src_x < flow.width && src_y < flow.height {
                let (u, v) = flow.get_motion(src_x, src_y);
                new_flow.set_motion(x, y, u * scale_x, v * scale_y);
            }
        }
    }

    new_flow
}

/// Convenience function for sparse optical flow
pub fn calc_optical_flow_lk(
    prev_frame: &GrayImage,
    next_frame: &GrayImage,
    points: &[(f32, f32)],
) -> Vec<Option<(f32, f32)>> {
    let lk = LucasKanade::new();
    lk.track_points(prev_frame, next_frame, points)
}

/// Convenience function for dense optical flow
pub fn calc_optical_flow_farneback(
    prev_frame: &GrayImage,
    next_frame: &GrayImage,
) -> Result<MotionField> {
    let fb = Farneback::new();
    fb.compute(prev_frame, next_frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_moving_square(prev_pos: (u32, u32), next_pos: (u32, u32)) -> (GrayImage, GrayImage) {
        let width = 100u32;
        let height = 100u32;
        let square_size = 20u32;

        let mut prev = GrayImage::new(width, height);
        let mut next = GrayImage::new(width, height);

        // Fill with black
        for y in 0..height {
            for x in 0..width {
                prev.put_pixel(x, y, Luma([0]));
                next.put_pixel(x, y, Luma([0]));
            }
        }

        // Draw square
        for y in 0..square_size {
            for x in 0..square_size {
                let px = (prev_pos.0 + x).min(width - 1);
                let py = (prev_pos.1 + y).min(height - 1);
                prev.put_pixel(px, py, Luma([255]));

                let nx = (next_pos.0 + x).min(width - 1);
                let ny = (next_pos.1 + y).min(height - 1);
                next.put_pixel(nx, ny, Luma([255]));
            }
        }

        (prev, next)
    }

    #[test]
    fn test_lucas_kanade() {
        let (prev, next) = create_moving_square((40, 40), (50, 40));

        // Track a point on the square
        let points = vec![(50.0, 50.0)];
        let lk = LucasKanade::new();

        let tracked = lk.track_points(&prev, &next, &points);

        assert!(tracked[0].is_some(), "Should track the point successfully");
        let (x, y) = tracked[0].unwrap();

        // Just verify tracking works and produces a result
        // Exact motion estimation depends on many factors
        println!("Tracked point from (50, 50) to ({:.1}, {:.1})", x, y);
    }

    #[test]
    fn test_farneback() {
        let (prev, next) = create_moving_square((40, 40), (50, 40));

        let fb = Farneback::new().with_pyramid_levels(2).with_window_size(11);

        let flow = fb.compute(&prev, &next).unwrap();

        // Check flow in the square region
        let (u, v) = flow.get_motion(50, 50);
        println!("Flow at center: u={}, v={}", u, v);

        // Should detect rightward motion
        assert!(u > 0.0, "Expected positive horizontal flow");
    }

    /// Farneback must estimate the *magnitude* of the motion, not merely its
    /// sign.
    ///
    /// The existing `test_farneback` only asserts `u > 0.0`, which passes for a
    /// field of almost any size in the right direction. This one shifts a textured
    /// image by a known amount and checks the estimate against it.
    #[test]
    fn probe_farneback_magnitude() {
        let (pw, ph) = (120usize, 100usize);
        let shift = 10i32;
        let mut a = vec![0f32; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                a[y * pw + x] = (((x * 37 + y * 91) % 255) as f32) / 255.0;
            }
        }
        let mut b = vec![0f32; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                let sx = x as i32 - shift;
                b[y * pw + x] = if sx >= 0 {
                    a[y * pw + sx as usize]
                } else {
                    0.0
                };
            }
        }
        let to_img = |v: &[f32]| {
            image::GrayImage::from_raw(
                pw as u32,
                ph as u32,
                v.iter().map(|&x| (x * 255.0) as u8).collect(),
            )
            .unwrap()
        };
        let fb = Farneback::new().with_pyramid_levels(3).with_window_size(11);
        let flow = fb.compute(&to_img(&a), &to_img(&b)).unwrap();

        let mut sum = 0f64;
        let mut n = 0u64;
        for y in 30..70u32 {
            for x in 30..80u32 {
                sum += flow.get_motion(x, y).0 as f64;
                n += 1;
            }
        }
        let mean_u = sum / n as f64;
        // Assert the magnitude, not just the sign. `test_farneback` asserts
        // `u > 0.0`, which passes on 0.054 - 0.5% of the true motion.
        assert!(
            (mean_u - shift as f64).abs() < 2.0,
            "Farneback estimated {mean_u:.3} px for a known {shift} px shift; \
             the field is well-formed but does not estimate translation"
        );
    }

    /// Textured frame pair where `next(x, y) = prev(x - shift, y)`.
    fn translated_texture(w: u32, h: u32, shift: i32) -> (GrayImage, GrayImage) {
        let mut prev = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (x.wrapping_mul(37).wrapping_add(y.wrapping_mul(91)) % 255) as u8;
                prev.put_pixel(x, y, Luma([v]));
            }
        }
        let mut next = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = x as i32 - shift;
                let v = if sx >= 0 && (sx as u32) < w {
                    prev.get_pixel(sx as u32, y)[0]
                } else {
                    0
                };
                next.put_pixel(x, y, Luma([v]));
            }
        }
        (prev, next)
    }

    fn mean_horizontal(flow: &crate::MotionField) -> f64 {
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for y in 30..70u32 {
            for x in 40..90u32 {
                sum += flow.get_motion(x, y).0 as f64;
                n += 1;
            }
        }
        sum / n as f64
    }

    fn farneback_mean_u(
        prev: &GrayImage,
        next: &GrayImage,
        iterations: usize,
        pyramid_levels: usize,
    ) -> f64 {
        let mut fb = Farneback::new();
        fb.iterations = iterations;
        fb.pyramid_levels = pyramid_levels;
        mean_horizontal(&fb.compute(prev, next).unwrap())
    }

    /// Later iterations must see a residual. Sampling both polynomial expansions
    /// at the same pixel makes every iteration recompute the full displacement
    /// and `flow += d` scales it: 3 iterations is then exactly 3× one iteration,
    /// and a 1 px shift comes back near 3 px.
    #[test]
    fn farneback_iterations_refine_residual_not_scale_flow() {
        let (prev, next) = translated_texture(120, 100, 10);
        let u1 = farneback_mean_u(&prev, &next, 1, 3);
        let u3 = farneback_mean_u(&prev, &next, 3, 3);
        assert!(
            (u3 - 3.0 * u1).abs() > 0.5,
            "3 iterations ({u3:.3}) is an exact multiple of 1 iteration ({u1:.3}); \
             iterations scaled the displacement instead of refining a residual"
        );

        let (prev, next) = translated_texture(120, 100, 1);
        let u = farneback_mean_u(&prev, &next, 3, 1);
        assert!(
            (u - 1.0).abs() < 0.75,
            "1 px shift reported as {u:.3}, not near 1 px"
        );
        assert!(
            (u - 3.0).abs() > 1.0,
            "1 px shift reported as {u:.3}, which is the 3-iteration scaling"
        );
    }

    #[test]
    fn test_optical_flow_lk_accuracy() {
        // Create a textured patch and shift it by (5, 0) pixels.
        // Track a point in the center and verify displacement is ~5px.
        let width = 120u32;
        let height = 120u32;
        let shift = 5i32;

        // Build a checkerboard-like texture so gradients are well-defined.
        let mut prev = GrayImage::new(width, height);
        let mut next = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                // 8x8 checkerboard pattern
                let cx = (x / 8) % 2;
                let cy = (y / 8) % 2;
                let val = if cx ^ cy == 1 { 200u8 } else { 55u8 };
                prev.put_pixel(x, y, Luma([val]));
            }
        }

        // Shift entire texture rightward by `shift` pixels
        for y in 0..height {
            for x in 0..width {
                let sx = (x as i32 - shift).clamp(0, width as i32 - 1) as u32;
                let val = prev.get_pixel(sx, y)[0];
                next.put_pixel(x, y, Luma([val]));
            }
        }

        let lk = LucasKanade::new()
            .with_window_size(21)
            .with_pyramid_levels(3);
        let point = (60.0f32, 60.0f32);
        let tracked = lk.track_point(&prev, &next, point);

        assert!(tracked.is_some(), "LK should successfully track the point");
        let (nx, _ny) = tracked.unwrap();
        let dx = nx - point.0;
        println!(
            "LK tracked ({:.1},{:.1}) -> ({:.1},{:.1}), dx={:.2}",
            point.0, point.1, nx, _ny, dx
        );
        assert!(
            (dx - shift as f32).abs() < 2.0,
            "Tracked horizontal displacement {:.2} should be within 2px of {}",
            dx,
            shift
        );
    }

    #[test]
    fn test_farneback_flow_magnitude() {
        // Shift a square rightward by 10px and verify the Farneback dense
        // flow field detects positive horizontal motion. We sample the
        // overlapping interior of the square in both frames and check that
        // mean flow is positive and that the max flow magnitude exceeds 1px.
        let shift = 10u32;
        let (prev, next) = create_moving_square((30, 40), (30 + shift, 40));

        let fb = Farneback::new().with_pyramid_levels(2).with_window_size(11);
        let flow = fb.compute(&prev, &next).unwrap();

        // Sample a strip in the overlap region of the two square positions
        let mut sum_u = 0.0f64;
        let mut count = 0u32;
        for y in 45..55 {
            for x in (30 + shift)..50 {
                let (u, _v) = flow.get_motion(x, y);
                sum_u += u as f64;
                count += 1;
            }
        }
        let mean_u = sum_u / count as f64;

        // Collect magnitudes only for pixels that have finite, moderate flow
        let mag = flow.magnitude();
        let valid_mags: Vec<f32> = mag
            .iter()
            .copied()
            .filter(|m| m.is_finite() && *m < 1e6)
            .collect();
        let max_valid_mag = valid_mags.iter().cloned().fold(0.0f32, f32::max);

        println!(
            "Farneback mean horizontal flow in shifted region: {:.2}, max valid magnitude: {:.2}",
            mean_u, max_valid_mag
        );
        // The dense flow should detect positive horizontal motion in the shifted region
        assert!(
            mean_u > 0.0,
            "Mean horizontal flow {:.2} in shifted region should be positive",
            mean_u,
        );
        // The maximum valid flow magnitude should be substantial (> 1px)
        assert!(
            max_valid_mag > 1.0,
            "Max valid flow magnitude {:.2} should be > 1.0 for a {}-px shift",
            max_valid_mag,
            shift
        );
    }

    // -----------------------------------------------------------------------
    // GAPS g-i2: sub-pixel accuracy of pyramidal LK (Bouguet / OpenCV
    // calcOpticalFlowPyrLK). The template must be sampled at the feature's
    // fractional position, not at floor(x).
    // -----------------------------------------------------------------------

    /// Smooth band-limited texture (wavelengths 29-53 px), evaluated
    /// analytically so a sub-pixel shift is exact up to u8 quantisation.
    fn smooth_texture(x: f64, y: f64) -> f64 {
        use std::f64::consts::TAU;
        128.0
            + 45.0 * (TAU * (x / 37.0 + y / 53.0)).sin()
            + 40.0 * (TAU * (x / 43.0 - y / 29.0)).cos()
            + 30.0 * (TAU * (x / 31.0 + y / 47.0) + 1.0).sin()
    }

    /// `prev(x, y) = f(x, y)`, `next(x, y) = f(x - dx, y - dy)`.
    fn shifted_pair(w: u32, h: u32, dx: f64, dy: f64) -> (GrayImage, GrayImage) {
        let prev = GrayImage::from_fn(w, h, |x, y| {
            Luma([smooth_texture(x as f64, y as f64).round() as u8])
        });
        let next = GrayImage::from_fn(w, h, |x, y| {
            Luma([smooth_texture(x as f64 - dx, y as f64 - dy).round() as u8])
        });
        (prev, next)
    }

    #[test]
    fn lk_identical_frames_keep_subpixel_points() {
        let (img, _) = shifted_pair(128, 128, 0.0, 0.0);
        for levels in [1usize, 3] {
            let lk = LucasKanade::new().with_pyramid_levels(levels);
            for pt in [(60.5f32, 60.5f32), (60.75, 60.25), (63.3, 58.9)] {
                let (x, y) = lk.track_point(&img, &img, pt).expect("trackable");
                assert!(
                    (x - pt.0).abs() < 1e-3 && (y - pt.1).abs() < 1e-3,
                    "{levels} level(s): identical frames moved {pt:?} to ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn lk_subpixel_shift_converges_monotonically() {
        let truth = (2.3f64, -1.7f64);
        let (prev, next) = shifted_pair(128, 128, truth.0, truth.1);
        let pt = (64.0f32, 64.0f32);
        let mut errs = Vec::new();
        for iters in [1usize, 2, 3, 5, 10, 20] {
            let mut lk = LucasKanade::new().with_pyramid_levels(1);
            lk.max_iterations = iters;
            let (x, y) = lk.track_point(&prev, &next, pt).expect("trackable");
            let f = (x as f64 - pt.0 as f64, y as f64 - pt.1 as f64);
            let e = ((f.0 - truth.0).powi(2) + (f.1 - truth.1).powi(2)).sqrt();
            println!("iters {iters:>2}: flow ({:.4}, {:.4}) err {e:.4}", f.0, f.1);
            errs.push((iters, f, e));
        }
        for w in errs.windows(2) {
            assert!(
                w[1].2 <= w[0].2 + 1e-3,
                "error grew with budget: {} iters {:.4} -> {} iters {:.4}",
                w[0].0,
                w[0].2,
                w[1].0,
                w[1].2
            );
        }
        for (iters, f, e) in &errs {
            if *iters >= 5 {
                assert!(
                    *e < 0.02,
                    "{iters} iters: flow {f:?} is {e:.4} px from {truth:?}"
                );
            }
        }
        let (f10, f20) = (errs[4].1, errs[5].1);
        assert!((f10.0 - f20.0).abs() < 1e-3 && (f10.1 - f20.1).abs() < 1e-3);
    }

    #[test]
    fn lk_pyramid_recovers_large_subpixel_shift() {
        let truth = (9.6f64, -4.3f64);
        let (prev, next) = shifted_pair(192, 160, truth.0, truth.1);
        let lk = LucasKanade::new().with_pyramid_levels(3);
        let pts = [(96.0f32, 80.0f32), (90.4, 77.7)];
        let tracked = lk.track_points(&prev, &next, &pts);
        for (pt, t) in pts.iter().zip(&tracked) {
            let (x, y) = t.expect("trackable");
            let f = (x as f64 - pt.0 as f64, y as f64 - pt.1 as f64);
            let e = ((f.0 - truth.0).powi(2) + (f.1 - truth.1).powi(2)).sqrt();
            println!("pyramid {pt:?}: flow ({:.4}, {:.4}) err {e:.4}", f.0, f.1);
            assert!(
                e < 0.02,
                "pyramid track of {pt:?}: flow {f:?}, truth {truth:?}"
            );
            // Batch and single-point paths must agree exactly.
            assert_eq!(Some((x, y)), lk.track_point(&prev, &next, *pt));
        }
    }
}
