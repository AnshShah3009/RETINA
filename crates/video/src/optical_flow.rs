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

    /// Track a single point using pyramidal iterative Lucas-Kanade (Bouguet's algorithm).
    ///
    /// Builds Gaussian image pyramids for both frames and performs coarse-to-fine
    /// tracking with iterative refinement at each pyramid level.
    pub fn track_point(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        point: (f32, f32),
    ) -> Option<(f32, f32)> {
        let (px, py) = point;
        let half_window = (self.window_size / 2) as i32;

        // Basic bounds check at the finest level
        if px < half_window as f32
            || px >= (prev_frame.width() as i32 - half_window) as f32
            || py < half_window as f32
            || py >= (prev_frame.height() as i32 - half_window) as f32
        {
            return None;
        }

        // Build image pyramids (level 0 = finest)
        let prev_pyramid = build_pyramid(prev_frame, self.pyramid_levels);
        let next_pyramid = build_pyramid(next_frame, self.pyramid_levels);

        // Initial guess at the coarsest level
        let mut flow = Vector2::<f64>::zeros();

        // Coarse-to-fine: iterate from coarsest (last) to finest (0)
        for level in (0..self.pyramid_levels).rev() {
            let scale = 1.0 / (1u32 << level) as f64;
            let prev_img = &prev_pyramid[level];
            let next_img = &next_pyramid[level];

            // Point location at this pyramid level
            let lx = px as f64 * scale;
            let ly = py as f64 * scale;

            let hw = half_window;
            let w = prev_img.width() as i32;
            let h = prev_img.height() as i32;

            // Check bounds at this level
            if (lx as i32) < hw
                || (lx as i32) >= w - hw
                || (ly as i32) < hw
                || (ly as i32) >= h - hw
            {
                // Cannot track at this level; propagate current guess down
                if level > 0 {
                    flow *= 2.0;
                }
                continue;
            }

            // Compute spatial gradients and structure tensor G once per level
            // G = sum over window of [Ix*Ix  Ix*Iy; Ix*Iy  Iy*Iy]
            let mut g: Matrix2<f64> = Matrix2::zeros();
            let mut grad_patch: Vec<(f64, f64)> =
                Vec::with_capacity(self.window_size * self.window_size);

            for dy in -hw..=hw {
                for dx in -hw..=hw {
                    let sx = lx as i32 + dx;
                    let sy = ly as i32 + dy;

                    let ix = compute_gradient_x(prev_img, sx as u32, sy as u32) as f64;
                    let iy = compute_gradient_y(prev_img, sx as u32, sy as u32) as f64;

                    g[(0, 0)] += ix * ix;
                    g[(0, 1)] += ix * iy;
                    g[(1, 0)] += ix * iy;
                    g[(1, 1)] += iy * iy;

                    grad_patch.push((ix, iy));
                }
            }

            // Singularity check via eigenvalues of G
            // For a 2x2 symmetric matrix: eigenvalues = (trace +/- sqrt(trace^2 - 4*det)) / 2
            let trace = g[(0, 0)] + g[(1, 1)];
            let det = g[(0, 0)] * g[(1, 1)] - g[(0, 1)] * g[(1, 0)];
            let discriminant = trace * trace - 4.0 * det;
            if discriminant < 0.0 || det < 1e-7 {
                // Structure tensor is degenerate; point not trackable
                return None;
            }
            let sqrt_disc = discriminant.sqrt();
            let min_eigenvalue = (trace - sqrt_disc) / 2.0;
            if min_eigenvalue < 1e-4 {
                return None;
            }

            let g_inv = g.try_inverse()?;

            // Iterative refinement at this level
            // flow already contains the guess propagated from the coarser level
            for _iter in 0..self.max_iterations {
                // Compute temporal gradient using warped second image
                let mut b_vec: Vector2<f64> = Vector2::zeros();
                let mut idx = 0;

                for dy in -hw..=hw {
                    for dx in -hw..=hw {
                        let sx = lx as i32 + dx;
                        let sy = ly as i32 + dy;

                        // Warped position in next image
                        let wx = lx + flow[0] + dx as f64;
                        let wy = ly + flow[1] + dy as f64;

                        let i1 = prev_img.get_pixel(sx as u32, sy as u32)[0] as f64;
                        let i2 = bilinear_sample(next_img, wx, wy);

                        let it = i2 - i1;

                        let (ix, iy) = grad_patch[idx];
                        b_vec[0] += -ix * it;
                        b_vec[1] += -iy * it;

                        idx += 1;
                    }
                }

                let delta = g_inv * b_vec;
                flow += delta;

                // Convergence check
                if delta[0] * delta[0] + delta[1] * delta[1]
                    < (self.epsilon as f64) * (self.epsilon as f64)
                {
                    break;
                }
            }

            // Scale flow up for the next finer level (unless we are already at level 0)
            if level > 0 {
                flow *= 2.0;
            }
        }

        let new_x = px as f64 + flow[0];
        let new_y = py as f64 + flow[1];

        // Reject if the tracked point falls outside the image
        if new_x < 0.0
            || new_x >= next_frame.width() as f64
            || new_y < 0.0
            || new_y >= next_frame.height() as f64
        {
            return None;
        }

        Some((new_x as f32, new_y as f32))
    }

    /// Single-pass (non-iterative, non-pyramidal) Lucas-Kanade tracking.
    ///
    /// This is the original simple implementation kept as a fallback.
    /// It computes a single gradient solve with no pyramid and no iteration.
    pub fn track_point_single_pass(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        point: (f32, f32),
    ) -> Option<(f32, f32)> {
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

    /// Track multiple points
    pub fn track_points(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        points: &[(f32, f32)],
    ) -> Vec<Option<(f32, f32)>> {
        points
            .iter()
            .map(|&p| self.track_point(prev_frame, next_frame, p))
            .collect()
    }

    /// Track keypoints
    pub fn track_keypoints(
        &self,
        prev_frame: &GrayImage,
        next_frame: &GrayImage,
        keypoints: &[KeyPoint],
    ) -> Vec<Option<KeyPoint>> {
        keypoints
            .iter()
            .map(|kp| {
                self.track_point(prev_frame, next_frame, (kp.x as f32, kp.y as f32))
                    .map(|(x, y)| {
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

/// Build a Gaussian image pyramid.
///
/// Returns a vector of images from finest (level 0) to coarsest.
/// Each subsequent level is Gaussian-blurred and downsampled by 2x.
fn build_pyramid(img: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(img.clone());

    for _ in 1..levels {
        let prev = pyramid.last().unwrap();
        // Apply Gaussian blur before downsampling to avoid aliasing
        let blurred = image::imageops::blur(prev, 1.0);
        let new_w = (blurred.width() / 2).max(1);
        let new_h = (blurred.height() / 2).max(1);
        let downsampled =
            image::imageops::resize(&blurred, new_w, new_h, image::imageops::FilterType::Nearest);
        pyramid.push(downsampled);
    }

    pyramid
}

/// Bilinear interpolation sampling of a grayscale image.
///
/// Returns the interpolated pixel intensity at sub-pixel coordinates (x, y).
/// Out-of-bounds coordinates are clamped to the nearest valid pixel.
fn bilinear_sample(img: &GrayImage, x: f64, y: f64) -> f64 {
    let w = img.width() as f64;
    let h = img.height() as f64;

    // Clamp to valid range
    let x = x.clamp(0.0, w - 1.001);
    let y = y.clamp(0.0, h - 1.001);

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(img.width() - 1);
    let y1 = (y0 + 1).min(img.height() - 1);

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = img.get_pixel(x0, y0)[0] as f64;
    let p10 = img.get_pixel(x1, y0)[0] as f64;
    let p01 = img.get_pixel(x0, y1)[0] as f64;
    let p11 = img.get_pixel(x1, y1)[0] as f64;

    p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy) + p01 * (1.0 - fx) * fy + p11 * fx * fy
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

    pub fn with_pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Compute dense optical flow
    pub fn compute(&self, prev_frame: &GrayImage, next_frame: &GrayImage) -> Result<MotionField> {
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

            // Refine flow using iterative method
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

        for y in 0..height {
            for x in 0..width {
                let (_u, _v) = flow.get_motion(x, y);

                // Correct Farneback formulation:
                // For each pixel, build A from quadratic coefficients and b from linear terms.
                // A_k = [[2*coeff.0, coeff.1], [coeff.1, 2*coeff.2]]
                // b_k = [coeff.3, coeff.4]
                // Then: A_avg = (A1 + A2) / 2, delta_b = -(b2 - b1) / 2
                // displacement d = A_avg^{-1} * delta_b
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
                        let next_c = next_poly.get(px, py);

                        let weight = gaussian_weight(dx as f32, dy as f32, self.poly_sigma) as f64;

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

                        // b from linear terms
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
                        let d = a_inv * db_sum;
                        new_flow.set_motion(x, y, d[0] as f32, d[1] as f32);
                    } else {
                        new_flow.set_motion(x, y, _u, _v);
                    }
                } else {
                    new_flow.set_motion(x, y, _u, _v);
                }
            }
        }

        new_flow
    }
}

/// Polynomial expansion coefficients for f(x,y) ~ Ax^2 + Bxy + Cy^2 + Dx + Ey + F.
/// Fields: (A: x^2, B: xy, C: y^2, D: x, E: y, F: constant)
#[derive(Clone, Copy)]
#[allow(dead_code)] // Field .5 (F constant) computed by full 6x6 solve, reserved for future use
struct PolyCoeffs(f32, f32, f32, f32, f32, f32);

/// Polynomial expansion result
struct PolynomialExpansion {
    coeffs: Vec<PolyCoeffs>,
    width: u32,
    _height: u32,
}

impl PolynomialExpansion {
    fn get(&self, x: u32, y: u32) -> PolyCoeffs {
        let idx = (y * self.width + x) as usize;
        self.coeffs
            .get(idx)
            .copied()
            .unwrap_or(PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    }
}

/// Solve 6x6 system A*x = b using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular (determinant ≈ 0).
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

    for y in 0..height {
        for x in 0..width {
            // Fit quadratic polynomial in neighborhood using weighted least squares
            let mut a = [[0.0f32; 6]; 6];
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
                    let weight = gaussian_weight(xf, yf, sigma);

                    // Basis functions
                    let basis = [
                        xf * xf, // x^2
                        xf * yf, // xy
                        yf * yf, // y^2
                        xf,      // x
                        yf,      // y
                        1.0,     // 1
                    ];

                    for i in 0..6 {
                        for j in 0..6 {
                            a[i][j] += basis[i] * basis[j] * weight;
                        }
                        b[i] += basis[i] * val * weight;
                    }
                }
            }

            // Solve the full 6x6 system A*c = b using Gaussian elimination with partial pivoting
            // (avoids 614,400 heap allocations of DMatrix/DVector per VGA frame)
            let c = solve_6x6_gauss(&mut a, &mut b).map(|sol| {
                PolyCoeffs(
                    sol[0], // A: x^2
                    sol[1], // B: xy
                    sol[2], // C: y^2
                    sol[3], // D: x
                    sol[4], // E: y
                    sol[5], // F: constant
                )
            }).unwrap_or_else(|| PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

            coeffs.push(c);
        }
    }

    PolynomialExpansion {
        coeffs,
        width,
        _height: height,
    }
}

/// Gaussian weight function
fn gaussian_weight(x: f32, y: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    (-(x * x + y * y) / (2.0 * sigma_sq)).exp()
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
}
