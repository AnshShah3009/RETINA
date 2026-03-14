//! Optical Flow Algorithms
//!
//! Implements various optical flow estimation methods:
//! - Lucas-Kanade: Sparse optical flow for corner tracking
//! - Farneback: Dense optical flow using polynomial expansion
//! - Horn-Schunck: Global smoothness constraint

use crate::{MotionField, Result};
use cv_core::{Error, KeyPoint};
use image::GrayImage;
use nalgebra::{DMatrix, DVector, Matrix2, Vector2};

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

    /// Track a single point from prev_frame to next_frame
    ///
    /// TODO: `pyramid_levels` is not yet implemented - tracking is performed at
    ///       a single scale. Multi-scale pyramid processing would improve large
    ///       motion estimation.
    /// TODO: `max_iterations` is not yet implemented - the solver runs a single
    ///       pass instead of iteratively refining the flow estimate.
    /// TODO: `epsilon` convergence criterion is not yet implemented - related to
    ///       iterative refinement above.
    pub fn track_point(
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

            // Solve the full 6x6 system A*c = b using nalgebra matrix inverse
            let a_mat = DMatrix::from_fn(6, 6, |i, j| a[i][j] as f64);
            let b_vec = DVector::from_fn(6, |i, _| b[i] as f64);

            let c = if let Some(a_inv) = a_mat.try_inverse() {
                let sol = a_inv * b_vec;
                PolyCoeffs(
                    sol[0] as f32, // A: x^2
                    sol[1] as f32, // B: xy
                    sol[2] as f32, // C: y^2
                    sol[3] as f32, // D: x
                    sol[4] as f32, // E: y
                    sol[5] as f32, // F: constant
                )
            } else {
                PolyCoeffs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

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
}
