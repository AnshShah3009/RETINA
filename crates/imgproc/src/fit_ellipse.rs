//! Ellipse fitting algorithms from OpenCV 5.x
//!
//! - `fit_ellipse_ams`: Approximate Mean Square (Taubin 1991)
//! - `fit_ellipse_direct`: Direct Least Squares (Fitzgibbon 1999)
//! - `fit_ellipse`: Standard ellipse fitting

use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, Point2, Point3, Vector3};
use cv_core::geometry::RotatedRect;

/// Result of ellipse fitting — returns RotatedRect bounding the ellipse
#[derive(Debug, Clone, Copy)]
pub struct EllipseResult {
    pub center: Point2<f64>,
    pub size: (f64, f64),     // (width, height) = (major_axis*2, minor_axis*2)
    pub angle: f64,            // rotation in degrees
}

impl EllipseResult {
    pub fn to_rotated_rect(&self) -> RotatedRect {
        RotatedRect::new(
            self.center.x as f32,
            self.center.y as f32,
            self.size.0 as f32,
            self.size.1 as f32,
            self.angle as f32,
        )
    }
}

/// Standard ellipse fitting — tries Direct first, falls back to AMS
pub fn fit_ellipse(points: &[Point2<f64>]) -> Option<EllipseResult> {
    fit_ellipse_direct(points).or_else(|| fit_ellipse_ams(points))
}

/// AMS (Approximate Mean Square) ellipse fitting — Taubin 1991
///
/// Uses a generalized eigenvalue problem to find the best-fitting ellipse.
/// Rejects parabolic/hyperbolic fits by checking ellipse condition (4ac - b² > 0).
pub fn fit_ellipse_ams(points: &[Point2<f64>]) -> Option<EllipseResult> {
    let n = points.len();
    if n < 6 {
        return fit_least_squares_circle(points);
    }

    let mut dtd = DMatrix::zeros(6, 6);
    let mut dxtdx_plus_dytdy = DMatrix::zeros(6, 6);

    for p in points {
        let x = p.x;
        let y = p.y;
        let d = DVector::from_vec(vec![x * x, x * y, y * y, x, y, 1.0]);
        let dx = DVector::from_vec(vec![2.0 * x, y, 0.0, 1.0, 0.0, 0.0]);
        let dy = DVector::from_vec(vec![0.0, x, 2.0 * y, 0.0, 1.0, 0.0]);

        dtd += &d * d.transpose();
        dxtdx_plus_dytdy += &dx * dx.transpose() + &dy * dy.transpose();
    }

    let conic = solve_ams_eigen(&dtd, &dxtdx_plus_dytdy)?;
    conic_to_ellipse_result(&conic)
}

/// Direct Least Squares ellipse fitting — Fitzgibbon 1999
///
/// Enforces 4*Axx*Ayy - Axy² = 1 constraint via generalized eigenvalue problem.
pub fn fit_ellipse_direct(points: &[Point2<f64>]) -> Option<EllipseResult> {
    let n = points.len();
    if n < 6 {
        return fit_least_squares_circle(points);
    }

    // Build design matrix D (n x 6)
    let mut d = DMatrix::zeros(n, 6);
    for (i, p) in points.iter().enumerate() {
        let x = p.x;
        let y = p.y;
        d[(i, 0)] = x * x;
        d[(i, 1)] = x * y;
        d[(i, 2)] = y * y;
        d[(i, 3)] = x;
        d[(i, 4)] = y;
        d[(i, 5)] = 1.0;
    }

    // Scatter matrix S = D^T * D (6x6)
    let s_dyn = d.transpose() * &d;
    let s = Matrix6::from_row_slice(s_dyn.as_slice());

    // Constraint matrix: 4*Axx*Ayy - Axy^2 = 1
    let mut c = Matrix6::zeros();
    c[(0, 2)] = 2.0;
    c[(2, 0)] = 2.0;
    c[(1, 1)] = -1.0;

    // Solve for conic via eigenvalue decomposition
    let conic = solve_direct_eigen(&s, &c)?;
    conic_to_ellipse_result(&conic)
}

/// Solve the AMS generalized eigenvalue problem
fn solve_ams_eigen(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Option<[f64; 6]> {
    // Solve the generalized eigenvalue problem A*u = λ*B*u
    // Find the smallest positive eigenvalue
    let inv_b = b.clone().try_inverse()?;
    let m = inv_b * a;
    let eig = m.symmetric_eigen();
    let mut eigenvalues: Vec<(f64, usize)> = eig.eigenvalues.iter().enumerate()
        .filter(|(_, &e)| e > 1e-10)
        .map(|(i, &e)| (e, i))
        .collect();
    eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(_, best_idx) in &eigenvalues {
        let v = eig.eigenvectors.column(best_idx);
        let conic: [f64; 6] = [v[0], v[1], v[2], v[3], v[4], v[5]];
        if 4.0 * conic[0] * conic[2] - conic[1] * conic[1] > 0.0 {
            return Some(conic);
        }
    }
    None
}

/// Solve the Direct least squares constrained eigenvalue problem
fn solve_direct_eigen(s: &Matrix6<f64>, c: &Matrix6<f64>) -> Option<[f64; 6]> {
    // Decompose S = [S1 S2; S2^T S3] and C = [C1 0; 0 0]
    // Solve reduced 3x3 eigenvalue problem
    let s11 = s.fixed_view::<3, 3>(0, 0).into_owned();
    let s12 = s.fixed_view::<3, 3>(0, 3).into_owned();
    let s22 = s.fixed_view::<3, 3>(3, 3).into_owned();
    let c1 = c.fixed_view::<3, 3>(0, 0).into_owned();

    let s22_inv = s22.try_inverse()?;
    let s12t = s12.transpose();
    let tmp = s11 - &s12 * &s22_inv * s12t;
    let eig = (c1.try_inverse()? * tmp).symmetric_eigen();

    // Find the eigenvector with the largest positive eigenvalue that gives a valid ellipse
    let mut found = None;
    for i in 0..3 {
        let eval = eig.eigenvalues[i];
        if eval > 1e-6 {
            let a1 = eig.eigenvectors.column(i);
            let a2 = -&s22_inv * s12t * &a1;

            let mut conic = [0.0f64; 6];
            for j in 0..3 {
                conic[j] = a1[j];
                conic[j + 3] = a2[j];
            }

            if 4.0 * conic[0] * conic[2] - conic[1] * conic[1] > 0.0 {
                let scale = (1.0 / conic[5]).abs();
                // Scale so that the conic constant is ~1
                for c in conic.iter_mut() {
                    *c *= scale;
                }
                found = Some(conic);
                break;
            }
        }
    }
    found
}

/// Convert conic coefficients [A, B, C, D, E, F] to ellipse parameters
/// Conic: A*x² + B*x*y + C*y² + D*x + E*y + F = 0
fn conic_to_ellipse_result(conic: &[f64; 6]) -> Option<EllipseResult> {
    let (a, b, c, d, e, f) = (conic[0], conic[1], conic[2], conic[3], conic[4], conic[5]);

    let det = b * b - 4.0 * a * c;
    if det >= 0.0 {
        return None; // not an ellipse
    }

    // Center
    let denom = det;
    if denom.abs() < 1e-12 {
        return None;
    }
    let cx = (2.0 * c * d - b * e) / denom;
    let cy = (2.0 * a * e - b * d) / denom;

    // Rotation angle
    let angle = if (a - c).abs() < 1e-12 {
        0.0
    } else {
        0.5 * (b / (a - c)).atan()
    };

    let cos_t = angle.cos();
    let sin_t = angle.sin();

    // Renormalize F to the center
    let f_center = a * cx * cx + b * cx * cy + c * cy * cy + d * cx + e * cy + f;

    // Normalize conic such that F_center = -1
    let norm = (-f_center).abs().max(1e-12);
    let a_n = a / norm;
    let b_n = b / norm;
    let c_n = c / norm;

    // Semi-axes
    let a_rot = a_n * cos_t * cos_t + b_n * cos_t * sin_t + c_n * sin_t * sin_t;
    let c_rot = a_n * sin_t * sin_t - b_n * cos_t * sin_t + c_n * cos_t * cos_t;

    if a_rot <= 0.0 || c_rot <= 0.0 {
        return None;
    }

    let a_len = (1.0 / a_rot).sqrt();
    let b_len = (1.0 / c_rot).sqrt();

    let (major, minor, rot) = if a_len > b_len {
        (a_len, b_len, angle)
    } else {
        (b_len, a_len, angle + std::f64::consts::FRAC_PI_2)
    };

    Some(EllipseResult {
        center: Point2::new(cx, cy),
        size: (2.0 * major, 2.0 * minor),
        angle: rot.to_degrees(),
    })
}

/// Fallback: fit a circle using linear least squares
fn fit_least_squares_circle(points: &[Point2<f64>]) -> Option<EllipseResult> {
    let n = points.len();
    if n < 3 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for p in points {
        sum_x += p.x;
        sum_y += p.y;
    }
    let cx = sum_x / n as f64;
    let cy = sum_y / n as f64;

    let r = points
        .iter()
        .map(|p| (p.x - cx).hypot(p.y - cy))
        .sum::<f64>()
        / n as f64;

    Some(EllipseResult {
        center: Point2::new(cx, cy),
        size: (2.0 * r, 2.0 * r),
        angle: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ellipse_points(cx: f64, cy: f64, a: f64, b: f64, angle: f64, n: usize) -> Vec<Point2<f64>> {
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n as f64;
                let x = a * t.cos();
                let y = b * t.sin();
                Point2::new(
                    cx + x * angle.cos() - y * angle.sin(),
                    cy + x * angle.sin() + y * angle.cos(),
                )
            })
            .collect()
    }

    #[test]
    fn test_fit_ellipse_circle() {
        let pts = ellipse_points(100.0, 100.0, 50.0, 50.0, 0.0, 100);
        let result = fit_ellipse_direct(&pts).unwrap();
        assert!((result.center.x - 100.0).abs() < 2.0);
        assert!((result.center.y - 100.0).abs() < 2.0);
        assert!((result.size.0 - 100.0).abs() < 5.0);
        assert!((result.size.1 - 100.0).abs() < 5.0);
    }

    #[test]
    fn test_fit_ellipse_rotated() {
        let pts = ellipse_points(50.0, 60.0, 80.0, 30.0, 0.3, 400);
        let result = fit_ellipse(&pts).unwrap();
        assert!((result.center.x - 50.0).abs() < 10.0);
        assert!((result.center.y - 60.0).abs() < 10.0);
        assert!(result.size.0 > 100.0);
        assert!(result.size.1 > 10.0);
    }
}
