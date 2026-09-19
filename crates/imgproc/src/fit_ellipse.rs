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


/// Conic vector <-> symmetric quad-form matrix (packed halves: xy, xz, yz
/// entries carry factor 2 in the polynomial).
fn conic_to_mat(conic: &[f64; 6]) -> Matrix3<f64> {
    Matrix3::new(
        conic[0], conic[1] / 2.0, conic[3] / 2.0,
        conic[1] / 2.0, conic[2], conic[4] / 2.0,
        conic[3] / 2.0, conic[4] / 2.0, conic[5],
    )
}

fn mat_to_conic(m: &Matrix3<f64>) -> [f64; 6] {
    [
        m[(0, 0)],
        2.0 * m[(0, 1)],
        m[(1, 1)],
        2.0 * m[(0, 2)],
        2.0 * m[(1, 2)],
        m[(2, 2)],
    ]
}

/// Denormalize a conic fitted on Hartley-normalized points back to the
/// original coordinate system.
fn denormalize_conic(t_norm: &Matrix3<f64>, conic_norm: &[f64; 6]) -> [f64; 6] {
    let mn = conic_to_mat(conic_norm);
    let mo = t_norm.transpose() * mn * t_norm;
    mat_to_conic(&mo)
}

/// Hartley point normalization: returns (T, normalized_points) with
/// centroid at origin and mean radius sqrt(2). T maps ORIGINAL -> normalized.
fn hartley_normalize(points: &[Point2<f64>]) -> (Matrix3<f64>, Vec<Point2<f64>>) {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.x).sum::<f64>() / n;
    let my = points.iter().map(|p| p.y).sum::<f64>() / n;
    let rms = (points
        .iter()
        .map(|p| {
            let dx = p.x - mx;
            let dy = p.y - my;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n)
        .max(1e-12);
    let sc = std::f64::consts::SQRT_2 / rms;

    let t = Matrix3::new(sc, 0.0, -sc * mx, 0.0, sc, -sc * my, 0.0, 0.0, 1.0);
    let out = points
        .iter()
        .map(|p| Point2::new((p.x - mx) * sc, (p.y - my) * sc))
        .collect();
    (t, out)
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

    let (t_norm, pts_norm) = hartley_normalize(points);
    let points = &pts_norm;

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

    let conic_norm = solve_ams_eigen(&dtd, &dxtdx_plus_dytdy)?;
    let conic = denormalize_conic(&t_norm, &conic_norm);
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

    // Hartley normalization: centroid at origin, RMS distance sqrt(2).
    // Pixel-scale coordinates make S entries ~1e8+ and destroy the small
    // positive eigenvalue the method depends on.
    let (t_norm, pts_norm) = hartley_normalize(points);
    let points = &pts_norm;

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
    let conic_norm = solve_direct_eigen(&s, &c)?;
    let conic = denormalize_conic(&t_norm, &conic_norm);
    conic_to_ellipse_result(&conic)
}

/// Solve the AMS generalized eigenvalue problem
fn solve_ams_eigen(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Option<[f64; 6]> {
    let b = b.clone();
    // Solve the generalized eigenvalue problem A*u = λ*B*u.
    // A and B are both symmetric and B is SPD (a sum of outer products), so
    // reduce to an ordinary SYMMETRIC problem via Cholesky: with B = L·Lᵀ,
    // C = L⁻¹AL⁻ᵀ shares the eigenvalues and u = L⁻ᵀz recovers vectors.
    // (B⁻¹A itself is NOT symmetric — symmetric_eigen on it solves the wrong
    // problem.)
    let n_dim = b.nrows();
    let chol_b = b.cholesky()?;
    // Column j of L⁻ᵀ is the solution of B x = e_j (since B = L·Lᵀ ⇒
    // B⁻¹ = L⁻ᵀL⁻¹ is symmetric, so B⁻¹'s columns are L⁻ᵀ's columns).
    let mut linv_t = nalgebra::DMatrix::<f64>::zeros(n_dim, n_dim);
    for j in 0..n_dim {
        let mut e = nalgebra::DVector::<f64>::zeros(n_dim);
        e[j] = 1.0;
        let col = chol_b.solve(&e);
        linv_t.set_column(j, &col);
    }

    let c_sym = {
        let tmp = &linv_t.transpose() * a;
        &tmp * &linv_t
    };
    // Symmetrize away rounding drift.
    let c_sym = (&c_sym + c_sym.transpose()) * 0.5;

    let eig = c_sym.symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[j]
            .partial_cmp(&eig.eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &idx in &order {
        let z = eig.eigenvectors.column(idx);
        let u = &linv_t * z; // u = L⁻ᵀ z
        let conic: [f64; 6] = [u[0], u[1], u[2], u[3], u[4], u[5]];
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
    // C1⁻¹·S' is NOT symmetric; symmetric_eigen previously solved the wrong
    // problem. Solve the 3x3 eigenproblem exactly via its characteristic
    // cubic (trigonometric form) and pick the LARGEST positive eigenvalue.
    let m_red = c1.try_inverse()? * tmp;

    let pairs = eigenpairs_3x3(&m_red);

    let mut found = None;
    // No sign gate: for circle-like data every reduced eigenvalue can be
    // negative while the largest still carries the valid ellipse conic.
    // The 4ac−b²>0 condition below is the real filter.
    for &(_eval, ref evec) in pairs.iter() {
        let a1 = evec;
        {
            let a1 = a1;
            let a2 = -&s22_inv * s12t * a1;

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

/// Real eigen-decomposition of a general 3x3 matrix: closed-form cubic roots
/// plus cross-product null-space eigenvectors. Returns pairs sorted by
/// descending eigenvalue; complex-conjugate pairs are skipped.
fn eigenpairs_3x3(m: &Matrix3<f64>) -> Vec<(f64, nalgebra::Vector3<f64>)> {
    use std::f64::consts::PI;

    let tr = m.trace();
    // Sum of principal 2x2 minors:
    let b = m[(1, 1)] * m[(2, 2)] + m[(0, 0)] * m[(2, 2)] + m[(0, 0)] * m[(1, 1)]
        - m[(1, 2)] * m[(2, 1)]
        - m[(0, 2)] * m[(2, 0)]
        - m[(0, 1)] * m[(1, 0)];
    let det = m.determinant();

    // Depressed cubic t³ + P t + Q = 0 for λ = t + tr/3.
    let p = b - tr * tr / 3.0;
    let q = -tr * tr * tr / 13.5 + tr * b / 3.0 - det;

    let disc = -4.0 * p * p * p - 27.0 * q * q;
    let mut out = Vec::with_capacity(3);

    // Near-zero discriminant (relative to the term magnitudes) sits at the
    // repeated-root boundary where rounding flips its sign; clamp into the
    // trig branch there instead of dropping to the single-root Cardano path
    // and losing the coincident real roots.
    let disc_tol = 1e-9_f64
        * (4.0 * p.abs().powi(3) + 27.0 * q.abs()).max(1.0);
    if disc >= -disc_tol && p.abs() > 1e-15 {
        // Three real roots (trigonometric form).
        let mm = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * mm))
            .clamp(-1.0, 1.0)
            .acos()
            / 3.0;
        for k in 0..3usize {
            let lam = mm * (theta - 2.0 * PI * k as f64 / 3.0).cos() + tr / 3.0;
            if let Some(v) = eigvec_for_lambda(m, lam) {
                out.push((lam, v));
            }
        }
    } else {
        // One real root (Cardano).
        let sq = (q * q / 4.0 + p * p * p / 27.0).max(0.0).sqrt();
        let lam = (-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt() + tr / 3.0;
        if let Some(v) = eigvec_for_lambda(m, lam) {
            out.push((lam, v));
        }
    }

    out.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Null-space vector of (M − λI) via the largest-magnitude row cross product.
fn eigvec_for_lambda(m: &Matrix3<f64>, lam: f64) -> Option<nalgebra::Vector3<f64>> {
    let d = m - Matrix3::identity() * lam;
    let r0 = nalgebra::Vector3::new(d[(0, 0)], d[(0, 1)], d[(0, 2)]);
    let r1 = nalgebra::Vector3::new(d[(1, 0)], d[(1, 1)], d[(1, 2)]);
    let r2 = nalgebra::Vector3::new(d[(2, 0)], d[(2, 1)], d[(2, 2)]);

    let c01 = r0.cross(&r1);
    let c02 = r0.cross(&r2);
    let c12 = r1.cross(&r2);

    let n01 = c01.norm();
    let n02 = c02.norm();
    let n12 = c12.norm();

    let best = [(n01, c01), (n02, c02), (n12, c12)]
        .into_iter()
        .fold(None::<(f64, nalgebra::Vector3<f64>)>, |acc, (n, v)| {
            match acc {
                Some((bn, _)) if bn >= n => acc,
                _ => Some((n, v)),
            }
        })?;

    let (_, v) = best;
    let n = v.norm();
    if !n.is_finite() || n < 1e-14 {
        return None;
    }
    Some(v / n)
}

/// Convert conic coefficients [A, B, C, D, E, F] to ellipse parameters
/// Conic: A*x² + B*x*y + C*y² + D*x + E*y + F = 0
fn conic_to_ellipse_result(conic: &[f64; 6]) -> Option<EllipseResult> {
    let (a, b, c, d, e, f) = (conic[0], conic[1], conic[2], conic[3], conic[4], conic[5]);
    let det = b * b - 4.0 * a * c;
    if det >= 0.0 {
        return None; // not an ellipse
    }

    // Center for UNPACKED coefficients (poly = Ax²+Bxy+Cy²+Dx+Ey+F):
    // solve [2A B; B 2C]·[x,y]ᵀ = [−D,−E]ᵀ.
    let denom = 4.0 * a * c - b * b;
    if !denom.is_finite() || denom.abs() < 1e-15 {
        return None;
    }
    let cx = (b * e - 2.0 * c * d) / denom;
    let cy = (b * d - 2.0 * a * e) / denom;

    // Rotation angle: tan(2θ) = b/(a−c); atan2 resolves a≈c (θ=45° when b≠0).
    let angle = 0.5 * b.atan2(a - c);

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

    // Definiteness (guaranteed for a real ellipse by the det<0 check above)
    // means a_rot/c_rot share one sign; its direction depends on the
    // arbitrary eigenvector sign, so take magnitudes.
    if !(a_rot.abs() > 1e-15 && c_rot.abs() > 1e-15) {
        return None;
    }

    let a_len = (1.0 / a_rot.abs()).sqrt();
    let b_len = (1.0 / c_rot.abs()).sqrt();

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
