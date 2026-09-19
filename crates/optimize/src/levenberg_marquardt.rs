//! Levenberg-Marquardt curve fitting.

use nalgebra::{DMatrix, DVector};

/// Result of a curve-fitting run.
#[derive(Debug, Clone)]
pub struct CurveFitResult {
    /// Optimal parameter vector.
    pub params: Vec<f64>,
    /// Approximate parameter covariance matrix (row-major, n x n).
    pub covariance: Vec<Vec<f64>>,
    /// Residuals at the solution (y_data - model).
    pub residuals: Vec<f64>,
    /// Coefficient of determination.
    pub r_squared: f64,
}

/// Fit a parametric model to data using Levenberg-Marquardt.
///
/// # Arguments
/// * `model` - Function `model(x, params) -> y` to fit.
/// * `x_data` - Independent variable data.
/// * `y_data` - Dependent variable data (same length as `x_data`).
/// * `p0` - Initial parameter guess.
/// * `max_iters` - Maximum number of LM iterations.
#[allow(clippy::needless_range_loop)]
pub fn curve_fit(
    model: impl Fn(f64, &[f64]) -> f64,
    x_data: &[f64],
    y_data: &[f64],
    p0: &[f64],
    max_iters: usize,
) -> Result<CurveFitResult, String> {
    let m = x_data.len();
    let np = p0.len();
    if m != y_data.len() {
        return Err("x_data and y_data must have the same length".into());
    }
    if m < np {
        return Err("Need at least as many data points as parameters".into());
    }

    let mut params = p0.to_vec();
    let mut lambda = 1e-3;
    let eps = 1e-8; // finite-difference step

    let residuals =
        |p: &[f64]| -> Vec<f64> { (0..m).map(|i| y_data[i] - model(x_data[i], p)).collect() };

    let jacobian = |p: &[f64]| -> Vec<Vec<f64>> {
        // J[i][j] = d(model(x_i, p)) / d(p_j)  (note: d(residual)/dp = -J)
        let mut j = vec![vec![0.0; np]; m];
        for k in 0..np {
            let mut p_plus = p.to_vec();
            let h = if p[k].abs() > 1e-12 {
                eps * p[k].abs()
            } else {
                eps
            };
            p_plus[k] += h;
            for i in 0..m {
                j[i][k] = (model(x_data[i], &p_plus) - model(x_data[i], p)) / h;
            }
        }
        j
    };

    let mut r = residuals(&params);
    let mut cost: f64 = r.iter().map(|v| v * v).sum();

    for _ in 0..max_iters {
        let j = jacobian(&params);

        // J^T J  (np x np)
        let mut jtj = vec![vec![0.0; np]; np];
        for i in 0..np {
            for k in 0..np {
                let mut s = 0.0;
                for row in 0..m {
                    s += j[row][i] * j[row][k];
                }
                jtj[i][k] = s;
            }
        }

        // J^T r  (np)
        let mut jtr = vec![0.0; np];
        for i in 0..np {
            let mut s = 0.0;
            for row in 0..m {
                s += j[row][i] * r[row];
            }
            jtr[i] = s;
        }

        // Solve (J^T J + lambda * diag(J^T J)) * dp = J^T r
        let mut a = jtj.clone();
        for i in 0..np {
            a[i][i] += lambda * (jtj[i][i].max(1e-12));
        }

        let dp = match solve_linear(&a, &jtr) {
            Some(v) => v,
            None => break,
        };

        let new_params: Vec<f64> = (0..np).map(|i| params[i] + dp[i]).collect();
        let new_r = residuals(&new_params);
        let new_cost: f64 = new_r.iter().map(|v| v * v).sum();

        if new_cost < cost {
            params = new_params;
            r = new_r;
            cost = new_cost;
            lambda *= 0.1;
        } else {
            lambda *= 10.0;
        }

        // Convergence check
        let dp_norm: f64 = dp.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dp_norm < 1e-10 {
            break;
        }
    }

    // Covariance approximation: (J^T J)^{-1} * (cost / (m - np))
    let j = jacobian(&params);
    let mut jtj = vec![vec![0.0; np]; np];
    for i in 0..np {
        for k in 0..np {
            let mut s = 0.0;
            for row in 0..m {
                s += j[row][i] * j[row][k];
            }
            jtj[i][k] = s;
        }
    }

    let dof = if m > np { m - np } else { 1 };
    let s2 = cost / dof as f64;

    let covariance = match invert_matrix(&jtj) {
        Some(inv) => inv
            .iter()
            .map(|row| row.iter().map(|v| v * s2).collect())
            .collect(),
        None => vec![vec![0.0; np]; np],
    };

    // R-squared
    let y_mean: f64 = y_data.iter().sum::<f64>() / m as f64;
    let ss_tot: f64 = y_data.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 1e-30 {
        1.0 - cost / ss_tot
    } else {
        1.0
    };

    let residuals = r;
    Ok(CurveFitResult {
        params,
        covariance,
        residuals,
        r_squared,
    })
}

/// Solve A * x = b via Gaussian elimination with partial pivoting.
/// Returns None if the system is singular.
///
/// Thin adapter over [`cv_math::linalg::solve`]: the elimination itself (and the
/// Gauss-Jordan inversion below) used to be implemented locally here; both are
/// now the shared `cv-math` linear-algebra routines.
#[allow(clippy::needless_range_loop)]
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mat = DMatrix::from_fn(n, n, |i, j| a[i][j]);
    let rhs = DVector::from_row_slice(b);
    cv_math::linalg::solve(&mat, &rhs)
        .ok()
        .map(|x| x.iter().copied().collect())
}

/// Invert a square matrix via Gauss-Jordan elimination.
#[allow(clippy::needless_range_loop)]
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mat = DMatrix::from_fn(n, n, |i, j| a[i][j]);
    cv_math::linalg::inv(&mat).ok().map(|inv| {
        (0..n)
            .map(|i| (0..n).map(|j| inv[(i, j)]).collect())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_fit_linear() {
        // y = a*x + b, true: a=2, b=1
        let x_data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x + p[1];
        let res = curve_fit(model, &x_data, &y_data, &[0.0, 0.0], 100).unwrap();

        assert!(
            (res.params[0] - 2.0).abs() < 1e-6,
            "a ≈ 2, got {}",
            res.params[0]
        );
        assert!(
            (res.params[1] - 1.0).abs() < 1e-6,
            "b ≈ 1, got {}",
            res.params[1]
        );
        assert!(res.r_squared > 0.9999);
    }

    #[test]
    fn curve_fit_exponential_decay() {
        // y = A * exp(-k * x), true: A=5, k=0.3
        let x_data: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 5.0 * (-0.3 * x).exp()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();
        let res = curve_fit(model, &x_data, &y_data, &[1.0, 0.1], 200).unwrap();

        assert!(
            (res.params[0] - 5.0).abs() < 0.1,
            "A ≈ 5, got {}",
            res.params[0]
        );
        assert!(
            (res.params[1] - 0.3).abs() < 0.01,
            "k ≈ 0.3, got {}",
            res.params[1]
        );
        assert!(res.r_squared > 0.999);
    }
}
