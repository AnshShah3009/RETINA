//! Levenberg-Marquardt curve fitting.

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
#[allow(clippy::needless_range_loop)]
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = a[i].clone();
        row.push(b[i]);
        aug.push(row);
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] = s / aug[i][i];
    }
    Some(x)
}

/// Invert a square matrix via Gauss-Jordan elimination.
#[allow(clippy::needless_range_loop)]
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    // Augment with identity
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = a[i].clone();
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        aug.push(row);
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    let inv: Vec<Vec<f64>> = aug.iter().map(|row| row[n..].to_vec()).collect();
    Some(inv)
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
