//! BFGS quasi-Newton optimization.

use super::lbfgsb::line_search;
use super::nelder_mead::OptimizeResult;

/// Configuration for the BFGS and L-BFGS-B algorithms.
#[derive(Debug, Clone)]
pub struct BfgsConfig {
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Gradient-norm tolerance for convergence.
    pub gtol: f64,
    /// Maximum number of line-search steps.
    pub line_search_max: usize,
}

impl Default for BfgsConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            gtol: 1e-5,
            line_search_max: 20,
        }
    }
}

/// Minimize a scalar function using the BFGS quasi-Newton method.
///
/// # Arguments
/// * `f` - Objective function.
/// * `grad` - Gradient function returning a vector the same length as `x0`.
/// * `x0` - Initial guess.
/// * `config` - Solver configuration.
///
/// Each step is found by a backtracking Armijo search with safeguarded
/// quadratic interpolation (the one `minimize_lbfgsb` uses), and is taken only
/// if it decreases `f`. When no such step exists along `-H g` (nor, if `H` has
/// been updated, along `-g`), the solver stops where it is with
/// `converged = false`, as SciPy's BFGS stops with status 2 ("Desired error
/// not necessarily achieved due to precision loss"). It used to take the last
/// halved step regardless, which could go uphill: on `f = 1e9 x^2` from
/// `x = 1` the first step went to `x = -1906`, `f = 3.6e15`.
#[allow(clippy::needless_range_loop)]
pub fn minimize_bfgs(
    f: impl Fn(&[f64]) -> f64,
    grad: impl Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    config: &BfgsConfig,
) -> OptimizeResult {
    let n = x0.len();
    assert!(n > 0, "x0 must be non-empty");

    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut fx = f(&x);

    // Inverse Hessian approximation (row-major n x n), start with identity
    let identity = |n: usize| {
        let mut h = vec![vec![0.0; n]; n];
        for i in 0..n {
            h[i][i] = 1.0;
        }
        h
    };
    let mut h_inv = identity(n);
    let mut h_is_identity = true;

    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..config.max_iters {
        iters = iter + 1;

        // Check gradient norm
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < config.gtol {
            converged = true;
            break;
        }

        // Search direction d = -H_inv g and a step along it that decreases f.
        // If the quasi-Newton direction yields none, retry once along steepest
        // descent with H reset; if that fails too, stop.
        let mut step = None;
        for attempt in 0..2 {
            if attempt == 1 {
                if h_is_identity {
                    break;
                }
                h_inv = identity(n);
                h_is_identity = true;
            }
            let mut d = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    d[i] -= h_inv[i][j] * g[j];
                }
            }
            let slope: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
            if !(slope < 0.0) {
                continue;
            }
            if let Some((alpha, f_new, x_new)) =
                line_search(&f, &x, fx, &d, slope, 1.0, config.line_search_max)
            {
                step = Some((alpha, d, f_new, x_new));
                break;
            }
        }
        let Some((alpha, d, f_new, x_new)) = step else {
            break;
        };

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        let g_new = grad(&x_new);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy > 1e-18 {
            // BFGS update of inverse Hessian
            // H' = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
            let rho_val = 1.0 / sy;

            // Compute H*y
            let mut hy = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    hy[i] += h_inv[i][j] * y[j];
                }
            }

            let yhy: f64 = y.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();

            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] += rho_val
                        * ((1.0 + rho_val * yhy) * s[i] * s[j] - hy[i] * s[j] - s[i] * hy[j]);
                }
            }
            h_is_identity = false;
        }

        x = x_new;
        g = g_new;
        fx = f_new;
    }

    OptimizeResult {
        x,
        fun: fx,
        iterations: iters,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0].powi(2)) * (-2.0 * x[0]);
        let dy = 200.0 * (x[1] - x[0].powi(2));
        vec![dx, dy]
    }

    #[test]
    fn bfgs_quadratic() {
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)];
        let res = minimize_bfgs(f, g, &[0.0, 0.0], &BfgsConfig::default());
        assert!(res.converged);
        assert!((res.x[0] - 3.0).abs() < 1e-6);
        assert!((res.x[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn bfgs_rosenbrock() {
        let res = minimize_bfgs(
            rosenbrock,
            rosenbrock_grad,
            &[-1.0, 1.0],
            &BfgsConfig {
                max_iters: 500,
                ..BfgsConfig::default()
            },
        );
        assert!((res.x[0] - 1.0).abs() < 1e-3, "x ≈ 1, got {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-3, "y ≈ 1, got {}", res.x[1]);
    }
}
