//! BFGS quasi-Newton optimization.

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

/// Backtracking line search satisfying the Armijo condition.
///
/// Returns the step size `alpha`.
fn backtracking_line_search(
    f: &impl Fn(&[f64]) -> f64,
    x: &[f64],
    direction: &[f64],
    grad: &[f64],
    max_steps: usize,
) -> f64 {
    let c = 1e-4;
    let rho = 0.5;
    let f0 = f(x);
    let slope: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
    let mut alpha = 1.0;
    let n = x.len();
    let mut x_new = vec![0.0; n];

    for _ in 0..max_steps {
        for i in 0..n {
            x_new[i] = x[i] + alpha * direction[i];
        }
        let f_new = f(&x_new);
        if f_new <= f0 + c * alpha * slope {
            return alpha;
        }
        alpha *= rho;
    }
    alpha
}

/// Minimize a scalar function using the BFGS quasi-Newton method.
///
/// # Arguments
/// * `f` - Objective function.
/// * `grad` - Gradient function returning a vector the same length as `x0`.
/// * `x0` - Initial guess.
/// * `config` - Solver configuration.
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
    let mut h_inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        h_inv[i][i] = 1.0;
    }

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

        // Search direction: d = -H_inv * g
        let mut d = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                d[i] -= h_inv[i][j] * g[j];
            }
        }

        // Line search
        let alpha = backtracking_line_search(&f, &x, &d, &g, config.line_search_max);

        // Step
        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        let x_new: Vec<f64> = (0..n).map(|i| x[i] + s[i]).collect();
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
        }

        x = x_new;
        g = g_new;
        fx = f(&x);
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
