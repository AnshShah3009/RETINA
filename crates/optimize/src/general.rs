//! General-purpose optimization routines.
//!
//! Provides minimizers (Nelder-Mead, BFGS, L-BFGS-B), curve fitting (Levenberg-Marquardt),
//! root finding (Brent's method, Newton's method), and a unified `minimize` dispatch API.
//!
//! All implementations are from scratch with no external optimization dependencies.

#[path = "nelder_mead.rs"]
pub mod nelder_mead;
#[path = "bfgs.rs"]
pub mod bfgs;
#[path = "lbfgsb.rs"]
pub mod lbfgsb;
#[path = "levenberg_marquardt.rs"]
pub mod levenberg_marquardt;
#[path = "brent.rs"]
pub mod brent;
#[path = "newton.rs"]
pub mod newton;

pub use nelder_mead::*;
pub use bfgs::*;
pub use lbfgsb::*;
pub use levenberg_marquardt::*;
pub use brent::*;
pub use newton::*;

// ---------------------------------------------------------------------------
// Unified minimize API
// ---------------------------------------------------------------------------

/// Method selector for the unified [`minimize`] function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    NelderMead,
    Bfgs,
    LBfgsB,
}

/// Compute a numerical gradient via central differences.
fn numerical_gradient(f: &impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let eps = 1e-8;
    let mut g = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for i in 0..n {
        let h = if x[i].abs() > 1e-12 {
            eps * x[i].abs()
        } else {
            eps
        };
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        g[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    g
}

/// Unified minimization interface dispatching to the chosen solver.
///
/// If `grad` is `None` and the method requires gradients, a numerical gradient
/// via central differences is used automatically.
#[allow(clippy::type_complexity)]
pub fn minimize(
    f: impl Fn(&[f64]) -> f64,
    x0: &[f64],
    method: Method,
    grad: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
) -> OptimizeResult {
    match method {
        Method::NelderMead => minimize_nelder_mead(&f, x0, &NelderMeadConfig::default()),
        Method::Bfgs => {
            let config = BfgsConfig::default();
            match grad {
                Some(g) => minimize_bfgs(&f, g, x0, &config),
                None => {
                    let g = |x: &[f64]| numerical_gradient(&f, x);
                    minimize_bfgs(&f, g, x0, &config)
                }
            }
        }
        Method::LBfgsB => {
            let config = BfgsConfig::default();
            let bounds: Vec<Bounds> = (0..x0.len()).map(|_| Bounds::free()).collect();
            match grad {
                Some(g) => minimize_lbfgsb(&f, g, x0, &bounds, 10, &config),
                None => {
                    let g = |x: &[f64]| numerical_gradient(&f, x);
                    minimize_lbfgsb(&f, g, x0, &bounds, 10, &config)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_dispatch_nelder_mead() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let res = minimize(f, &[0.0, 0.0], Method::NelderMead, None);
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1e-4);
        assert!((res.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn minimize_dispatch_bfgs_numerical_grad() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let res = minimize(f, &[0.0, 0.0], Method::Bfgs, None);
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1e-4);
        assert!((res.x[1] - 2.0).abs() < 1e-4);
    }
}
