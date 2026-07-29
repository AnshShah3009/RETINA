//! L-BFGS-B (limited-memory BFGS with box constraints).

use super::bfgs::BfgsConfig;
use super::nelder_mead::OptimizeResult;

/// Box bound for a single variable.
#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl Bounds {
    /// No bounds.
    pub fn free() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }
}

/// Project a point onto the feasible box.
fn project(x: &mut [f64], bounds: &[Bounds]) {
    for (xi, b) in x.iter_mut().zip(bounds.iter()) {
        if let Some(lo) = b.lower {
            if *xi < lo {
                *xi = lo;
            }
        }
        if let Some(hi) = b.upper {
            if *xi > hi {
                *xi = hi;
            }
        }
    }
}

/// Minimize a scalar function using L-BFGS-B (limited-memory BFGS with box constraints).
///
/// # Arguments
/// * `f` - Objective function.
/// * `grad` - Gradient function.
/// * `x0` - Initial guess.
/// * `bounds` - Per-variable box constraints.
/// * `m` - Number of correction pairs to store (memory size).
/// * `config` - Solver configuration (shared with BFGS).
pub fn minimize_lbfgsb(
    f: impl Fn(&[f64]) -> f64,
    grad: impl Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    bounds: &[Bounds],
    m: usize,
    config: &BfgsConfig,
) -> OptimizeResult {
    let n = x0.len();
    assert!(n > 0, "x0 must be non-empty");
    assert_eq!(bounds.len(), n, "bounds length must equal x0 length");

    let mut x = x0.to_vec();
    project(&mut x, bounds);
    let mut fx = f(&x);
    let mut g = grad(&x);

    // Storage for L-BFGS pairs
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);

    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..config.max_iters {
        iters = iter + 1;

        // Projected gradient norm for convergence
        let mut pg_norm = 0.0f64;
        for i in 0..n {
            let gi = g[i];
            let projected = {
                let trial = x[i] - gi;
                let mut p = trial;
                if let Some(lo) = bounds[i].lower {
                    if p < lo {
                        p = lo;
                    }
                }
                if let Some(hi) = bounds[i].upper {
                    if p > hi {
                        p = hi;
                    }
                }
                p - x[i]
            };
            pg_norm += projected * projected;
        }
        pg_norm = pg_norm.sqrt();
        if pg_norm < config.gtol {
            converged = true;
            break;
        }

        // Two-loop recursion to compute search direction
        let k = s_hist.len();
        let mut q = g.clone();
        let mut alphas = vec![0.0; k];

        for i in (0..k).rev() {
            alphas[i] = rho_hist[i]
                * s_hist[i]
                    .iter()
                    .zip(q.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            for j in 0..n {
                q[j] -= alphas[i] * y_hist[i][j];
            }
        }

        // Scale by gamma = s^T y / y^T y of most recent pair
        let gamma = if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
            let sy: f64 = s_last.iter().zip(y_last.iter()).map(|(a, b)| a * b).sum();
            let yy: f64 = y_last.iter().map(|v| v * v).sum::<f64>();
            if yy > 1e-30 {
                sy / yy
            } else {
                1.0
            }
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|v| v * gamma).collect();

        for i in 0..k {
            let beta = rho_hist[i]
                * y_hist[i]
                    .iter()
                    .zip(r.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            for j in 0..n {
                r[j] += s_hist[i][j] * (alphas[i] - beta);
            }
        }

        // Negate for descent direction
        let d: Vec<f64> = r.iter().map(|v| -v).collect();

        // Projected line search
        let mut alpha_step = 1.0;
        let c1 = 1e-4;
        let slope: f64 = g.iter().zip(d.iter()).map(|(a, b)| a * b).sum();

        let mut x_new = vec![0.0; n];
        let mut found = false;
        for _ in 0..config.line_search_max {
            for i in 0..n {
                x_new[i] = x[i] + alpha_step * d[i];
            }
            project(&mut x_new, bounds);
            let f_new = f(&x_new);
            if f_new <= fx + c1 * alpha_step * slope {
                fx = f_new;
                found = true;
                break;
            }
            alpha_step *= 0.5;
        }

        if !found {
            // Accept the last tried point anyway
            for i in 0..n {
                x_new[i] = x[i] + alpha_step * d[i];
            }
            project(&mut x_new, bounds);
            fx = f(&x_new);
        }

        let g_new = grad(&x_new);

        let s_vec: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let y_vec: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();

        if sy > 1e-18 {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        g = g_new;
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

    #[test]
    fn lbfgsb_bounded() {
        // Minimize (x-3)^2 + (y+2)^2 with x in [0, 2], y in [-1, 1]
        // Unconstrained optimum is (3, -2), but bounded optimum is (2, -1)
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)];
        let bounds = vec![
            Bounds {
                lower: Some(0.0),
                upper: Some(2.0),
            },
            Bounds {
                lower: Some(-1.0),
                upper: Some(1.0),
            },
        ];
        let config = BfgsConfig::default();
        let res = minimize_lbfgsb(f, g, &[0.0, 0.0], &bounds, 10, &config);
        assert!((res.x[0] - 2.0).abs() < 1e-4, "x ≈ 2, got {}", res.x[0]);
        assert!((res.x[1] + 1.0).abs() < 1e-4, "y ≈ -1, got {}", res.x[1]);
    }

    #[test]
    fn lbfgsb_unbounded_quadratic() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let bounds = vec![Bounds::free(), Bounds::free()];
        let res = minimize_lbfgsb(f, g, &[5.0, -3.0], &bounds, 10, &BfgsConfig::default());
        assert!(res.converged);
        assert!((res.x[0]).abs() < 1e-4);
        assert!((res.x[1]).abs() < 1e-4);
    }
}
