//! Nelder-Mead simplex optimization.

/// Result of an optimization (minimization) run.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Optimal parameter vector.
    pub x: Vec<f64>,
    /// Function value at the optimum.
    pub fun: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver declared convergence.
    pub converged: bool,
}

/// Configuration for the Nelder-Mead simplex algorithm.
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Tolerance on the simplex diameter (parameter space).
    pub x_tol: f64,
    /// Tolerance on the function value spread.
    pub f_tol: f64,
    /// Use dimension-adaptive coefficients (Gao & Han 2012).
    pub adaptive: bool,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            x_tol: 1e-8,
            f_tol: 1e-8,
            adaptive: true,
        }
    }
}

/// Minimize a scalar function using the Nelder-Mead simplex method.
///
/// # Arguments
/// * `f` - Objective function mapping a parameter slice to a scalar.
/// * `x0` - Initial guess.
/// * `config` - Solver configuration.
#[allow(clippy::needless_range_loop)]
pub fn minimize_nelder_mead(
    f: impl Fn(&[f64]) -> f64,
    x0: &[f64],
    config: &NelderMeadConfig,
) -> OptimizeResult {
    let n = x0.len();
    assert!(n > 0, "x0 must be non-empty");

    // Adaptive coefficients (Gao & Han 2012) or standard
    let (alpha, gamma, rho, sigma) = if config.adaptive {
        let nd = n as f64;
        (1.0, 1.0 + 2.0 / nd, 0.75 - 1.0 / (2.0 * nd), 1.0 - 1.0 / nd)
    } else {
        (1.0, 2.0, 0.5, 0.5)
    };

    // Build initial simplex: x0 plus n vertices offset along each axis
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        let delta = if x0[i].abs() > 1e-12 {
            0.05 * x0[i]
        } else {
            0.00025
        };
        v[i] += delta;
        simplex.push(v);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..config.max_iters {
        iters = iter + 1;

        // Sort simplex by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();
        simplex = sorted_simplex;
        fvals = sorted_fvals;

        // Convergence checks
        let f_range = fvals[n] - fvals[0];
        let mut x_range = 0.0f64;
        for i in 1..=n {
            for j in 0..n {
                let d = (simplex[i][j] - simplex[0][j]).abs();
                if d > x_range {
                    x_range = d;
                }
            }
        }
        if f_range < config.f_tol && x_range < config.x_tol {
            converged = true;
            break;
        }

        // Centroid of all vertices except the worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflect
        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            // Expand
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (xr[j] - centroid[j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            // Accept reflection
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contract
            if fr < fvals[n] {
                // Outside contraction
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                if fc <= fr {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // Shrink
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            } else {
                // Inside contraction
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect();
                let fc = f(&xc);
                if fc < fvals[n] {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // Shrink
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    // Final sort
    let best = fvals
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    OptimizeResult {
        x: simplex[best].clone(),
        fun: fvals[best],
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

    #[test]
    fn nelder_mead_rosenbrock() {
        let res = minimize_nelder_mead(rosenbrock, &[-1.0, 1.0], &NelderMeadConfig::default());
        assert!(res.converged, "should converge");
        assert!((res.x[0] - 1.0).abs() < 1e-4, "x ≈ 1, got {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-4, "y ≈ 1, got {}", res.x[1]);
        assert!(res.fun < 1e-8, "f ≈ 0, got {}", res.fun);
    }

    #[test]
    fn nelder_mead_quadratic() {
        // f(x) = (x-3)^2 + (y+2)^2
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let res = minimize_nelder_mead(f, &[0.0, 0.0], &NelderMeadConfig::default());
        assert!(res.converged);
        assert!((res.x[0] - 3.0).abs() < 1e-6);
        assert!((res.x[1] + 2.0).abs() < 1e-6);
    }
}
