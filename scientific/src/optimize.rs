//! General-purpose optimization routines.
//!
//! Provides minimizers (Nelder-Mead, BFGS, L-BFGS-B), curve fitting (Levenberg-Marquardt),
//! root finding (Brent's method, Newton's method), and a unified `minimize` dispatch API.
//!
//! All implementations are from scratch with no external optimization dependencies.

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Nelder-Mead
// ---------------------------------------------------------------------------

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
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));
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
        let xr: Vec<f64> = (0..n).map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j])).collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            // Expand
            let xe: Vec<f64> = (0..n).map(|j| centroid[j] + gamma * (xr[j] - centroid[j])).collect();
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
                let xc: Vec<f64> = (0..n).map(|j| centroid[j] + rho * (xr[j] - centroid[j])).collect();
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
                let xc: Vec<f64> = (0..n).map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j])).collect();
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

// ---------------------------------------------------------------------------
// BFGS
// ---------------------------------------------------------------------------

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
                    h_inv[i][j] += rho_val * ((1.0 + rho_val * yhy) * s[i] * s[j]
                        - hy[i] * s[j]
                        - s[i] * hy[j]);
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

// ---------------------------------------------------------------------------
// L-BFGS-B
// ---------------------------------------------------------------------------

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
            if yy > 1e-30 { sy / yy } else { 1.0 }
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

// ---------------------------------------------------------------------------
// Curve Fitting (Levenberg-Marquardt)
// ---------------------------------------------------------------------------

/// Fit a parametric model to data using Levenberg-Marquardt.
///
/// # Arguments
/// * `model` - Function `model(x, params) -> y` to fit.
/// * `x_data` - Independent variable data.
/// * `y_data` - Dependent variable data (same length as `x_data`).
/// * `p0` - Initial parameter guess.
/// * `max_iters` - Maximum number of LM iterations.
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

    let residuals = |p: &[f64]| -> Vec<f64> {
        (0..m).map(|i| y_data[i] - model(x_data[i], p)).collect()
    };

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
        Some(inv) => inv.iter().map(|row| row.iter().map(|v| v * s2).collect()).collect(),
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

// ---------------------------------------------------------------------------
// Root Finding
// ---------------------------------------------------------------------------

/// Find a root of `f` in the bracket `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs.
pub fn brentq(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err("f(a) and f(b) must have opposite signs".into());
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let fc = fa;
    let mut mflag = true;
    let mut d = 0.0; // will be set before used

    for _ in 0..max_iter {
        if fb.abs() < tol {
            return Ok(b);
        }
        if fa.abs() < tol {
            return Ok(a);
        }
        if (b - a).abs() < tol {
            return Ok(b);
        }

        let s;
        if (fa - fc).abs() > 1e-30 && (fb - fc).abs() > 1e-30 {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        let cond1 = {
            let lo = (3.0 * a + b) / 4.0;
            let (min_ab, max_ab) = if lo < b { (lo, b) } else { (b, lo) };
            s < min_ab || s > max_ab
        };
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            // Bisection
            let s_new = (a + b) / 2.0;
            mflag = true;
            d = c; // safe: d used only when mflag is false on next iteration, and we set c below
            c = b;
            let fs = f(s_new);
            if fa * fs < 0.0 {
                b = s_new;
                fb = fs;
            } else {
                a = s_new;
                fa = fs;
            }
        } else {
            mflag = false;
            d = c;
            c = b;
            let fs = f(s);
            if fa * fs < 0.0 {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    Ok(b)
}

/// Find a root of `f` using Newton's method.
///
/// # Arguments
/// * `f` - Function whose root is sought.
/// * `fprime` - Derivative of `f`.
/// * `x0` - Initial guess.
/// * `tol` - Convergence tolerance.
/// * `max_iter` - Maximum iterations.
pub fn newton(
    f: impl Fn(f64) -> f64,
    fprime: impl Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return Ok(x);
        }
        let fp = fprime(x);
        if fp.abs() < 1e-30 {
            return Err("Derivative is zero; Newton's method cannot continue".into());
        }
        let x_new = x - fx / fp;
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }
        x = x_new;
    }
    Ok(x)
}

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
pub fn minimize(
    f: impl Fn(&[f64]) -> f64,
    x0: &[f64],
    method: Method,
    grad: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
) -> OptimizeResult {
    match method {
        Method::NelderMead => {
            minimize_nelder_mead(&f, x0, &NelderMeadConfig::default())
        }
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

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0].powi(2)) * (-2.0 * x[0]);
        let dy = 200.0 * (x[1] - x[0].powi(2));
        vec![dx, dy]
    }

    // ---- Nelder-Mead ----

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

    // ---- BFGS ----

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

    // ---- L-BFGS-B ----

    #[test]
    fn lbfgsb_bounded() {
        // Minimize (x-3)^2 + (y+2)^2 with x in [0, 2], y in [-1, 1]
        // Unconstrained optimum is (3, -2), but bounded optimum is (2, -1)
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)];
        let bounds = vec![
            Bounds { lower: Some(0.0), upper: Some(2.0) },
            Bounds { lower: Some(-1.0), upper: Some(1.0) },
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

    // ---- Curve Fitting ----

    #[test]
    fn curve_fit_linear() {
        // y = a*x + b, true: a=2, b=1
        let x_data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x + p[1];
        let res = curve_fit(model, &x_data, &y_data, &[0.0, 0.0], 100).unwrap();

        assert!((res.params[0] - 2.0).abs() < 1e-6, "a ≈ 2, got {}", res.params[0]);
        assert!((res.params[1] - 1.0).abs() < 1e-6, "b ≈ 1, got {}", res.params[1]);
        assert!(res.r_squared > 0.9999);
    }

    #[test]
    fn curve_fit_exponential_decay() {
        // y = A * exp(-k * x), true: A=5, k=0.3
        let x_data: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 5.0 * (-0.3 * x).exp()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();
        let res = curve_fit(model, &x_data, &y_data, &[1.0, 0.1], 200).unwrap();

        assert!((res.params[0] - 5.0).abs() < 0.1, "A ≈ 5, got {}", res.params[0]);
        assert!((res.params[1] - 0.3).abs() < 0.01, "k ≈ 0.3, got {}", res.params[1]);
        assert!(res.r_squared > 0.999);
    }

    // ---- Brent's method ----

    #[test]
    fn brentq_sqrt2() {
        // Root of x^2 - 2 = 0 in [1, 2] => sqrt(2)
        let root = brentq(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!(
            (root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root ≈ √2, got {}",
            root
        );
    }

    #[test]
    fn brentq_cubic() {
        // Root of x^3 - x - 2 = 0 near x ≈ 1.5214
        let root = brentq(|x| x.powi(3) - x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root.powi(3) - root - 2.0).abs() < 1e-10);
    }

    // ---- Newton's method ----

    #[test]
    fn newton_cos_x_minus_x() {
        // Root of cos(x) - x = 0 (Dottie number ≈ 0.7390851332)
        let root = newton(
            |x| x.cos() - x,
            |x| -x.sin() - 1.0,
            0.5,
            1e-12,
            100,
        )
        .unwrap();
        assert!(
            (root.cos() - root).abs() < 1e-10,
            "cos(root) should equal root, got {}",
            root
        );
        assert!((root - 0.7390851332).abs() < 1e-8);
    }

    #[test]
    fn newton_square_root() {
        // Root of x^2 - 5 = 0 => sqrt(5)
        let root = newton(|x| x * x - 5.0, |x| 2.0 * x, 2.0, 1e-12, 100).unwrap();
        assert!((root - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    // ---- Unified minimize ----

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
