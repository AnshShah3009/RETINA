//! L-BFGS-B (limited-memory BFGS with box constraints).
//!
//! Byrd, Lu, Nocedal & Zhu (1995), "A limited memory algorithm for bound
//! constrained optimization", with the Morales-Nocedal (2011) projection step
//! after the subspace minimisation - the algorithm behind SciPy's
//! `minimize(method='L-BFGS-B')`. Each iteration:
//!
//! 1. **Generalized Cauchy point.** Walk the projected steepest-descent path
//!    `P(x - t g)` breakpoint by breakpoint, minimising the compact L-BFGS
//!    model `m(z) = gᵀ(z-x) + ½(z-x)ᵀB(z-x)` with `B = θI - W M Wᵀ`, until the
//!    first local minimiser `x_c`. Variables at a bound there form the active
//!    set.
//! 2. **Subspace minimisation.** Minimise the model over the free variables
//!    with the active ones fixed (direct primal method, Sherman-Morrison-
//!    Woodbury on the compact form), project onto the box, and fall back to
//!    the truncated step if the projection is not a descent direction.
//! 3. **Line search** on `x + α(x̄ - x)`, `α ∈ (0, 1]` - always feasible, so the
//!    Armijo slope `gᵀ(x̄ - x)` is the slope of the path actually searched.
//!    A step that does not decrease `f` is never accepted: the memory is
//!    dropped and the iteration retried from the Cauchy direction; if that
//!    fails too the solver stops (`converged = false`) at the best point.
//!
//! Convergence: the projected gradient `‖P(x - g) - x‖∞ ≤ gtol` (SciPy's
//! `pgtol`), or a relative reduction `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1)
//! ≤ 2.22e-9` (SciPy's default `factr · eps`).
//!
//! The previous implementation ran the unconstrained two-loop recursion,
//! projected the trial point, took the Armijo slope from the unprojected
//! direction and accepted a failed line search even when `f` increased: on
//! `f = (x+1)² + 10(y-1)² + 3xy`, `x ≥ 0`, from `(2, 5)` it returned
//! `f = 1.01345` unconverged after 200 iterations (SciPy: `(0, 1)`, `f = 1`).

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

/// SciPy's default `ftol = factr * eps = 1e7 * 2.22e-16`.
const FTOL: f64 = 2.220446049250313e-9;

fn lo(b: &Bounds) -> f64 {
    b.lower.unwrap_or(f64::NEG_INFINITY)
}

fn hi(b: &Bounds) -> f64 {
    b.upper.unwrap_or(f64::INFINITY)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Project a point onto the feasible box.
fn project(x: &mut [f64], bounds: &[Bounds]) {
    for (xi, b) in x.iter_mut().zip(bounds.iter()) {
        *xi = xi.max(lo(b)).min(hi(b));
    }
}

/// `‖P(x - g) - x‖∞`.
fn projected_gradient_norm(x: &[f64], g: &[f64], bounds: &[Bounds]) -> f64 {
    x.iter()
        .zip(g)
        .zip(bounds)
        .map(|((&xi, &gi), b)| ((xi - gi).max(lo(b)).min(hi(b)) - xi).abs())
        .fold(0.0, f64::max)
}

/// Solve the dense system `a z = r` (Gaussian elimination, partial pivoting).
/// Returns `None` if `a` is numerically singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut r: Vec<f64>) -> Option<Vec<f64>> {
    let n = r.len();
    let scale = a
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() <= 1e-14 * scale {
            return None;
        }
        a.swap(col, piv);
        r.swap(col, piv);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            if factor != 0.0 {
                for k in col..n {
                    a[row][k] -= factor * a[col][k];
                }
                r[row] -= factor * r[col];
            }
        }
    }
    let mut z = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row][k] * z[k]).sum();
        z[row] = (r[row] - s) / a[row][row];
    }
    Some(z)
}

/// Compact representation `B = θI - W M Wᵀ` of the L-BFGS matrix built from
/// the stored pairs (Byrd, Nocedal & Schnabel 1994).
struct Compact {
    theta: f64,
    /// `W = [Y, θS]`, stored row-wise: `w[i]` is the 2k-vector for variable i.
    w: Vec<Vec<f64>>,
    /// `M` (2k x 2k), the inverse of `[[-D, Lᵀ], [L, θ SᵀS]]`.
    m: Vec<Vec<f64>>,
}

impl Compact {
    fn new(n: usize, s: &[Vec<f64>], y: &[Vec<f64>]) -> Option<Self> {
        let k = s.len();
        if k == 0 {
            return Some(Self {
                theta: 1.0,
                w: vec![Vec::new(); n],
                m: Vec::new(),
            });
        }
        let (s_last, y_last) = (&s[k - 1], &y[k - 1]);
        let theta = dot(y_last, y_last) / dot(s_last, y_last);
        let w = (0..n)
            .map(|i| {
                let mut row = Vec::with_capacity(2 * k);
                row.extend(y.iter().map(|yj| yj[i]));
                row.extend(s.iter().map(|sj| theta * sj[i]));
                row
            })
            .collect();
        // Middle matrix before inversion.
        let mut a = vec![vec![0.0; 2 * k]; 2 * k];
        for i in 0..k {
            a[i][i] = -dot(&s[i], &y[i]);
            for j in 0..k {
                // L_ij = s_iᵀ y_j for i > j.
                let l_ij = if i > j { dot(&s[i], &y[j]) } else { 0.0 };
                a[k + i][j] = l_ij;
                a[j][k + i] = l_ij;
                a[k + i][k + j] = theta * dot(&s[i], &s[j]);
            }
        }
        let mut m = vec![vec![0.0; 2 * k]; 2 * k];
        for c in 0..2 * k {
            let mut e = vec![0.0; 2 * k];
            e[c] = 1.0;
            let col = solve_dense(a.clone(), e)?;
            for r in 0..2 * k {
                m[r][c] = col[r];
            }
        }
        Some(Self { theta, w, m })
    }

    fn mv(&self, v: &[f64]) -> Vec<f64> {
        self.m.iter().map(|row| dot(row, v)).collect()
    }
}

/// Generalized Cauchy point (BLNZ Algorithm CP). Returns `x_c` and
/// `c = Wᵀ(x_c - x)`.
fn cauchy_point(x: &[f64], g: &[f64], bounds: &[Bounds], b: &Compact) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let k2 = b.m.len();
    let mut d = vec![0.0; n];
    let mut breaks: Vec<(f64, usize)> = Vec::new();
    for i in 0..n {
        let t = if g[i] < 0.0 {
            (x[i] - hi(&bounds[i])) / g[i]
        } else if g[i] > 0.0 {
            (x[i] - lo(&bounds[i])) / g[i]
        } else {
            f64::INFINITY
        };
        if t > 0.0 {
            d[i] = -g[i];
            if t.is_finite() {
                breaks.push((t, i));
            }
        }
    }
    breaks.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut xc = x.to_vec();
    let mut p = vec![0.0; k2];
    for i in 0..n {
        if d[i] != 0.0 {
            for (pj, wj) in p.iter_mut().zip(&b.w[i]) {
                *pj += wj * d[i];
            }
        }
    }
    let mut c = vec![0.0; k2];
    let mut f1 = -dot(&d, &d);
    let mut f2 = -b.theta * f1 - dot(&p, &b.mv(&p));
    let f2_org = f2;
    let mut dt_min = if f2 > 0.0 { -f1 / f2 } else { f64::INFINITY };
    let mut t_old = 0.0;

    let mut next = 0;
    while next < breaks.len() {
        let (t, bi) = breaks[next];
        let dt = t - t_old;
        if dt_min < dt {
            break;
        }
        // Variable `bi` hits its bound at `t`.
        let xb = if d[bi] > 0.0 {
            hi(&bounds[bi])
        } else {
            lo(&bounds[bi])
        };
        let zb = xb - x[bi];
        xc[bi] = xb;
        let gb = g[bi];
        for (cj, pj) in c.iter_mut().zip(&p) {
            *cj += dt * pj;
        }
        let wb = &b.w[bi];
        let mc = b.mv(&c);
        let mp = b.mv(&p);
        let mw = b.mv(wb);
        f1 += dt * f2 + gb * gb + b.theta * gb * zb - gb * dot(wb, &mc);
        f2 -= b.theta * gb * gb + 2.0 * gb * dot(wb, &mp) + gb * gb * dot(wb, &mw);
        // Guard against loss of positive curvature from rounding (as lbfgsb.f).
        f2 = f2.max(f64::EPSILON * f2_org);
        for (pj, wj) in p.iter_mut().zip(wb) {
            *pj += gb * wj;
        }
        d[bi] = 0.0;
        dt_min = -f1 / f2;
        t_old = t;
        next += 1;
    }
    let dt_min = dt_min.max(0.0);
    let t_final = if dt_min.is_finite() {
        t_old + dt_min
    } else {
        t_old
    };
    for i in 0..n {
        if d[i] != 0.0 {
            xc[i] = x[i] + t_final * d[i];
        }
    }
    let dt_min = if dt_min.is_finite() { dt_min } else { 0.0 };
    for (cj, pj) in c.iter_mut().zip(&p) {
        *cj += dt_min * pj;
    }
    (xc, c)
}

/// Subspace minimisation over the variables free at `x_c` (direct primal
/// method), followed by the Morales-Nocedal projection. Returns `x̄`.
fn subspace_min(
    x: &[f64],
    g: &[f64],
    bounds: &[Bounds],
    b: &Compact,
    xc: &[f64],
    c: &[f64],
) -> Vec<f64> {
    let n = x.len();
    let k2 = b.m.len();
    let free: Vec<usize> = (0..n)
        .filter(|&i| xc[i] > lo(&bounds[i]) && xc[i] < hi(&bounds[i]))
        .collect();
    if free.is_empty() {
        return xc.to_vec();
    }
    // Reduced gradient of the model at x_c: Zᵀ(g + θ(x_c - x) - W M c).
    let mc = b.mv(c);
    let r: Vec<f64> = free
        .iter()
        .map(|&i| g[i] + b.theta * (xc[i] - x[i]) - dot(&b.w[i], &mc))
        .collect();
    // du = -(1/θ) r - (1/θ²) ZᵀW (I - (1/θ) M WᵀZZᵀW)⁻¹ M WᵀZ r.
    let mut du: Vec<f64> = r.iter().map(|v| -v / b.theta).collect();
    if k2 > 0 {
        let mut wtzr = vec![0.0; k2];
        let mut wtzzw = vec![vec![0.0; k2]; k2];
        for (fi, &i) in free.iter().enumerate() {
            let wi = &b.w[i];
            for a in 0..k2 {
                wtzr[a] += wi[a] * r[fi];
                for bb in 0..k2 {
                    wtzzw[a][bb] += wi[a] * wi[bb];
                }
            }
        }
        let mut nmat = vec![vec![0.0; k2]; k2];
        for a in 0..k2 {
            for bb in 0..k2 {
                let mw: f64 = (0..k2).map(|q| b.m[a][q] * wtzzw[q][bb]).sum();
                nmat[a][bb] = if a == bb { 1.0 } else { 0.0 } - mw / b.theta;
            }
        }
        let v = b.mv(&wtzr);
        if let Some(sol) = solve_dense(nmat, v) {
            for (fi, &i) in free.iter().enumerate() {
                du[fi] -= dot(&b.w[i], &sol) / (b.theta * b.theta);
            }
        }
    }

    // Morales-Nocedal: project the subspace minimiser onto the box ...
    let mut xbar = xc.to_vec();
    for (fi, &i) in free.iter().enumerate() {
        xbar[i] = (xc[i] + du[fi]).max(lo(&bounds[i])).min(hi(&bounds[i]));
    }
    let descent: f64 = (0..n).map(|i| (xbar[i] - x[i]) * g[i]).sum();
    if descent < 0.0 {
        return xbar;
    }
    // ... else truncate the step to stay feasible (original BLNZ).
    let mut alpha = 1.0f64;
    for (fi, &i) in free.iter().enumerate() {
        let step = du[fi];
        if step < 0.0 {
            alpha = alpha.min((lo(&bounds[i]) - xc[i]) / step);
        } else if step > 0.0 {
            alpha = alpha.min((hi(&bounds[i]) - xc[i]) / step);
        }
    }
    let alpha = alpha.max(0.0);
    let mut xbar = xc.to_vec();
    for (fi, &i) in free.iter().enumerate() {
        xbar[i] = xc[i] + alpha * du[fi];
    }
    xbar
}

/// Backtracking Armijo search on `x + α d`, `α ∈ (0, 1]`, with safeguarded
/// quadratic interpolation. Every trial point is feasible because `x` and
/// `x + d` are. Returns `(α, f)` of an accepted step that decreases `f`.
pub(crate) fn line_search(
    f: &impl Fn(&[f64]) -> f64,
    x: &[f64],
    fx: f64,
    d: &[f64],
    slope: f64,
    alpha0: f64,
    max_steps: usize,
) -> Option<(f64, f64, Vec<f64>)> {
    const C1: f64 = 1e-4;
    let mut alpha = alpha0;
    let mut trial = vec![0.0; x.len()];
    for _ in 0..max_steps.max(1) {
        for i in 0..x.len() {
            trial[i] = x[i] + alpha * d[i];
        }
        let ft = f(&trial);
        if ft.is_finite() && ft <= fx + C1 * alpha * slope && ft < fx {
            return Some((alpha, ft, trial));
        }
        // Minimiser of the quadratic through f(0), f'(0), f(α), kept in
        // [0.1α, 0.5α]; a non-finite value just halves.
        let next = if ft.is_finite() {
            let denom = 2.0 * (ft - fx - slope * alpha);
            if denom > 0.0 {
                -slope * alpha * alpha / denom
            } else {
                0.5 * alpha
            }
        } else {
            0.5 * alpha
        };
        alpha = next.max(0.1 * alpha).min(0.5 * alpha);
    }
    None
}

/// Minimize a scalar function using L-BFGS-B (limited-memory BFGS with box constraints).
///
/// # Arguments
/// * `f` - Objective function.
/// * `grad` - Gradient function.
/// * `x0` - Initial guess (projected onto the box first).
/// * `bounds` - Per-variable box constraints.
/// * `m` - Number of correction pairs to store (memory size).
/// * `config` - `max_iters`; `gtol` is the projected-gradient infinity-norm
///   tolerance (SciPy's `pgtol`); `line_search_max` caps function evaluations
///   per line search.
///
/// `converged` is `true` when the projected gradient or the relative reduction
/// of `f` falls below tolerance (SciPy's two success criteria); it is `false`
/// on the iteration cap or when no decreasing step can be found. `fun` is
/// always `f(x)` and is never above `f` at the projected start.
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
    let m = m.max(1);

    let mut x = x0.to_vec();
    project(&mut x, bounds);
    let mut fx = f(&x);
    let mut g = grad(&x);

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);

    let mut converged = false;
    let mut iters = 0usize;

    while iters < config.max_iters {
        if projected_gradient_norm(&x, &g, bounds) <= config.gtol {
            converged = true;
            break;
        }

        let mut step = None;
        // First with the stored memory; if that yields no decrease, once more
        // from scratch (B = I), as lbfgsb.f does after a line-search failure.
        for attempt in 0..2 {
            if attempt == 1 {
                if s_hist.is_empty() {
                    break;
                }
                s_hist.clear();
                y_hist.clear();
            }
            let Some(bmat) = Compact::new(n, &s_hist, &y_hist) else {
                continue;
            };
            let (xc, c) = cauchy_point(&x, &g, bounds, &bmat);
            let xbar = subspace_min(&x, &g, bounds, &bmat, &xc, &c);
            let d: Vec<f64> = xbar.iter().zip(&x).map(|(a, b)| a - b).collect();
            let slope = dot(&g, &d);
            if !(slope < 0.0) {
                continue;
            }
            // First iteration without curvature information: start with a
            // unit-length step as lbfgsb.f does.
            let dnorm = dot(&d, &d).sqrt();
            let alpha0 = if s_hist.is_empty() {
                (1.0 / dnorm).min(1.0)
            } else {
                1.0
            };
            if let Some(found) = line_search(&f, &x, fx, &d, slope, alpha0, config.line_search_max)
            {
                step = Some(found);
                break;
            }
        }
        let Some((_, f_new, x_new)) = step else {
            // No decreasing step even from steepest descent: stop here.
            break;
        };
        iters += 1;

        let g_new = grad(&x_new);
        let s_vec: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y_vec: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s_vec, &y_vec);
        // Skip the update unless the curvature condition holds robustly.
        if sy > f64::EPSILON * dot(&y_vec, &y_vec) {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
        }

        let f_old = fx;
        x = x_new;
        g = g_new;
        fx = f_new;
        if (f_old - fx) / f_old.abs().max(fx.abs()).max(1.0) <= FTOL {
            converged = true;
            break;
        }
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
