//! `minimize_bfgs` never takes a step that increases `f`.
//!
//! Its backtracking search halved `alpha` up to 20 times and, when no trial
//! met the Armijo condition, returned the last `alpha` anyway, so the step
//! was taken uphill. SciPy's BFGS (1.17.1) rejects the step instead and stops
//! with status 2, "Desired error not necessarily achieved due to precision
//! loss", leaving `x` where it was.
//!
//! * `f = 1e9 x^2` from `x = 1`: the first step (`d = -g = -2e9`) overshoots
//!   for every `alpha >= 2^-20`, so the old search stepped to x = -1906
//!   (f = 3.6e15 from 1e9). SciPy converges in 2 iterations to f = 3e-27;
//!   the search now interpolates (as `minimize_lbfgsb`'s does) and converges
//!   too.
//! * A gradient of the wrong sign (an ascent direction): no step decreases f.
//!   SciPy returns x0 with success False; the old search moved uphill.

use cv_optimize::general::{BfgsConfig, minimize_bfgs};

#[test]
fn a_badly_scaled_quadratic_is_not_stepped_uphill() {
    let f = |x: &[f64]| 1e9 * x[0] * x[0];
    let g = |x: &[f64]| vec![2e9 * x[0]];
    // One iteration: the first step must not go uphill.
    let one = BfgsConfig {
        max_iters: 1,
        ..BfgsConfig::default()
    };
    let res = minimize_bfgs(f, g, &[1.0], &one);
    assert!(
        res.fun <= 1e9,
        "first step: f rose from 1e9 to {} at x = {:?}",
        res.fun,
        res.x
    );
    // To convergence. SciPy: success, x = 1.7e-18, f = 3.0e-27.
    let res = minimize_bfgs(f, g, &[1.0], &BfgsConfig::default());
    assert!(res.converged, "{res:?}");
    assert!(res.x[0].abs() < 1e-8, "{res:?}");
}

#[test]
fn no_decreasing_step_leaves_x_and_reports_failure() {
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let wrong_sign = |x: &[f64]| vec![-2.0 * x[0], -2.0 * x[1]];
    let res = minimize_bfgs(f, wrong_sign, &[1.0, 2.0], &BfgsConfig::default());
    // SciPy: x = [1, 2], f = 5, success False, status 2.
    assert_eq!(res.x, vec![1.0, 2.0], "{res:?}");
    assert_eq!(res.fun, 5.0);
    assert!(!res.converged);
}
