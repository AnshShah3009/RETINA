//! Regression tests for `curve_fit` on a model with **no parameters**.
//!
//! The existing guards cover "lengths disagree" (`m != n`) and "`m < np`".
//! A third shape is not covered and it is not a rare one: `p0.is_empty()` is the
//! natural way to write "this model has no free parameters", and it reached the
//! covariance solve, where nalgebra's Cholesky indexes `0..dim - 1` on a
//! zero-by-zero matrix.
//!
//! Measured against baseline `4029e46`, sweeping the empty shapes:
//!
//! ```text
//!   m=0 np=0  PANIC   attempt to subtract with overflow
//!   m=0 np=1  Err     "Need at least as many data points as parameters"
//!   m=0 np=2  Err     "Need at least as many data points as parameters"
//!   m=1 np=0  PANIC   attempt to subtract with overflow
//!   m=2 np=0  PANIC   attempt to subtract with overflow
//! ```
//!
//! (nalgebra-0.33.2 `linalg/solve.rs:125`). The panic is on `np == 0`, **not**
//! on empty data — empty data with a real model was already handled. Note that
//! `m=1` and `m=2` panic just as hard as the fully empty case, so a guard
//! written for `m == 0` would have been both wrong and dead code.

use cv_optimize::general::curve_fit;

/// `np == 0` must be reported, not unwound on. A model with no free parameters
/// has a well-defined result: the residuals are whatever the model produces on
/// the empty parameter vector, and the fit cannot change them.
#[test]
fn curve_fit_a_model_with_no_parameters_returns_instead_of_panicking() {
    let model = |_x: f64, _p: &[f64]| 42.0;

    // The fully empty shape.
    let res = curve_fit(model, &[], &[], &[], 10)
        .expect("a model with no parameters is a reportable shape, not a panic");
    assert!(res.params.is_empty());
    assert!(res.residuals.is_empty());
    assert!(res.covariance.is_empty());

    // With data: the residual is `y - model(x)` and the fit cannot change it.
    let x_data: Vec<f64> = (0..3).map(|i| i as f64).collect();
    let y_data: Vec<f64> = vec![40.0, 41.0, 44.0];
    let res = curve_fit(model, &x_data, &y_data, &[], 10).expect("no parameters, three points");
    assert!(res.params.is_empty());
    assert_eq!(
        res.residuals,
        vec![-2.0, -1.0, 2.0],
        "with no parameters the residual is y - model(x)"
    );
    // r_squared must describe *these* residuals, not a stale or default value.
    let cost: f64 = res.residuals.iter().map(|v| v * v).sum();
    let y_mean = y_data.iter().sum::<f64>() / 3.0;
    let ss_tot: f64 = y_data.iter().map(|&y| (y - y_mean).powi(2)).sum();
    assert!(
        (res.r_squared - (1.0 - cost / ss_tot)).abs() < 1e-12,
        "r_squared = {} but the residuals give {}",
        res.r_squared,
        1.0 - cost / ss_tot
    );
    assert!(res.r_squared.is_finite(), "r_squared must not be NaN");
}

/// Control: the panic must not have been "fixed" by rejecting everything, and
/// the *adjacent* guard must be untouched — `m < np` is still a clean `Err`,
/// not a panic and not a silent success. This is the test that would catch a
/// guard widened to swallow `m == 0 np > 0` as well.
#[test]
fn curve_fit_control_still_rejects_fewer_points_than_parameters() {
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    match curve_fit(model, &[1.0], &[2.0], &[0.0, 0.0], 50) {
        Ok(r) => panic!("m == np produced a fit: {:?}", r.params),
        Err(e) => assert!(
            e.contains("at least as many data points"),
            "unexpected error: {e}"
        ),
    }
    match curve_fit(model, &[1.0, 2.0], &[2.0, 4.0], &[0.0, 0.0, 0.0], 50) {
        Ok(r) => panic!("m < np produced a fit: {:?}", r.params),
        Err(e) => assert!(
            e.contains("at least as many data points"),
            "unexpected error: {e}"
        ),
    }
}

/// Control: a real fit with one parameter must still recover it, so the guard
/// above cannot be satisfied by refusing to fit anything.
#[test]
fn curve_fit_control_still_fits_a_real_model() {
    let x_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| 3.0 * x + 1.0).collect();
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let res = curve_fit(model, &x_data, &y_data, &[0.0, 0.0], 50).expect("a two-point fit");
    assert!(
        (res.params[0] - 3.0).abs() < 1e-6,
        "slope = {}, expected 3",
        res.params[0]
    );
    assert!(
        (res.params[1] - 1.0).abs() < 1e-6,
        "intercept = {}, expected 1",
        res.params[1]
    );
    assert!(res.covariance.len() == 2 && res.covariance[0].len() == 2);
}
