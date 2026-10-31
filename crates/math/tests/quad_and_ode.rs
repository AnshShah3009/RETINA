//! `quad` against `scipy.integrate.quad`, and `ode_solve` landing on `t_end`.
//!
//! Before the fix `quad` was a fixed 100-panel trapezoid rule that reported an
//! error of exactly 0 and evaluated the endpoints, so `1/sqrt(x)` on `[0, 1]`
//! returned `inf` (SciPy: 2). `ode_solve` stepped by `dt` past `t_end`
//! (`0..1`, `dt = 0.3` ended at `t = 1.2`) and silently stopped after 10 000
//! steps.

use cv_math::integrate::{ode_solve, quad, quad_with};

/// (name, f, a, b, SciPy value). SciPy's own error estimates are all below
/// 2.4e-8, its default tolerance being `epsabs = epsrel = 1.49e-8`.
fn cases() -> Vec<(&'static str, Box<dyn Fn(f64) -> f64>, f64, f64, f64)> {
    vec![
        (
            "1/sqrt(x)",
            Box::new(|x: f64| 1.0 / x.sqrt()),
            0.0,
            1.0,
            1.9999999999999984,
        ),
        (
            "ln x",
            Box::new(|x: f64| x.ln()),
            0.0,
            1.0,
            -0.9999999999999999,
        ),
        (
            "x^2",
            Box::new(|x: f64| x * x),
            0.0,
            1.0,
            0.33333333333333337,
        ),
        ("sin", Box::new(f64::sin), 0.0, std::f64::consts::PI, 2.0),
        (
            "exp(-x^2)",
            Box::new(|x: f64| (-x * x).exp()),
            -5.0,
            5.0,
            1.7724538509027912,
        ),
        // exact: sin(50) / 50
        (
            "cos(50x)",
            Box::new(|x: f64| (50.0 * x).cos()),
            0.0,
            1.0,
            -0.005247497074078575,
        ),
        // exact: 100 (atan(70) + atan(30))
        (
            "peak",
            Box::new(|x: f64| 1.0 / (1e-4 + (x - 0.3) * (x - 0.3))),
            0.0,
            1.0,
            309.3986915124149,
        ),
    ]
}

#[test]
fn quad_matches_scipy_with_an_honest_error_estimate() {
    for (name, f, a, b, want) in cases() {
        let (v, err) = quad(&f, a, b).unwrap_or_else(|e| panic!("{name}: {e}"));
        let actual = (v - want).abs();
        assert!(
            actual <= 1.49e-8 * want.abs().max(1.0),
            "{name}: {v} vs SciPy {want}"
        );
        // The estimate must bound the actual error and must not be zero
        // unless the result is exact.
        assert!(
            err >= actual * 0.5 || actual < 1e-14,
            "{name}: err {err:e}, actual {actual:e}"
        );
        assert!(err.is_finite() && err >= 0.0);
        assert!(
            err <= 1.49e-8 * v.abs().max(1.0),
            "{name}: err {err:e} above tolerance"
        );
    }
}

#[test]
fn quad_never_evaluates_the_endpoints() {
    let f = |x: f64| {
        assert!(x > 0.0 && x < 1.0, "endpoint {x} evaluated");
        1.0 / (x * (1.0 - x)).sqrt()
    };
    // integral = pi
    let (v, _) = quad(f, 0.0, 1.0).expect("integrable endpoint singularities");
    assert!((v - std::f64::consts::PI).abs() < 1e-7, "{v}");
}

#[test]
fn quad_reports_failure_instead_of_a_number() {
    // 1/x is not integrable on [0, 1]: SciPy warns and returns a junk value.
    assert!(quad(|x| 1.0 / x, 0.0, 1.0).is_err());
    // Too small a subdivision limit for a hard integrand.
    let peak = |x: f64| 1.0 / (1e-6 + (x - 0.3) * (x - 0.3));
    assert!(quad_with(peak, 0.0, 1.0, 1e-12, 1e-12, 3).is_err());
    // NaN from the integrand is an error, not a result.
    assert!(quad(|x| if x > 0.5 { f64::NAN } else { 1.0 }, 0.0, 1.0).is_err());
}

#[test]
fn quad_degenerate_and_reversed_intervals() {
    assert_eq!(quad(|_| 7.0, 1.0, 1.0).unwrap(), (0.0, 0.0));
    let (v, _) = quad(|x| x * x, 1.0, 0.0).unwrap();
    assert!((v + 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn ode_solve_lands_exactly_on_t_end() {
    // y' = -y, y(0) = 1, on [0, 1] with dt = 0.3: steps 0.3, 0.6, 0.9, 1.0.
    let sol = ode_solve(|_, y| vec![-y[0]], &[1.0], (0.0, 1.0), 0.3).unwrap();
    let ts: Vec<f64> = sol.iter().map(|(t, _)| *t).collect();
    assert_eq!(ts.len(), 5, "{ts:?}");
    assert_eq!(*ts.last().unwrap(), 1.0, "{ts:?}");
    assert!((ts[3] - 0.9).abs() < 1e-15);
    let y1 = sol.last().unwrap().1[0];
    assert!((y1 - (-1.0f64).exp()).abs() < 1e-4, "y(1) = {y1}");
}

#[test]
fn ode_solve_errors_instead_of_truncating() {
    // 2e7 steps exceeds the cap: an error, not a silently shortened solution.
    assert!(ode_solve(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 5e-8).is_err());
    assert!(ode_solve(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 0.0).is_err());
    assert!(ode_solve(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), f64::NAN).is_err());
    // 20 000 steps used to stop at 10 000 (t = 0.5); it now completes.
    let sol = ode_solve(|_, y| vec![y[0]], &[1.0], (0.0, 1.0), 5e-5).unwrap();
    assert_eq!(sol.last().unwrap().0, 1.0);
    assert!((sol.last().unwrap().1[0] - std::f64::consts::E).abs() < 1e-10);
    // An empty span is just the initial state.
    let sol = ode_solve(|_, y| vec![y[0]], &[1.0], (2.0, 2.0), 0.1).unwrap();
    assert_eq!(sol.len(), 1);
}
