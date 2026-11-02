//! With as many data points as parameters there are no residual degrees of
//! freedom, so the noise variance - and with it the parameter covariance - is
//! not estimable. `scipy.optimize.curve_fit` fills `pcov` with `inf` and warns
//! "Covariance of the parameters could not be estimated":
//!
//! ```python
//! >>> curve_fit(lambda x, a, b: a * x + b, [1.0, 2.0], [3.0, 5.0], p0=[0, 0])
//! (array([2., 1.]), array([[inf, inf], [inf, inf]]))
//! ```
//!
//! `curve_fit` used `dof = 1` for that case and returned
//! `(JᵀJ)⁻¹ · RSS`, where the RSS of an exact interpolation is round-off: a
//! covariance of ~1e-30, i.e. "every parameter known exactly", for a fit that
//! carries no information at all about its own uncertainty.

use cv_optimize::general::curve_fit;

#[test]
fn zero_residual_dof_reports_infinite_covariance_like_scipy() {
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let res = curve_fit(model, &[1.0, 2.0], &[3.0, 5.0], &[0.0, 0.0], 100).expect("fits");
    assert!((res.params[0] - 2.0).abs() < 1e-8, "{:?}", res.params);
    assert!((res.params[1] - 1.0).abs() < 1e-8, "{:?}", res.params);
    for row in &res.covariance {
        for &c in row {
            assert!(
                c == f64::INFINITY,
                "covariance with zero residual dof must be inf (SciPy), got {:?}",
                res.covariance
            );
        }
    }
}

/// Three points, three parameters: the quadratic through them is exact, and
/// the covariance is again not estimable.
#[test]
fn exactly_determined_quadratic_reports_infinite_covariance() {
    let model = |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2];
    let res = curve_fit(
        model,
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0, 7.0],
        &[1.0, 1.0, 1.0],
        100,
    )
    .expect("fits");
    assert_eq!(res.covariance.len(), 3);
    assert!(
        res.covariance.iter().flatten().all(|&c| c == f64::INFINITY),
        "got {:?}",
        res.covariance
    );
}

/// Control: one spare point gives one degree of freedom, and the covariance
/// is the finite closed form `(JᵀJ)⁻¹ · RSS / (m - np)`.
#[test]
fn one_residual_dof_keeps_a_finite_covariance() {
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let res = curve_fit(model, &[0.0, 1.0, 2.0], &[0.0, 1.0, 3.0], &[0.0, 0.0], 100).expect("fits");
    assert!(
        res.covariance.iter().flatten().all(|c| c.is_finite()),
        "{:?}",
        res.covariance
    );
    // scipy: curve_fit(lambda x,a,b: a*x+b, [0,1,2], [0,1,3]) ->
    // pcov = [[0.0833333, -0.0833333], [-0.0833333, 0.1388889]], the closed form
    // (JᵀJ)⁻¹ · RSS / 1 = [[3, -3], [-3, 5]] / 6 · (1/6).
    let want = [[3.0 / 36.0, -3.0 / 36.0], [-3.0 / 36.0, 5.0 / 36.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (res.covariance[i][j] - want[i][j]).abs() < 1e-6,
                "{:?} vs {want:?}",
                res.covariance
            );
        }
    }
}
