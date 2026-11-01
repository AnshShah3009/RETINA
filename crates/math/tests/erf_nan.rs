//! `erf` must propagate NaN.
//!
//! `erf` selects its branch with `ax <= 2.0` and `ax < ERF_SATURATES_AT`.
//! Both comparisons are false for a NaN `ax`, so control fell through to the
//! saturated tail and returned `sign` (`1.0`) - a plausible answer that no
//! caller can recognise as wrong, and the opposite of what the function's own
//! documentation promises ("NaN in gives NaN out"). `erfc` already guards this
//! with `if !x.is_finite()`; `erf` did not.
//!
//! The controls pin the finite and infinite behaviour that was already
//! correct, so the fix cannot be "return NaN more often".

use cv_math::special::erf;

#[test]
fn erf_of_nan_is_nan() {
    let v = erf(f64::NAN);
    assert!(
        v.is_nan(),
        "erf(NaN) = {v}, want NaN (the saturated-tail branch turned it into 1.0)"
    );

    // CONTROL: exact at the origin, and the two saturated tails stay finite.
    assert_eq!(erf(0.0), 0.0, "erf(0) must stay exactly 0");
    assert_eq!(erf(f64::INFINITY), 1.0, "erf(+inf) must stay 1");
    assert_eq!(erf(f64::NEG_INFINITY), -1.0, "erf(-inf) must stay -1");

    // CONTROL: ordinary finite arguments are unchanged.
    assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-15);
    assert!((erf(-1.0) + 0.8427007929497149).abs() < 1e-15);
    assert!((erf(3.0) - 0.9999779095030014).abs() < 1e-15);

    // CONTROL: `erfc` already propagated NaN; this pins that it stays so.
    assert!(cv_math::special::erfc(f64::NAN).is_nan());
}
