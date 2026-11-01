//! Regression test for two independent `Interp1d::call` defects.
//!
//! 1. A one-knot series whose knot is NaN panicked. `Interp1d::new` accepts any
//!    single knot, NaN included. Every clamp comparison against a NaN knot is
//!    false, so the query fell through to the interval search, whose
//!    `Ok(pos) => ... .min(self.x.len() - 2)` is `1usize - 2` when `len == 1`:
//!    "attempt to subtract with overflow", and in release a wrap to
//!    `usize::MAX` followed by an out-of-bounds `self.x[idx + 1]`.
//!
//! 2. A descending knot sequence was searched as if it were ascending, so the
//!    bracketing interval was wrong and the blend used the wrong pair of knots.
//!    Its sibling `interpolate::interp1d_linear` refuses a non-increasing
//!    sequence; `call` returns a bare `f64` and cannot, so it has to read the
//!    caller's ordering instead of assuming one.

use cv_math::Interp1d;

#[test]
fn interp1d_nan_knot_and_descending_knots() {
    // CONTROL 1: a one-knot series with a NaN knot and a finite query. This
    // used to panic inside `call` before any value was produced.
    let nan_knot = Interp1d::new(vec![f64::NAN], vec![1.0]).unwrap();
    let v = nan_knot.call(0.5);
    assert_eq!(
        v, 1.0,
        "Interp1d::call(0.5) on a NaN knot = {v}, want the lone value 1.0 \
         (it must not panic on `self.x.len() - 2`)"
    );
    // The NaN query path is unchanged.
    assert!(nan_knot.call(f64::NAN).is_nan());

    // CONTROL 1b: the same shape with a finite knot still returns the lone
    // value, so the fix is scoped to the knot being unusable.
    let one = Interp1d::new(vec![0.0], vec![5.0]).unwrap();
    assert_eq!(one.call(0.5), 5.0);

    // CONTROL 2: the exact descending case from the report. x runs 1.0 -> 0.0
    // while y runs 0.0 -> 10.0, so 0.5 is halfway between the knots, and the
    // old ascending search instead bracketed the whole series as [x[0], x[1]]
    // and returned the far end, 10.
    let desc = Interp1d::new(vec![1.0, 0.0], vec![0.0, 10.0]).unwrap();
    let mid = desc.call(0.5);
    assert!(
        (mid - 5.0).abs() < 1e-12,
        "Interp1d::call(0.5) on descending knots = {mid}, want 5"
    );

    // Clamping follows the caller's own orientation: the first knot still caps
    // the high end and the last caps the low end.
    assert!((desc.call(2.0) - 0.0).abs() < 1e-12);
    assert!((desc.call(-1.0) - 10.0).abs() < 1e-12);

    // Longer descending series: pick the correct interior interval.
    let desc3 = Interp1d::new(vec![3.0, 2.0, 1.0, 0.0], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    assert!(
        (desc3.call(1.5) - 15.0).abs() < 1e-12,
        "got {}",
        desc3.call(1.5)
    );
    assert!(
        (desc3.call(2.5) - 5.0).abs() < 1e-12,
        "got {}",
        desc3.call(2.5)
    );
    assert!(
        (desc3.call(1.0) - 20.0).abs() < 1e-12,
        "got {}",
        desc3.call(1.0)
    );

    // Ascending series keep interpolating exactly as before.
    let asc = Interp1d::new(vec![0.0, 1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    assert!((asc.call(1.5) - 25.0).abs() < 1e-12);
    assert!((asc.call(-5.0) - 10.0).abs() < 1e-12);
    assert!((asc.call(9.0) - 40.0).abs() < 1e-12);
}
