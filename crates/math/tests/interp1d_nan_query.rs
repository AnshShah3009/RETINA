//! Regression test for defect 4: `Interp1d::call` panicked with an unsigned
//! underflow on a NaN query.
//!
//! `binary_search_by` returned `Err(0)` for a NaN query, and
//! `unwrap_or_else(|pos| pos - 1)` computed `0usize - 1`, which wraps to
//! `usize::MAX`; the following `self.x[idx + 1]` then went out of bounds.
//! The sibling `interpolate.rs::find_interval` already had the NaN guard —
//! this was a second, independent implementation that was missed.

use cv_math::Interp1d;

#[test]
fn interp1d_nan_query_does_not_panic() {
    let ip = Interp1d::new(vec![0.0, 1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

    // CONTROL: the exact case that used to panic
    //   "attempt to subtract with overflow" at src/lib.rs:113, and
    //   before the overflow-check was even reached,
    //   "index out of bounds: the len is 4 but the index is 18446744073709551615".
    let v = ip.call(f64::NAN);
    assert!(
        v.is_nan(),
        "Interp1d::call(NaN) = {v}, want NaN (it must not panic)"
    );

    // CONTROL: ordinary queries still interpolate.
    assert!((ip.call(1.5) - 25.0).abs() < 1e-12);
    assert!((ip.call(0.0) - 10.0).abs() < 1e-12);
    assert!((ip.call(3.0) - 40.0).abs() < 1e-12);
    // Clamping at both ends is unchanged.
    assert!((ip.call(-5.0) - 10.0).abs() < 1e-12);
    assert!((ip.call(9.0) - 40.0).abs() < 1e-12);
    // Interior node hit exactly.
    assert!((ip.call(2.0) - 30.0).abs() < 1e-12);
    // Values between every adjacent pair.
    assert!((ip.call(0.25) - 12.5).abs() < 1e-12);
    assert!((ip.call(1.75) - 27.5).abs() < 1e-12);
    assert!((ip.call(2.5) - 35.0).abs() < 1e-12);
}

#[test]
fn interp1d_two_point_and_single_point() {
    // CONTROL: degenerate sizes must not panic either.
    let two = Interp1d::new(vec![0.0, 1.0], vec![5.0, 7.0]).unwrap();
    assert!((two.call(0.5) - 6.0).abs() < 1e-12);
    assert!(two.call(f64::NAN).is_nan());

    let one = Interp1d::new(vec![0.0], vec![5.0]).unwrap();
    assert!((one.call(0.5) - 5.0).abs() < 1e-12);
    assert!(one.call(f64::NAN).is_nan());
}
