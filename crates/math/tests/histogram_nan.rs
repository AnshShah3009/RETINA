//! `histogram` against `numpy.histogram`.
//!
//! NaN used to be counted into bin 0 (`NaN as usize` is 0), and a range below
//! an absolute `1e-30` was treated as constant data.

use cv_math::stats::histogram;

#[test]
fn nan_is_skipped() {
    // numpy.histogram([0.5, nan, 1.5, 2.5], bins=2, range=(0.5, 2.5)) -> [1, 2]
    let (edges, counts) = histogram(&[0.5, f64::NAN, 1.5, 2.5], 2);
    assert_eq!(edges, vec![0.5, 1.5, 2.5]);
    assert_eq!(counts, vec![1, 2]);
    let (edges, counts) = histogram(&[f64::NAN, f64::NAN], 3);
    assert!(edges.is_empty() && counts.is_empty());
}

#[test]
fn tiny_ranges_are_ranges() {
    // numpy: counts [1, 2], edges [1e-40, 2e-40, 3e-40]
    let (edges, counts) = histogram(&[1e-40, 2e-40, 3e-40], 2);
    assert_eq!(counts, vec![1, 2]);
    assert!((edges[1] - 2e-40).abs() < 1e-55);
}

#[test]
fn constant_data_is_centred_like_numpy() {
    // numpy.histogram([7, 7], bins=2) -> counts [0, 2], edges [6.5, 7, 7.5]
    let (edges, counts) = histogram(&[7.0, 7.0], 2);
    assert_eq!(edges, vec![6.5, 7.0, 7.5]);
    assert_eq!(counts, vec![0, 2]);
}
