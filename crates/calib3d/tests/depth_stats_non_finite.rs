//! `compute_depth_stats` must not report a NaN as a measurement.
//!
//! A depth of `NaN` or infinity is not a reading - it is a failed or absent
//! measurement - and this workspace has found five separate defects of the shape
//! "a plausible value returned for input that supports no conclusion".
//!
//! The one here is quiet in a way worth naming: `f64::min` and `f64::max` return
//! **the non-NaN operand**, so NaN does not propagate through a fold the way one
//! expects. Given `[Some(1.0), Some(NaN), Some(inf)]` the old code returned
//!
//! ```text
//! Some((1.0, inf, NaN))
//! ```
//!
//! where `min` is a clean `1.0` and only the mean is destroyed. Every component
//! looks plausible in isolation, which is what makes this hard to spot by reading
//! and easy to miss in a pipeline.

#![forbid(unsafe_code)]

use cv_calib3d::stereo_matching::depth::compute_depth_stats;

/// The decisive case: a NaN among finite depths must not reach the mean.
#[test]
fn a_nan_depth_does_not_reach_the_statistics() {
    let got = compute_depth_stats(&[Some(1.0), Some(f64::NAN), Some(3.0)])
        .expect("two finite depths remain, so statistics must be reported");

    assert!(
        got.0.is_finite() && got.1.is_finite() && got.2.is_finite(),
        "no component may be non-finite; got min={}, max={}, mean={}",
        got.0,
        got.1,
        got.2
    );
    assert_eq!(
        got,
        (1.0, 3.0, 2.0),
        "the two finite depths are 1.0 and 3.0, so min=1, max=3, mean=2"
    );
}

/// An infinite depth is equally not a measurement.
#[test]
fn an_infinite_depth_is_excluded_too() {
    let got =
        compute_depth_stats(&[Some(1.0), Some(f64::INFINITY)]).expect("one finite depth remains");

    assert_eq!(
        got,
        (1.0, 1.0, 1.0),
        "a single finite depth of 1.0 gives min = max = mean = 1.0"
    );

    // Negative infinity too — it would otherwise become the reported minimum.
    let got = compute_depth_stats(&[Some(2.0), Some(f64::NEG_INFINITY)]).expect("one remains");
    assert_eq!(got, (2.0, 2.0, 2.0), "-inf must not become the minimum");
}

/// When *nothing* is finite, the answer is `None` - not a triple of sentinels.
#[test]
fn an_all_non_finite_input_reports_nothing_rather_than_sentinels() {
    assert!(
        compute_depth_stats(&[Some(f64::NAN), Some(f64::INFINITY)]).is_none(),
        "no finite depth means no statistics; returning `(inf, -inf, NaN)` would \\
         be three well-formed numbers describing nothing"
    );
    assert!(
        compute_depth_stats(&[None, None]).is_none(),
        "all-`None` must also report nothing"
    );
}

/// CONTROL: ordinary input is unaffected, and the empty case still reports nothing.
#[test]
fn ordinary_input_is_unchanged() {
    assert_eq!(
        compute_depth_stats(&[Some(1.0), Some(2.0), Some(3.0)]),
        Some((1.0, 3.0, 2.0))
    );
    // `None` entries have always been skipped; that must still hold.
    assert_eq!(
        compute_depth_stats(&[Some(1.0), None, Some(3.0)]),
        Some((1.0, 3.0, 2.0)),
        "`None` is a missing measurement and is skipped, as before"
    );
    assert!(
        compute_depth_stats(&[]).is_none(),
        "no input, no statistics"
    );
    assert_eq!(
        compute_depth_stats(&[Some(7.0)]),
        Some((7.0, 7.0, 7.0)),
        "a single depth gives min = max = mean = that depth"
    );
}
