//! `ErrorStats::from_errors` must not mix `NaN` and plausible numbers.
//!
//! The four fields that go through a sum propagate a non-finite value on their
//! own: `mean`, `rmse` and `std` all came back `NaN` for `[1.0, NaN, 3.0]`.
//! `max` did not, because it is a `fold` with `f64::max`, and `f64::max` returns
//! **the non-NaN operand** - `f64::max(3.0, NaN) == 3.0` - so the fold walked
//! straight past the NaN and produced `max = 3.0`.
//!
//! Measured, before the fix:
//!
//! ```text
//! [1.0, NaN, 3.0] -> mean = NaN   rmse = NaN   std = NaN   max = 3.0
//! [1.0, inf, 3.0] -> mean = inf   rmse = inf   std = NaN   max = 3.0
//! ```
//!
//! A single confident `max` in a struct whose other four fields say
//! "unmeasurable" is exactly the "plausible value for input that supports no
//! conclusion" failure: a caller that reports worst-case error, thresholds on
//! it, or sorts by it reads `3.0` and concludes the worst pose was off by
//! three metres when the truth is that the series was never measured.
//!
//! This is the asymmetry to remember, and it is worth stating plainly because
//! the two halves sound alike and only one is true:
//!
//! * `"min/max clamp NaN"` — **TRUE**. `f64::max` returns the non-NaN operand,
//!   so it never yields NaN and never panics; it silently drops the bad value.
//! * `"it panics on NaN"` — **FALSE**. That is `partial_cmp(..).unwrap()`, which
//!   is a different function and lives elsewhere in this workspace.

use cv_eval::trajectory::ErrorStats;

#[test]
fn a_non_finite_sample_poisons_every_statistic_including_max() {
    for (name, series) in [
        ("NaN", [1.0, f64::NAN, 3.0]),
        ("+inf", [1.0, f64::INFINITY, 3.0]),
        ("-inf", [1.0, f64::NEG_INFINITY, 3.0]),
    ] {
        let s = ErrorStats::from_errors(&series);
        assert!(s.rmse.is_nan(), "{name}: rmse = {}", s.rmse);
        assert!(s.mean.is_nan(), "{name}: mean = {}", s.mean);
        assert!(s.median.is_nan(), "{name}: median = {}", s.median);
        assert!(s.std.is_nan(), "{name}: std = {}", s.std);
        assert!(
            s.max.is_nan(),
            "{name}: max = {} - a series holding a value with no defined \
             ordering cannot report a worst case",
            s.max
        );
    }

    // The case that was wrong: `max` reported the largest *finite* sample.
    let s = ErrorStats::from_errors(&[1.0, f64::NAN, 3.0]);
    assert!(
        !s.max.is_finite(),
        "max was {} before the fix, where the correct answer is NaN",
        s.max
    );
}

#[test]
fn a_single_non_finite_sample_is_enough() {
    // It does not have to be in the middle, and a long series does not dilute
    // it: one bad sample out of a thousand still makes the whole summary
    // unmeasurable, because there is no way to say which pose it came from.
    let mut series: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    series[997] = f64::NAN;
    let s = ErrorStats::from_errors(&series);
    assert!(s.max.is_nan(), "max = {}", s.max);
    assert!(s.rmse.is_nan());
    assert!(s.mean.is_nan());
    assert!(s.std.is_nan());
}

#[test]
fn a_finite_series_is_summarised_exactly() {
    // Control for the two tests above: ordinary data must be untouched, and
    // every statistic is the one you would compute by hand.
    let s = ErrorStats::from_errors(&[1.0, 2.0, 3.0, 4.0]);
    assert!((s.mean - 2.5).abs() < 1e-12, "mean = {}", s.mean);
    assert!((s.median - 2.5).abs() < 1e-12, "median = {}", s.median);
    assert_eq!(s.max, 4.0);
    // rmse = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5)
    assert!((s.rmse - 7.5_f64.sqrt()).abs() < 1e-12, "rmse = {}", s.rmse);
    // population standard deviation: sqrt(((1.5)^2 x2 + (0.5)^2 x2) / 4)
    assert!((s.std - 1.25_f64.sqrt()).abs() < 1e-12, "std = {}", s.std);
    // Control: all-identical samples - a zero spread computed from the data,
    // not a fabricated zero.
    let same = ErrorStats::from_errors(&[2.5; 8]);
    assert_eq!(same.mean, 2.5);
    assert_eq!(same.median, 2.5);
    assert_eq!(same.max, 2.5);
    assert_eq!(same.std, 0.0);
    assert!((same.rmse - 2.5).abs() < 1e-12);

    // Control: monotonic in the error magnitude, the check that catches a
    // metric averaging over the wrong denominator.
    let mut previous = f64::NEG_INFINITY;
    for k in [0.0_f64, 0.1, 1.0, 10.0, 1000.0] {
        let series: Vec<f64> = (0..16).map(|i| (i as f64 - 8.0).abs() * k).collect();
        let s = ErrorStats::from_errors(&series);
        assert!(
            s.rmse > previous,
            "rmse must grow with the error magnitude: {} then {}",
            previous,
            s.rmse
        );
        previous = s.rmse;
    }
}

#[test]
fn an_empty_series_is_still_nan_rather_than_zero() {
    // The pre-existing contract, kept as a control so the new guard cannot be
    // satisfied by an early `return` that happens to zero the fields.
    let s = ErrorStats::from_errors(&[]);
    assert!(s.rmse.is_nan() && s.mean.is_nan() && s.median.is_nan());
    assert!(s.max.is_nan() && s.std.is_nan());
}
