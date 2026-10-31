//! `registration_gnc` must not report a clean registration for a malformed cloud.
//!
//! Before the guard, one NaN coordinate was dropped *by accident*: its residual
//! was NaN, `exp(-NaN)` is 0, so every robust loss gave it weight 0 and it
//! quietly contributed nothing. Measured, clouds differing only in that one
//! coordinate:
//!
//! ```text
//! finite  -> fitness 1.000000, rmse 0.000000
//! one NaN -> fitness 0.983333, rmse 0.000000
//! ```
//!
//! Both `Some`, both with a correct identity transform. The answer was right, but
//! the caller was told a malformed cloud registered cleanly, and the point that
//! vanished (1 of 60) was only visible by comparing 0.983333 against 59/60. That
//! is an accident of `exp`, not a contract - a loss whose weight does not
//! saturate would instead poison the covariance.

use cv_core::PointCloud;
use nalgebra::Point3;

/// 60 points, source and target identical so the correct answer is the identity.
fn cloud(bad: bool) -> PointCloud<f32> {
    let mut pts = Vec::new();
    for i in 0..60 {
        pts.push([
            i as f32 * 0.37,
            ((i * 7) % 23) as f32 * 0.21,
            ((i * 13) % 19) as f32 * 0.11,
        ]);
    }
    if bad {
        pts[5][0] = f32::NAN;
    }
    PointCloud::new(pts.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect())
}

fn identity_correspondences() -> Vec<(usize, usize)> {
    (0..60).map(|i| (i, i)).collect()
}

fn solve(src: &PointCloud<f32>, tgt: &PointCloud<f32>) -> Option<()> {
    cv_registration::registration::registration_gnc(
        &src.points,
        &tgt.points,
        &identity_correspondences(),
        0.05,
        cv_registration::registration::RobustLossType::Welsch,
    )
    .map(|_| ())
}

#[test]
fn a_cloud_with_one_nan_reports_absence() {
    let finite = cloud(false);
    let bad = cloud(true);

    // Control: identical finite clouds must still solve, or this test would
    // pass for the wrong reason.
    assert!(
        solve(&finite, &finite).is_some(),
        "control: a well-formed identity registration must succeed"
    );

    assert!(
        solve(&bad, &finite).is_none(),
        "a NaN coordinate must be reported as absence, not silently dropped \
         while returning fitness 0.983333"
    );
}

#[test]
fn a_nan_in_the_target_cloud_reports_absence() {
    let finite = cloud(false);
    let bad = cloud(true);
    assert!(
        solve(&finite, &bad).is_none(),
        "the target cloud must be validated too, not only the source"
    );
}

#[test]
fn an_infinite_coordinate_reports_absence() {
    let mut src = cloud(false);
    let mut pts: Vec<[f32; 3]> = src.points.iter().map(|p| [p.x, p.y, p.z]).collect();
    pts[7][2] = f32::INFINITY;
    src = PointCloud::new(pts.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect());

    let finite = cloud(false);
    assert!(
        solve(&src, &finite).is_none(),
        "an infinite coordinate must be reported, not accepted"
    );
}

/// A correspondence index out of bounds must not panic.
#[test]
fn an_out_of_bounds_correspondence_does_not_panic() {
    let finite = cloud(false);
    let mut pairs = identity_correspondences();
    pairs[3] = (3, 10_000);
    let r = cv_registration::registration::registration_gnc(
        &finite.points,
        &finite.points,
        &pairs,
        0.05,
        cv_registration::registration::RobustLossType::Welsch,
    );
    assert!(
        r.is_none(),
        "a correspondence indexing past the end of the cloud must be reported"
    );
}
