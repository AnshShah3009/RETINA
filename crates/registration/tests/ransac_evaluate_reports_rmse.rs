//! The RANSAC/FGR `evaluate_registration` must report a real RMSE, and must
//! make absence visible.
//!
//! Two defects, both in `registration/global/ransac.rs`:
//!
//! 1. `dist` there is the KDTree's **squared** neighbour distance. The function
//!    did `total_error += dist` and then `(total_error / n).sqrt()`, which is
//!    `sqrt(mean(d^4))` - the root-mean-square of the *squares*, not a
//!    root-mean-square. It is systematically optimistic, because the fourth
//!    power is dominated by the largest inlier and the many small ones barely
//!    register. The correct sibling at `registration/mod.rs` accumulates
//!    `dist * dist` because its `dist` is already Euclidean.
//!
//! 2. An empty source returned `(0.0, 0.0)` and `inlier_count == 0` returned
//!    `rmse 0.0` - perfect scores for a registration never attempted, and a
//!    claim of *zero* inliers and *zero* error at once, which is not a coherent
//!    answer for any transform.
//!
//! `evaluate_registration` is private, so it is driven through the public
//! `registration_fgr_based_on_feature_matching`. The transform it evaluates
//! would normally be the solver's output, which fits the misalignment away and
//! leaves a residual of ~0 - useless for pinning a *value*. Setting
//! `maximum_tuple_count: 1` makes the graduated solver's correspondence set
//! drop below its own 3-point minimum, so it keeps the identity it started from
//! and the residual reaching the evaluation is exactly the displacement the
//! test built in. That is stated here rather than left implicit, because the
//! whole test depends on it.

use cv_core::point_cloud::PointCloud;
use cv_registration::registration::{
    registration_fgr_based_on_feature_matching, FPFHFeature, FastGlobalRegistrationOption,
};
use nalgebra::Point3;

fn feature(v: f32) -> FPFHFeature {
    FPFHFeature { histogram: [v; 33] }
}

fn cloud(points: Vec<(f32, f32, f32)>) -> PointCloud {
    PointCloud {
        points: points
            .into_iter()
            .map(|p| Point3::new(p.0, p.1, p.2))
            .collect(),
        colors: None,
        normals: None,
    }
}

/// Distinct per-index histograms, so the FPFH matching is exact and the Lowe
/// ratio test keeps every match. Constant or near-constant histograms are
/// rejected earlier by the solver and would prove nothing.
fn features(n: usize) -> Vec<FPFHFeature> {
    (0..n).map(|i| feature(i as f32)).collect()
}

/// Drive the evaluation with the identity transform - see the module comment for
/// why - and return `(fitness, inlier_rmse)`.
///
/// `max_dist` is the *geometric* gate the evaluation itself applies.
fn evaluated(
    source: Vec<(f32, f32, f32)>,
    target: Vec<(f32, f32, f32)>,
    max_dist: f32,
) -> (f32, f32) {
    let n = source.len();
    let r = registration_fgr_based_on_feature_matching(
        &cloud(source),
        &cloud(target),
        &features(n),
        &features(n),
        FastGlobalRegistrationOption {
            maximum_correspondence_distance: max_dist as f64,
            iteration_number: 64,
            maximum_tuple_count: 1,
            tuple_scale: 0.95,
        },
    )
    .expect("FGR over exactly-matching features must succeed");
    assert_eq!(
        r.transformation,
        nalgebra::Matrix4::<f32>::identity(),
        "this test only means something with the identity transform reaching the \
         evaluation; the solver moved the pose: {:?}",
        r.transformation
    );
    (r.fitness, r.inlier_rmse)
}

/// `n` source points, each displaced from its matched target point by `dy`.
fn uniform(n: usize, dy: f32) -> (Vec<(f32, f32, f32)>, Vec<(f32, f32, f32)>) {
    let target: Vec<(f32, f32, f32)> = (0..n).map(|i| (i as f32, 0.0, 0.0)).collect();
    let source: Vec<(f32, f32, f32)> = target.iter().map(|p| (p.0, p.1 + dy, p.2)).collect();
    (source, target)
}

/// CONTROL: a well-formed registration with a uniform offset must still report
/// that offset, and a perfect fit must still report zero.
///
/// Uniform distances are the case where a true RMSE and `sqrt(mean(d^4))`
/// coincide, so these two cannot tell the two formulas apart - which is exactly
/// why they are the *control* and the wide-spread test below is the real one.
#[test]
fn a_uniform_offset_reports_that_offset() {
    let (source, target) = uniform(8, 0.02);
    let (fitness, rmse) = evaluated(source, target, 1.0);
    assert!(
        (fitness - 1.0).abs() < 1e-6,
        "control: every source point should be an inlier, got fitness {fitness}"
    );
    assert!(
        (f64::from(rmse) - 0.02).abs() < 1e-6,
        "control: a uniform 0.02 offset must report an rmse of 0.02, got {rmse}"
    );
}

/// The perfect-fit control: already-aligned clouds must report zero, not a
/// sentinel. This pins that the "make absence visible" change did not turn
/// into "never report a small rmse".
#[test]
fn a_perfect_fit_still_reports_zero() {
    let (source, target) = uniform(8, 0.0);
    let (fitness, rmse) = evaluated(source, target, 0.075);
    assert!(
        (fitness - 1.0).abs() < 1e-6,
        "control: an already-aligned pair must be all inliers, got {fitness}"
    );
    assert_eq!(
        rmse, 0.0,
        "control: a perfect fit must report 0, not a sentinel"
    );
}

/// A WIDE SPREAD of inlier distances is the case that separates the two
/// formulas. `sqrt(mean(d^4))` is dominated by the largest inlier and lands
/// far below the true RMSE.
#[test]
fn a_wide_spread_of_inlier_distances_reports_a_true_rmse() {
    let n = 30usize;
    // Geometric in distance, 1e-3 up to ~7.1e-3, so the true RMSE is well above
    // the smallest inlier and any sqrt(mean(d^4)) is far below it.
    let distances: Vec<f32> = (0..n).map(|i| 1e-3 * 1.07f32.powi(i as i32)).collect();

    let target: Vec<(f32, f32, f32)> = (0..n).map(|i| (i as f32, 0.0, 0.0)).collect();
    let source: Vec<(f32, f32, f32)> = target
        .iter()
        .zip(distances.iter())
        .map(|(p, d)| (p.0, p.1 + d, p.2))
        .collect();

    let true_rmse = (distances.iter().map(|d| f64::from(*d * *d)).sum::<f64>() / n as f64).sqrt();
    let naive = (distances
        .iter()
        .map(|d| f64::from(*d * *d * *d * *d))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    let (fitness, rmse) = evaluated(source, target, 1.0);

    assert!(
        (fitness - 1.0).abs() < 1e-6,
        "every source point is within the gate, so fitness must be 1.0, got {fitness}"
    );
    assert!(
        (f64::from(rmse) - true_rmse).abs() <= 1e-3 * true_rmse,
        "reported inlier_rmse {rmse} is not the RMSE of these inlier distances \
         (true {true_rmse:.6e}; the buggy sqrt(mean(d^4)) would say {naive:.6e})"
    );
}

/// Zero inliers means the error was never measured, and `0.0` reads as a
/// flawless fit. The old code reported `inlier_rmse 0.0` beside `fitness 0.0`
/// - claiming no inliers *and* no error simultaneously.
#[test]
fn no_inliers_is_not_zero_error() {
    // 5 units apart with a 0.075 gate: nothing is an inlier.
    let (source, target) = uniform(8, 5.0);
    let (fitness, rmse) = evaluated(source, target, 0.075);
    assert_eq!(
        fitness, 0.0,
        "nothing is within the gate, so there are no inliers"
    );
    assert!(
        rmse > 1e30,
        "with no inlier the error was never measured: reporting {rmse} is a \
         perfect score for a registration that found nothing"
    );
}

/// An empty source has nothing to register, and must not score as a perfect
/// registration. The old code returned `(0.0, 0.0)` here.
///
/// An empty source means no FPFH features, so FGR's own "insufficient
/// correspondences" guard rejects it before the evaluation is reached. That
/// guard is upstream of the defect and is left as it is; what is asserted here
/// is the contract it is *supposed* to uphold. This test therefore does not
/// fail against the unfixed code - it is a regression guard for the
/// `(0.0, 0.0)` return, not a reproduction. The reproductions of the defect are
/// `no_inliers_is_not_zero_error` (the reachable `rmse 0.0` branch) and
/// `a_wide_spread_of_inlier_distances_reports_a_true_rmse` (the formula).
#[test]
fn an_empty_source_is_not_a_perfect_registration() {
    let r = registration_fgr_based_on_feature_matching(
        &cloud(Vec::new()),
        &cloud(uniform(8, 0.0).1),
        &features(0),
        &features(8),
        FastGlobalRegistrationOption {
            maximum_correspondence_distance: 0.075,
            iteration_number: 64,
            maximum_tuple_count: 1,
            tuple_scale: 0.95,
        },
    );
    match r {
        Err(_) => {}
        Ok(v) => panic!(
            "an empty source has nothing to register, but it returned Ok with \
             fitness {} and inlier_rmse {}",
            v.fitness, v.inlier_rmse
        ),
    }
}
