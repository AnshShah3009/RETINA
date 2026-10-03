//! Degenerate-input contract for the `cv-eval` metrics.
//!
//! Two defect classes are pinned here, both silent by construction: they return
//! a number that looks like a measurement, so nothing downstream notices.
//!
//! 1. **A metric over zero samples reported a perfect score.** `0.0` is the best
//!    possible ATE/RPE/RMSE/Chamfer value, so "compared nothing" used to win any
//!    ranking or threshold against a real result. Empty inputs now yield `NaN`
//!    (error metrics) and the same `0.0` only for *score* metrics, where it is
//!    the worst value and cannot flatter anyone.
//! 2. **Averages divided by the wrong count.** `mean_average_precision` divided
//!    by the number of hits instead of the number of relevant items, so finding
//!    one of ten relevant items at rank 1 scored a perfect 1.0, and a query that
//!    found nothing was dropped from the mean. The recall/precision metrics
//!    counted repeated ids more than once, which pushed `recall_at_k` above 1.
//!
//! Every test carries a control: the well-formed case has to keep working, so a
//! metric that simply rejected everything would not pass.

use cv_core::Pose;
use cv_eval::reconstruction::{chamfer_distance, reprojection_rmse, rmse_over_extent};
use cv_eval::retrieval::{mean_average_precision, precision_at_k, recall_at_k};
use cv_eval::trajectory::{Alignment, ErrorStats, Trajectory};
use nalgebra::{UnitQuaternion, Vector3};

fn trajectory(centres: &[[f64; 3]]) -> Trajectory {
    let positions: Vec<Vector3<f64>> = centres
        .iter()
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
    let quaternions = vec![UnitQuaternion::identity(); centres.len()];
    Trajectory::from_positions_and_quaternions(&positions, &quaternions)
}

// ── retrieval: the average was normalised by the wrong count ────────────────

#[test]
fn average_precision_is_normalised_by_the_relevant_count() {
    // One query with three relevant ids; only one of them retrieved, and at
    // rank 1 - the best possible place for it.
    let predictions = vec![vec![7usize, 8, 9, 10]];
    let ground_truth = vec![vec![7usize, 11, 12]];

    let map = mean_average_precision(&predictions, &ground_truth);
    assert!(
        (map - 1.0 / 3.0).abs() < 1e-12,
        "a query that retrieved 1 of its 3 relevant items scored {map}, \
         expected 1/3 (the precision sums divided by |relevant|)"
    );

    // Control: retrieving all three, ranked best-first, is a perfect score.
    let perfect = vec![vec![7usize, 11, 12]];
    assert!((mean_average_precision(&perfect, &ground_truth) - 1.0).abs() < 1e-12);
}

#[test]
fn a_query_with_no_hits_is_part_of_the_mean() {
    let predictions = vec![vec![7usize, 8, 9], vec![90usize, 91, 92]];
    let ground_truth = vec![vec![7usize, 8, 9], vec![1usize, 2, 3]];

    let map = mean_average_precision(&predictions, &ground_truth);
    assert!(
        (map - 0.5).abs() < 1e-12,
        "one perfect query and one that retrieved nothing scored {map}, \
         expected 0.5: the failed query must count as 0.0, not be skipped"
    );

    // Control: two perfect queries stay at 1.0.
    let perfect = vec![vec![7usize, 8, 9], vec![1usize, 2, 3]];
    assert!((mean_average_precision(&perfect, &ground_truth) - 1.0).abs() < 1e-12);
}

#[test]
fn a_repeated_id_counts_once() {
    let predictions = vec![vec![7usize, 7, 7]];
    let ground_truth = vec![vec![7usize]];

    let recall = recall_at_k(&predictions, &ground_truth, 3);
    assert!(
        (0.0..=1.0).contains(&recall),
        "recall@3 for a list of one repeated id was {recall}, which is outside [0, 1]"
    );
    assert!((recall - 1.0).abs() < 1e-12);

    let precision = precision_at_k(&predictions, &ground_truth, 3);
    assert!(
        (precision - 1.0 / 3.0).abs() < 1e-12,
        "precision@3 for [7, 7, 7] against a single relevant 7 was {precision}, \
         expected 1/3: only one of the three slots retrieved anything"
    );

    // Control: three distinct relevant ids in the top three slots are full
    // precision.
    let distinct = vec![vec![7usize, 8, 9]];
    let gt3 = vec![vec![7usize, 8, 9]];
    assert!((precision_at_k(&distinct, &gt3, 3) - 1.0).abs() < 1e-12);
    assert!((recall_at_k(&distinct, &gt3, 3) - 1.0).abs() < 1e-12);
}

#[test]
fn retrieval_metrics_stay_within_unit_range() {
    let predictions = vec![vec![1usize, 1, 2, 2, 2, 3], vec![0usize, 0, 0]];
    let ground_truth = vec![vec![1usize, 2, 3], vec![0usize, 5]];

    for k in 0..8 {
        let recall = recall_at_k(&predictions, &ground_truth, k);
        let precision = precision_at_k(&predictions, &ground_truth, k);
        assert!((0.0..=1.0).contains(&recall), "recall@{k} = {recall}");
        assert!(
            (0.0..=1.0).contains(&precision),
            "precision@{k} = {precision}"
        );
    }
    let map = mean_average_precision(&predictions, &ground_truth);
    assert!((0.0..=1.0).contains(&map), "MAP = {map}");
}

// ── trajectory: no samples is not a zero error ──────────────────────────────

#[test]
fn error_stats_of_no_samples_are_nan() {
    let stats = ErrorStats::from_errors(&[]);
    assert!(stats.rmse.is_nan(), "rmse = {}", stats.rmse);
    assert!(stats.mean.is_nan(), "mean = {}", stats.mean);
    assert!(stats.median.is_nan(), "median = {}", stats.median);
    assert!(stats.max.is_nan(), "max = {}", stats.max);
    assert!(stats.std.is_nan(), "std = {}", stats.std);

    // Control: one sample is summarised from the data, including the zero
    // standard deviation that a single sample really has.
    let one = ErrorStats::from_errors(&[3.0]);
    assert_eq!(one.rmse, 3.0);
    assert_eq!(one.mean, 3.0);
    assert_eq!(one.median, 3.0);
    assert_eq!(one.max, 3.0);
    assert_eq!(one.std, 0.0);
}

#[test]
fn rpe_with_a_zero_frame_gap_measures_nothing() {
    // Ground truth advances one metre per frame; the estimate also drifts in y.
    let gt = trajectory(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]);
    let est = trajectory(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.5, 0.0],
        [4.0, 2.0, 0.0],
        [5.0, 2.5, 0.0],
    ]);

    // A zero gap compares every pose with itself - the identity for *any* pair
    // of trajectories - so it used to emit six exact zeros and report a perfect
    // RPE for two trajectories that disagree by half a metre per frame.
    let zero = est.rpe(&gt, 0);
    assert!(
        zero.translation_errors.is_empty(),
        "a zero frame gap fabricated {} error samples: {:?}",
        zero.translation_errors.len(),
        zero.translation_errors
    );
    assert!(zero.rotation_errors.is_empty());
    assert!(zero.translation.rmse.is_nan(), "{}", zero.translation.rmse);
    assert!(zero.rotation.rmse.is_nan(), "{}", zero.rotation.rmse);

    // A gap past the end of the trajectory has no pair either.
    let beyond = est.rpe(&gt, 6);
    assert!(beyond.translation_errors.is_empty());
    assert!(beyond.translation.rmse.is_nan());

    // Control: a real gap measures the real discrepancy.
    let one = est.rpe(&gt, 1);
    assert_eq!(one.translation_errors.len(), 5);
    assert!(
        one.translation.rmse > 0.4 && one.translation.rmse.is_finite(),
        "RPE at delta 1 = {}",
        one.translation.rmse
    );
    let last = est.rpe(&gt, 5);
    assert_eq!(last.translation_errors.len(), 1);
    assert!(last.translation.rmse.is_finite());
}

#[test]
fn ate_over_no_overlap_measures_nothing() {
    let est = trajectory(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let empty = Trajectory::default();

    for align in [Alignment::None, Alignment::Se3, Alignment::Sim3] {
        let r = empty.ate(&est, align);
        assert!(
            r.rmse.is_nan(),
            "ATE over no overlapping pose was {} with {align:?}",
            r.rmse
        );
        assert!(
            r.is_valid,
            "an empty trajectory is a well-formed input, only its result is empty"
        );
        assert!(r.errors.is_empty());
        assert!(r.transform.is_none());
        assert!(r.scale.is_nan());

        let other = est.ate(&empty, align);
        assert!(other.rmse.is_nan());
    }

    // Control: one pose of overlap is one real measurement.
    let shifted = trajectory(&[[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]]);
    let r = est.ate(&shifted, Alignment::None);
    assert!((r.rmse - 0.25).abs() < 1e-12, "{}", r.rmse);
    assert!(r.scale == 1.0);
}

#[test]
fn camera_center_rmse_over_no_overlap_measures_nothing() {
    let est = trajectory(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let empty = Trajectory::default();

    assert!(
        empty.camera_center_rmse(&est).is_nan(),
        "camera-centre RMSE over no overlap was {}",
        empty.camera_center_rmse(&est)
    );
    assert!(est.camera_center_rmse(&empty).is_nan());

    // Control: a pure translation of 0.1 is measured as 0.1.
    let shifted: Vec<Pose> = est
        .poses
        .iter()
        .map(|p| {
            Pose::from_quat_translation(p.rotation, p.translation + Vector3::new(0.1, 0.0, 0.0))
        })
        .collect();
    let other = Trajectory::from_poses(&shifted);
    assert!((other.camera_center_rmse(&est) - 0.1).abs() < 1e-12);
}

#[test]
fn explicit_timestamps_stay_parallel_to_the_poses() {
    let positions: Vec<Vector3<f64>> = (0..4).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect();
    let quaternions = [UnitQuaternion::identity(); 4];

    // Too few timestamps: the shorter input wins, as `Trajectory::new` says.
    let short = Trajectory::from_positions_and_quaternions_with_timestamps(
        &positions,
        &quaternions,
        &[1.0, 2.0],
    );
    assert_eq!(
        short.poses.len(),
        short.timestamps.len(),
        "poses and timestamps must stay parallel: {} poses, {} timestamps",
        short.poses.len(),
        short.timestamps.len()
    );
    assert_eq!(short.timestamps, vec![1.0, 2.0]);

    // Control: matching lengths keep every pose and every timestamp.
    let full = Trajectory::from_positions_and_quaternions_with_timestamps(
        &positions,
        &quaternions,
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(full.poses.len(), 4);
    assert_eq!(full.timestamps, vec![1.0, 2.0, 3.0, 4.0]);

    // Extra timestamps are trimmed to the poses.
    let long = Trajectory::from_positions_and_quaternions_with_timestamps(
        &positions,
        &quaternions,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_eq!(long.poses.len(), long.timestamps.len());
    assert_eq!(long.timestamps.len(), 4);
}

// ── reconstruction: an error metric's empty value is not 0.0 ────────────────

#[test]
fn reconstruction_error_metrics_do_not_report_zero_for_no_samples() {
    // No residual is no measurement; 0.0 is a perfect reconstruction.
    assert!(reprojection_rmse(&[]).is_nan());

    // A scene with no extent has no scale to normalise by.
    assert!(rmse_over_extent(0.5, 0.0).is_nan());
    assert!(rmse_over_extent(0.5, -1.0).is_nan());
    assert!(rmse_over_extent(0.5, f64::INFINITY).is_nan());
    assert!(rmse_over_extent(0.5, f64::NAN).is_nan());

    // An empty set is not the same cloud as a perfect match.
    let cloud = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
    assert!(chamfer_distance(&cloud, &[]).is_nan());
    assert!(chamfer_distance(&[], &cloud).is_nan());
    assert!(chamfer_distance(&[], &[]).is_nan());

    // Controls: the well-formed cases are unchanged.
    assert!((reprojection_rmse(&[3.0, 4.0]) - 12.5_f64.sqrt()).abs() < 1e-12);
    assert!((reprojection_rmse(&[0.0, 0.0]) - 0.0).abs() < 1e-12);
    assert!((rmse_over_extent(0.5, 2.0) - 0.25).abs() < 1e-12);
    assert_eq!(chamfer_distance(&cloud, &cloud), 0.0);
}
