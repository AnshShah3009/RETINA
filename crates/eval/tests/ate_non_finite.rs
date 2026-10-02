//! `Trajectory::ate` must distinguish a malformed trajectory from a bad result.
//!
//! `ate` returns a struct of plain `f64` with no error channel. A trajectory
//! holding one non-finite coordinate produced `rmse = NaN` alongside every other
//! statistic, so a malformed input and a genuinely poor registration were
//! indistinguishable at the call site - and a NaN comparison is always false, so
//! any `if ate.rmse > threshold` quietly takes the other branch.
//!
//! Not reachable from a dataset file: the readers reject `inf`/`nan`/`1e400` at
//! parse time. It is reachable from a `Trajectory` assembled in memory, which is
//! what these tests do.

use cv_core::Pose;
use cv_eval::trajectory::{Alignment, Trajectory};
use nalgebra::{Matrix3, Vector3};

fn pose(x: f64, y: f64) -> Pose {
    Pose::new(Matrix3::identity(), Vector3::new(x, y, 0.0))
}

fn trajectory(n: usize, offset: f64) -> Trajectory {
    Trajectory::from_poses(
        &(0..n)
            .map(|i| pose(i as f64 * 1.5 + offset, i as f64 * 0.5))
            .collect::<Vec<_>>(),
    )
}

#[test]
fn a_finite_trajectory_reports_a_valid_result() {
    let est = trajectory(12, 0.0);
    let gt = trajectory(12, 0.1);
    let r = est.ate(&gt, Alignment::Se3);
    assert!(r.is_valid, "a finite trajectory must report a valid result");
    assert!(r.rmse.is_finite(), "rmse={}", r.rmse);
    assert_eq!(r.errors.len(), 12, "per-pose errors must be populated");
}

#[test]
fn a_non_finite_coordinate_is_reported_as_invalid() {
    let mut poses: Vec<Pose> = (0..12)
        .map(|i| pose(i as f64 * 1.5, i as f64 * 0.5))
        .collect();
    poses[4] = pose(f64::NAN, 2.0);
    let est = Trajectory::from_poses(&poses);
    let gt = trajectory(12, 0.1);

    let r = est.ate(&gt, Alignment::Se3);
    assert!(
        !r.is_valid,
        "a NaN coordinate must be reported, not returned as rmse = NaN alongside \
         a populated error list that looks like a real measurement"
    );
    assert!(
        r.errors.is_empty(),
        "an invalid result must not carry per-pose errors: {:?}",
        r.errors
    );
}

#[test]
fn a_non_finite_coordinate_in_the_ground_truth_is_reported() {
    let est = trajectory(12, 0.0);
    let mut gt_poses: Vec<Pose> = (0..12)
        .map(|i| pose(i as f64 * 1.5, i as f64 * 0.5))
        .collect();
    gt_poses[9] = pose(1.0, f64::INFINITY);
    let gt = Trajectory::from_poses(&gt_poses);

    assert!(
        !est.ate(&gt, Alignment::Se3).is_valid,
        "the ground truth must be validated too, not only the estimate"
    );
}

#[test]
fn an_empty_trajectory_is_valid_and_reports_zero_error() {
    let est = Trajectory::from_poses(&[]);
    let gt = Trajectory::from_poses(&[]);
    let r = est.ate(&gt, Alignment::Se3);
    assert!(
        r.is_valid,
        "an empty trajectory is a valid degenerate input"
    );
    assert_eq!(r.rmse, 0.0);
}
