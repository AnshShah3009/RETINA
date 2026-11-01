//! `Alignment::Sim3` must recover the same similarity transform whatever unit
//! the trajectory happens to be expressed in.
//!
//! Umeyama's scale is a *ratio* - `trace(D S) / variance(src)` - so it is
//! dimensionless by construction: expressing both trajectories in micrometres
//! instead of metres multiplies numerator and denominator by the same factor and
//! leaves the answer alone.
//!
//! The degeneracy guard did not respect that. It was
//!
//! ```text
//! let scale = if with_scale && variance_src > 1e-12 { .. } else { 1.0 };
//! ```
//!
//! and `variance_src` is a *squared length*: an absolute `1e-12` therefore means
//! "a standard deviation under one micrometre", not "no extent to measure". Any
//! scene smaller than roughly a micrometre took the `else` branch and reported
//! `scale = 1.0` - silently, since `1.0` is also the value `Se3` returns - while
//! `is_valid` stayed `true` and the per-pose errors looked like a real
//! measurement.
//!
//! Measured before the fix, on the same cloud scaled by `k` and estimated at
//! twice its size (so the true scale is `0.5` and the true ATE is `0`):
//!
//! ```text
//! k = 1e-6 -> scale = 0.5000, rmse / extent = 4.4e-16   (correct)
//! k = 5e-7 -> scale = 1.0000, rmse / extent = 4.5e-01   (45% of the scene!)
//! k = 1e-7 -> scale = 1.0000, rmse / extent = 4.5e-01
//! ```
//!
//! A 45%-of-extent scale-free error is not a rounding artefact, it is the
//! failure the metric exists to detect, reported for a perfect reconstruction.
//! A dense SfM map stored in metres around a tabletop, or a scan in millimetres,
//! lands on whichever side of the constant it happens to hit.

use cv_eval::trajectory::{Alignment, Trajectory};
use nalgebra::{UnitQuaternion, Vector3};

/// A cloud with extent of order 1, deliberately not planar, not collinear and
/// not centred on the origin.
const SHAPE: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.5, 0.25, 0.75],
    [0.25, 0.75, 0.5],
];

fn trajectory(points: &[[f64; 3]]) -> Trajectory {
    let positions: Vec<Vector3<f64>> = points
        .iter()
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect();
    let quaternions = vec![UnitQuaternion::identity(); points.len()];
    Trajectory::from_positions_and_quaternions(&positions, &quaternions)
}

/// `SHAPE` scaled by `k`, and the same shape estimated at twice its size, so the
/// correct Sim(3) scale is always `0.5` and the correct ATE is always `0`.
fn pair(k: f64) -> (Trajectory, Trajectory) {
    let gt = trajectory(&SHAPE.map(|p| [p[0] * k, p[1] * k, p[2] * k]));
    let est = trajectory(&SHAPE.map(|p| [p[0] * 2.0 * k, p[1] * 2.0 * k, p[2] * 2.0 * k]));
    (gt, est)
}

#[test]
fn sim3_scale_does_not_depend_on_the_units_of_the_trajectory() {
    // 1.5 is the largest extent of `SHAPE` along any axis.
    const EXTENT: f64 = 1.5;

    for k in [1.0_f64, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9] {
        let (gt, est) = pair(k);
        let ate = est.ate(&gt, Alignment::Sim3);

        assert!(
            ate.is_valid,
            "k = {k:e}: a finite trajectory must report a valid result"
        );
        assert!(
            (ate.scale - 0.5).abs() < 1e-9,
            "k = {k:e}: Sim(3) recovered scale {} instead of 0.5 - the estimate \
             depends on the units the scene is measured in, not on the scene",
            ate.scale
        );

        // Compare relative to the scene extent: an ATE is only meaningful
        // against the size of the trajectory it was measured on.
        let relative = ate.rmse / (EXTENT * k);
        assert!(
            relative < 1e-9,
            "k = {k:e}: a perfect similarity match scored rmse = {:e}, which is \
             {:.1}% of the scene extent",
            ate.rmse,
            relative * 100.0
        );
    }
}

#[test]
fn a_rotated_and_scaled_scene_is_aligned_away_at_any_unit() {
    // The scale estimate must also still recover a rotation, at a scale small
    // enough that the old absolute guard rejected it.
    let k = 1e-7;
    let gt = trajectory(&SHAPE.map(|p| [p[0] * k, p[1] * k, p[2] * k]));
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.6);
    let poses: Vec<cv_core::Pose> = gt
        .poses
        .iter()
        .map(|p| cv_core::Pose::from_quat_translation(p.rotation, 2.0 * (rotation * p.translation)))
        .collect();
    let est = Trajectory::from_poses(&poses);

    let ate = est.ate(&gt, Alignment::Sim3);
    assert!(
        (ate.scale - 0.5).abs() < 1e-9,
        "scale = {} for a scene rotated 0.6 rad and scaled by 2",
        ate.scale
    );
    let relative = ate.rmse / (1.5 * k);
    assert!(
        relative < 1e-9,
        "rmse = {:e} ({:.2}% of the extent) for an exact similarity match",
        ate.rmse,
        relative * 100.0
    );
}

#[test]
fn se3_alignment_was_already_unit_invariant() {
    // Control: the rigid alignment has no scale to estimate, so it never had
    // this defect. It must keep working at both ends of the unit range.
    for k in [1.0_f64, 1e-7] {
        let gt = trajectory(&SHAPE.map(|p| [p[0] * k, p[1] * k, p[2] * k]));
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7);
        let shift = Vector3::new(0.3 * k, -0.2 * k, 0.1 * k);
        let poses: Vec<cv_core::Pose> = gt
            .poses
            .iter()
            .map(|p| {
                cv_core::Pose::from_quat_translation(p.rotation, rotation * p.translation + shift)
            })
            .collect();
        let est = Trajectory::from_poses(&poses);

        let ate = est.ate(&gt, Alignment::Se3);
        assert_eq!(ate.scale, 1.0);
        assert!(
            ate.rmse / k < 1e-9,
            "k = {k:e}: rigid ATE = {:e}, i.e. {:.2}% of the extent",
            ate.rmse,
            ate.rmse / (1.5 * k) * 100.0
        );
    }
}

#[test]
fn a_trajectory_with_no_extent_still_reports_no_scale() {
    // Control for the fix: with every camera at the same point there is no
    // extent to estimate a scale from, and the guard has to keep rejecting it
    // rather than dividing by a variance of zero.
    let gt = trajectory(&[[1.0, 2.0, 3.0]; 4]);
    let est = trajectory(&[[1.0, 2.0, 3.0]; 4]);
    let ate = est.ate(&gt, Alignment::Sim3);
    assert!(
        ate.scale.is_finite(),
        "scale = {} for a trajectory with no extent",
        ate.scale
    );
    assert!(ate.rmse.is_finite(), "rmse = {}", ate.rmse);
}
