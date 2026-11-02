//! `bundle_adjust` fixes the gauge by holding the first camera's pose, and
//! `remove_outliers` takes its threshold in pixels.
//!
//! Reprojection error is unchanged by any similarity transform of the whole
//! reconstruction, so a bundle adjustment that frees every camera has a
//! 7-dimensional null space (rotation, translation, scale) in its normal
//! equations. Only the Levenberg-Marquardt damping kept those directions from
//! being singular, and the solution moved along them: the reference frame the
//! caller set up with camera 0 drifted (by 2.0e-2 in translation and
//! 2.5e-3 rad on the scene below, robust path).
//!
//! `remove_outliers(threshold)` compared the threshold with each landmark's
//! *mean squared* error, so it was in px^2 (bundle_adjust's 10.0 meant
//! 3.16 px) and nothing said so: `remove_outliers(3.0)` kept only landmarks
//! under 1.73 px and pruned the 2 px one below. It is now RMS pixels.

use cv_core::{CameraIntrinsics, Pose};
use cv_sfm::bundle_adjustment::{BundleAdjustmentConfig, SfMState, bundle_adjust};
use nalgebra::{Point2, Point3, Rotation3, Vector3};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
}

/// Four cameras looking at 60 points, observations with +-0.5 px of
/// deterministic noise, and every camera but the first plus every point
/// perturbed from the truth.
fn scene() -> (SfMState, Vec<Pose>) {
    let k = intrinsics();
    let truth: Vec<Pose> = (0..4)
        .map(|i| {
            let a = i as f64 * 0.12;
            Pose::new(
                Rotation3::from_euler_angles(0.02 * i as f64, -a, 0.01).into_inner(),
                Vector3::new(-0.6 * i as f64, 0.05 * i as f64, 0.1 * i as f64),
            )
        })
        .collect();
    let points: Vec<Point3<f64>> = (0..60)
        .map(|i| {
            let t = i as f64;
            Point3::new(
                (t * 0.37).sin() * 1.5,
                (t * 0.53).cos() * 1.0,
                5.0 + (t * 0.29).sin(),
            )
        })
        .collect();

    let mut state = SfMState::new(k);
    for (i, pose) in truth.iter().enumerate() {
        let mut p = *pose;
        if i > 0 {
            p.translation += Vector3::new(0.03, -0.02, 0.04);
        }
        state.add_camera(p);
    }
    for (j, x) in points.iter().enumerate() {
        let obs: Vec<(usize, Point2<f64>)> = truth
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                let pc = pose.rotation * x.coords + pose.translation;
                let uv = k.project(&Point3::from(pc));
                let n = ((i * 61 + j * 17) % 11) as f64 / 10.0 - 0.5;
                (i, Point2::new(uv.x + n, uv.y - 0.7 * n))
            })
            .collect();
        state.add_landmark(
            Point3::from(x.coords + Vector3::new(0.02, 0.01, -0.03)),
            obs,
        );
    }
    (state, truth)
}

#[test]
fn bundle_adjust_holds_the_first_camera() {
    for config in [
        BundleAdjustmentConfig::default(),
        BundleAdjustmentConfig {
            robust_kernel: false,
            ..BundleAdjustmentConfig::default()
        },
    ] {
        let (mut state, _) = scene();
        let before = state.cameras[0];
        let e0 = state.total_reprojection_error();
        bundle_adjust(&mut state, &config);
        let e1 = state.total_reprojection_error();
        let after = state.cameras[0];
        let dt = (after.translation - before.translation).norm();
        let dr = after.rotation.angle_to(&before.rotation);
        assert!(
            // The pose round-trips through axis-angle parameters: ~1e-15.
            dt < 1e-12 && dr < 1e-12,
            "robust={}: camera 0 moved by {dt:.3e} (translation), {dr:.3e} rad",
            config.robust_kernel
        );
        assert!(e1 < e0 && e1 < 0.5, "mean squared error {e0} -> {e1} px^2");
    }
}

#[test]
fn remove_outliers_threshold_is_rms_pixels() {
    let k = intrinsics();
    let mut state = SfMState::new(k);
    state.add_camera(Pose::identity());
    let x = Point3::new(0.0, 0.0, 5.0);
    let uv = k.project(&x);
    // Landmark 0: off by 2 px. Landmark 1: off by 4 px.
    state.add_landmark(x, vec![(0, Point2::new(uv.x + 2.0, uv.y))]);
    state.add_landmark(x, vec![(0, Point2::new(uv.x + 4.0, uv.y))]);
    state.remove_outliers(3.0);
    assert!(state.landmarks[0].is_valid, "2 px is within 3 px");
    assert!(!state.landmarks[1].is_valid, "4 px is beyond 3 px");
}
