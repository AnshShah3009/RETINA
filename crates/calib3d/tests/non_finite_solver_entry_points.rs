//! Public solvers that the earlier non-finite sweep (`non_finite_input.rs`,
//! `dlt_non_finite.rs`) did not reach.
//!
//! Measured before the guards, each call on a worker thread with a 10 s
//! deadline:
//!
//! | call | result |
//! | --- | --- |
//! | `Triangulator::triangulate_linear`, one NaN coordinate | no return |
//! | `Triangulator::triangulate_points`, NaN pose | no return |
//! | `PnpSolver::estimate_epnp`, one NaN pixel | no return |
//! | `stereo_rectify_matrices`, NaN baseline | `Ok`, every matrix NaN |
//! | `refine_camera_calibration_iterative`, `fx = NaN` | `Ok(fx = NaN)` |
//!
//! The hangs are nalgebra's SVD, whose convergence test is a comparison and
//! never succeeds on NaN.

use cv_calib3d::{
    CameraCalibrationResult, PnpSolver, Triangulator, generate_chessboard_object_points,
    refine_camera_calibration_iterative, stereo_rectify_matrices,
};
use cv_core::{CameraIntrinsics, Distortion, PinholeModel, Pose};
use nalgebra::{Matrix3, Matrix3x4, Point2, Vector3};
use std::sync::mpsc;
use std::time::Duration;

/// `Some(is_err)` if `f` returns within 10 s, `None` if it hangs.
fn returns_err<F: FnOnce() -> bool + Send + 'static>(f: F) -> Option<bool> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(10)).ok()
}

fn k() -> CameraIntrinsics {
    CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480)
}

#[test]
fn triangulate_linear_refuses_nan() {
    let p1 = Matrix3x4::identity();
    let mut p2 = Matrix3x4::identity();
    p2[(0, 3)] = -1.0;
    assert!(Triangulator::triangulate_linear(&p1, &p2, &[0.0, 0.0], &[-0.2, 0.0]).is_ok());
    let r = returns_err(move || {
        Triangulator::triangulate_linear(&p1, &p2, &[f64::NAN, 0.0], &[-0.2, 0.0]).is_err()
    });
    assert_eq!(r, Some(true), "None = no return within 10 s");
}

#[test]
fn triangulate_points_refuses_a_nan_pose() {
    let r = returns_err(|| {
        let p = Pose::new(Matrix3::identity(), Vector3::new(f64::NAN, 0.0, 0.0));
        Triangulator::triangulate_points(&Pose::identity(), &p, &[[0.0, 0.0]], &[[0.1, 0.0]])
            .iter()
            .all(|x| x.is_err())
    });
    assert_eq!(r, Some(true), "None = no return within 10 s");
}

#[test]
fn epnp_refuses_a_nan_pixel() {
    let r = returns_err(|| {
        let obj: Vec<Vector3<f64>> = (0..8)
            .map(|i| {
                let s = i as f64;
                Vector3::new(0.1 * s, 0.02 * s * s, 3.0 + s.sin())
            })
            .collect();
        let mut img: Vec<[f64; 2]> = (0..8)
            .map(|i| [100.0 + i as f64, 200.0 - 3.0 * i as f64])
            .collect();
        img[2][0] = f64::NAN;
        let m = PinholeModel::new(k(), Distortion::none());
        PnpSolver::estimate_epnp(&obj, &img, &m).is_err()
    });
    assert_eq!(r, Some(true), "None = no return within 10 s");
}

#[test]
fn stereo_rectify_refuses_a_nan_baseline() {
    let right = Pose::new(Matrix3::identity(), Vector3::new(f64::NAN, 0.0, 0.0));
    let r = stereo_rectify_matrices(&k(), &k(), &Pose::identity(), &right);
    assert!(
        r.is_err(),
        "returned Ok with P2 = {:?}",
        r.map(|m| m.p2).unwrap()
    );
}

#[test]
fn calibration_refinement_refuses_a_nan_start() {
    let obj = generate_chessboard_object_points((4, 3), 0.1);
    let img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| Point2::new(300.0 + 500.0 * p.x, 200.0 + 500.0 * p.y))
        .collect();
    let init = CameraCalibrationResult {
        intrinsics: CameraIntrinsics::new(f64::NAN, 500.0, 320.0, 240.0, 640, 480),
        extrinsics: vec![Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, 1.0))],
        distortion: Distortion::none(),
        rms_reprojection_error: 0.0,
    };
    let r = refine_camera_calibration_iterative(&init, &[obj], &[img], 10);
    assert!(
        r.is_err(),
        "returned Ok with fx = {:?}",
        r.map(|c| c.intrinsics.fx).unwrap()
    );
}
