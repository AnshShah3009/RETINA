//! `calibrate_camera_planar` must terminate on degenerate input.
//!
//! Found by a scratch probe while auditing untested code: observations
//! containing NaN sent into `calibrate_camera_planar` did not return. The probe
//! hung, and the workspace test suite timed out at 360 s waiting on it.
//!
//! A solver that iterates to convergence needs a stopping rule that invalid
//! input cannot defeat. Every comparison against NaN is false, so an
//! "iterate while the error is above a threshold" loop becomes infinite - and a
//! hang is worse than a wrong answer, because nothing reports it.
//!
//! The fixture reuses the chessboard pattern from `lib_tests.rs`, which is
//! already known to calibrate correctly. That matters: a control case that fails
//! would make the hang test meaningless.

use cv_calib3d::calibrate_camera_planar;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Point2, Point3, Rotation3, Vector3};
use std::time::Duration;

/// The board points, in millimetres, in the plane z = 0.
fn board_points(cols: usize, rows: usize, spacing: f64) -> Vec<Point3<f64>> {
    let mut pts = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        for c in 0..cols {
            pts.push(Point3::new(c as f64 * spacing, r as f64 * spacing, 0.0));
        }
    }
    pts
}

fn project_point(k: &CameraIntrinsics, ext: &Pose, p: &Point3<f64>) -> Point2<f64> {
    let pc = ext.rotation * p.coords + ext.translation;
    if pc[2].abs() < 1e-10 {
        return Point2::new(f64::NAN, f64::NAN);
    }
    Point2::new(k.fx * (pc[0] / pc[2]) + k.cx, k.fy * (pc[1] / pc[2]) + k.cy)
}

fn poses() -> Vec<Pose> {
    vec![
        Pose::new(
            Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
            Vector3::new(0.05, -0.03, 2.6),
        ),
        Pose::new(
            Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
            Vector3::new(-0.08, 0.02, 2.9),
        ),
        Pose::new(
            Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
            Vector3::new(0.04, 0.05, 3.1),
        ),
    ]
}

/// Build the views, optionally corrupting one observation with NaN.
fn views(k: &CameraIntrinsics, inject_nan: bool) -> (Vec<Vec<Point3<f64>>>, Vec<Vec<Point2<f64>>>) {
    let board = board_points(7, 6, 40.0);
    let mut obj = Vec::new();
    let mut img = Vec::new();
    for (i, ext) in poses().iter().enumerate() {
        obj.push(board.clone());
        let mut v: Vec<Point2<f64>> = board.iter().map(|p| project_point(k, ext, p)).collect();
        if inject_nan && i == 1 {
            v[10] = Point2::new(f64::NAN, f64::NAN);
        }
        img.push(v);
    }
    (obj, img)
}

/// The clean case must work, or the hang test below proves nothing.
#[test]
fn a_clean_planar_sequence_calibrates() {
    let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
    let (obj, img) = views(&k, false);
    let cal = calibrate_camera_planar(&obj, &img, (640, 480)).expect("a clean sequence calibrates");
    assert!(
        (cal.intrinsics.fx - k.fx).abs() < 1e-2,
        "fx was {}, expected {}",
        cal.intrinsics.fx,
        k.fx
    );
}

/// NaN observations must not hang the solver.
///
/// Run on a worker thread with a deadline, so a hang fails *this* test rather
/// than wedging the whole suite - which is what the original probe did, at a cost
/// of 360 s.
#[test]
fn nan_observations_do_not_hang_the_solver() {
    let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
    let (obj, img) = views(&k, true);
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name("calibrate-planar-nan".into())
        .spawn(move || {
            let r = calibrate_camera_planar(&obj, &img, (640, 480));
            let _ = tx.send(r.map(|c| c.rms_reprojection_error));
        })
        .expect("spawn");

    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(result) => {
            // Terminating is the requirement. Either an error, or a finite
            // result - a NaN flowing out would be the same defect in another hat.
            if let Ok(rms) = result {
                assert!(
                    rms.is_finite(),
                    "calibration returned a non-finite rms of {rms} for NaN input"
                );
            }
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!(
                "calibrate_camera_planar did not return within 30s for NaN \
                 observations: every comparison against NaN is false, so a loop \
                 that iterates while the error is large never terminates"
            );
        }
        Err(e) => panic!("the worker thread failed: {e}"),
    }
}

/// A degenerate observation count must also return rather than spin.
///
/// Three points is exactly the minimum for a homography, leaving nothing to
/// estimate the intrinsics from.
#[test]
fn too_few_observations_terminate() {
    let k = CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
    let board = board_points(7, 6, 40.0);
    let ext = poses()[0].clone();
    let obj = vec![board[..3].to_vec(); 3];
    let img = vec![
        board[..3]
            .iter()
            .map(|p| project_point(&k, &ext, p))
            .collect::<Vec<_>>();
        3
    ];

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(calibrate_camera_planar(&obj, &img, (640, 480)).is_ok());
    });
    assert!(
        rx.recv_timeout(Duration::from_secs(30)).is_ok(),
        "calibrate_camera_planar did not return for a degenerate three-point input"
    );
}
