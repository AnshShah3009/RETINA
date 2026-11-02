//! `calibrate_robot_world_hand_eye` must refuse non-finite input.
//!
//! `calibrate_hand_eye` checks for it because `symmetric_eigen` on a matrix
//! holding NaN does not return (recorded there: killed at 60 s). The
//! robot-world variant reaches the same eigendecomposition through the same
//! `solve_ax_xb` with no check: one NaN rotation entry and the call never
//! returned. The signature is already `Option`, so `None` is the answer.

use cv_calib3d::{RobotWorldHandEyeMethod, calibrate_robot_world_hand_eye};
use nalgebra::{Matrix3, Rotation3, Vector3};
use std::sync::mpsc;
use std::time::Duration;

fn views() -> (
    Vec<Matrix3<f64>>,
    Vec<Vector3<f64>>,
    Vec<Matrix3<f64>>,
    Vec<Vector3<f64>>,
) {
    let x = (
        Rotation3::from_euler_angles(0.2, -0.1, 0.3).into_inner(),
        Vector3::new(0.3, 0.0, -0.1),
    );
    let z = (
        Rotation3::from_euler_angles(-0.3, 0.2, 0.1).into_inner(),
        Vector3::new(-0.02, 0.04, 0.06),
    );
    let (mut ra, mut ta, mut rb, mut tb) = (vec![], vec![], vec![], vec![]);
    for i in 0..6 {
        let s = i as f64;
        let b = (
            Rotation3::from_euler_angles(0.4 * s.sin(), 0.5 * (1.3 * s).cos(), 0.7 * s)
                .into_inner(),
            Vector3::new(0.1 * s, -0.2, 0.3 + 0.05 * s),
        );
        // A_i = Z · B_i · X⁻¹
        let xi_r = x.0.transpose();
        let xi_t = -(xi_r * x.1);
        let m_r = z.0 * b.0;
        let m_t = z.1 + z.0 * b.1;
        ra.push(m_r * xi_r);
        ta.push(m_t + m_r * xi_t);
        rb.push(b.0);
        tb.push(b.1);
    }
    (ra, ta, rb, tb)
}

/// Run on a worker thread so a hang fails the test instead of the suite.
fn returns_within(
    ra: Vec<Matrix3<f64>>,
    ta: Vec<Vector3<f64>>,
    rb: Vec<Matrix3<f64>>,
    tb: Vec<Vector3<f64>>,
) -> Option<bool> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let r = calibrate_robot_world_hand_eye(&ra, &ta, &rb, &tb, RobotWorldHandEyeMethod::Shah);
        let _ = tx.send(r.is_some());
    });
    rx.recv_timeout(Duration::from_secs(20)).ok()
}

#[test]
fn finite_views_still_calibrate() {
    let (ra, ta, rb, tb) = views();
    assert_eq!(returns_within(ra, ta, rb, tb), Some(true));
}

#[test]
fn a_nan_rotation_entry_is_refused() {
    let (mut ra, ta, rb, tb) = views();
    ra[2][(1, 0)] = f64::NAN;
    assert_eq!(
        returns_within(ra, ta, rb, tb),
        Some(false),
        "None = the call did not return within 20 s"
    );
}

#[test]
fn an_infinite_translation_is_refused() {
    let (ra, ta, rb, mut tb) = views();
    tb[4].y = f64::INFINITY;
    assert_eq!(
        returns_within(ra, ta, rb, tb),
        Some(false),
        "None = the call did not return within 20 s"
    );
}
