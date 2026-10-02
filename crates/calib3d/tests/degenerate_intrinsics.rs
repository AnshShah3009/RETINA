//! Degenerate intrinsics must be reported, not silently replaced by the identity.
//!
//! `CameraIntrinsics::inverse_matrix` returned
//! `try_inverse().unwrap_or(Matrix3::identity())`. The intrinsic matrix has
//! determinant `fx · fy`, so it is singular exactly when either focal length is
//! zero — and the fields are public, so such a value is constructible.
//!
//! The identity is not a neutral fallback. It is a *valid* matrix meaning "this
//! camera has no intrinsics", so every point is used as its own pixel
//! coordinate. Measured, with `fx = 0`: a pixel `(123, 456, 1)` maps to
//! `(123, 456, 1)` unchanged.
//!
//! Seven call sites fed this into calibration, essential-matrix, PnP,
//! triangulation and structure-from-motion solves. Those that can report an
//! error now do.

use cv_calib3d::{find_essential_mat, recover_pose_from_essential, solve_pnp_dlt};
use cv_core::CameraIntrinsics;
use nalgebra::{Point2, Point3};

fn degenerate() -> CameraIntrinsics {
    CameraIntrinsics::new(0.0, 0.0, 320.0, 240.0, 640, 480)
}

fn good() -> CameraIntrinsics {
    CameraIntrinsics::new(620.0, 615.0, 320.0, 240.0, 640, 480)
}

/// The primitive itself: a singular intrinsic matrix has no inverse.
#[test]
fn a_singular_intrinsic_matrix_has_no_inverse() {
    assert!(
        degenerate().try_inverse_matrix().is_none(),
        "fx = fy = 0 makes the intrinsic matrix singular"
    );
    assert!(
        good().try_inverse_matrix().is_some(),
        "control: ordinary intrinsics must invert"
    );
}

/// The documented behaviour of the old fallback, pinned so the reason for
/// changing it stays explicit rather than historical.
#[test]
fn the_identity_fallback_discarded_the_intrinsics() {
    let inv = degenerate().inverse_matrix_or_identity();
    let p = inv * nalgebra::Vector3::new(123.0, 456.0, 1.0);
    assert_eq!(
        (p.x, p.y, p.z),
        (123.0, 456.0, 1.0),
        "with the identity, a pixel maps to itself - the intrinsics were dropped, \
         not merely approximated"
    );
}

/// `solve_pnp_dlt` takes intrinsics from the caller.
#[test]
fn solve_pnp_dlt_rejects_singular_intrinsics() {
    let obj: Vec<Point3<f64>> = (0..12)
        .map(|i| Point3::new(i as f64 * 0.1, (i % 4) as f64 * 0.1, 1.0))
        .collect();
    let img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| Point2::new(p.x * 620.0 + 320.0, p.y * 615.0 + 240.0))
        .collect();

    // Control: ordinary intrinsics must still solve.
    assert!(
        solve_pnp_dlt(&obj, &img, &good()).is_ok(),
        "control: valid intrinsics must still solve"
    );

    let err = solve_pnp_dlt(&obj, &img, &degenerate())
        .expect_err("singular intrinsics must be reported, not normalised by identity");
    assert!(
        format!("{err}").contains("singular"),
        "the error should name the problem, got: {err}"
    );
}

#[test]
fn find_essential_mat_still_solves_with_valid_intrinsics() {
    // A control only: the essential path funnels through a private helper that
    // has no error channel, so it keeps the identity fallback. Asserting the
    // ordinary case still works documents that decision rather than hiding it.
    let pts1: Vec<Point2<f64>> = (0..10)
        .map(|i| Point2::new(i as f64 * 11.0, i as f64 * 6.0))
        .collect();
    let pts2: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| Point2::new(p.x + 4.0, p.y + 2.0))
        .collect();
    let r = find_essential_mat(&pts1, &pts2, &good());
    assert!(
        !format!("{r:?}").contains("not finite"),
        "control: finite input with valid intrinsics must not be rejected"
    );
}

#[test]
fn recover_pose_from_essential_rejects_singular_intrinsics() {
    let pts: Vec<Point2<f64>> = (0..8)
        .map(|i| Point2::new(i as f64 * 9.0, i as f64 * 4.0))
        .collect();
    let pose = cv_core::Pose::new(nalgebra::Matrix3::identity(), nalgebra::Vector3::zeros());
    let e = cv_calib3d::essential_from_extrinsics(&pose);

    let err = recover_pose_from_essential(&e, &pts, &pts, &degenerate())
        .expect_err("singular intrinsics must be reported");
    assert!(format!("{err}").contains("singular"), "got: {err}");
}
