//! Undistortion with strong radial *and* tangential terms, against
//! `cv2.undistortPointsIter` (100 iterations, eps 1e-14), which recovers the
//! true pixel exactly for all three.
//!
//! The tangential correction used to be the fixed-point step
//! `u += x - apply(u)`, which diverges once the map's Jacobian leaves (0, 2);
//! `remove` then returned the *distorted* point unchanged, so
//! (773.12, 580.16) "undistorted" to itself, 166.6 px from the truth.

use cv_calib3d::undistort_points;
use cv_core::{CameraIntrinsics, Distortion, DistortionF32};
use nalgebra::Point2;

#[test]
fn undistort_points_matches_cv2_with_strong_tangential_distortion() {
    let k = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 1280, 960);
    let d = Distortion::new(0.5, 0.2, 1e-3, 0.0, 0.05);
    let cases = [
        ((547.77625, 410.9571875), (520.0, 390.0)),
        ((773.115904, 580.156928), (640.0, 480.0)),
        ((1020.48, 765.86), (720.0, 540.0)),
    ];
    for ((dx, dy), (ux, uy)) in cases {
        let out = undistort_points(&[Point2::new(dx, dy)], &k, &d).expect("invertible");
        assert!(
            (out[0].x - ux).abs() < 1e-6 && (out[0].y - uy).abs() < 1e-6,
            "({dx}, {dy}) -> {:?}, cv2 gives ({ux}, {uy})",
            out[0]
        );
    }
}

#[test]
fn f32_removal_converges_where_f64_does() {
    let d = DistortionF32::new(0.5, 0.2, 1e-3, 0.0, 0.05);
    for r in [0.5f32, 0.8, 1.0] {
        let (ux, uy) = (r * 0.8, r * 0.6);
        let (dx, dy) = d.apply(ux, uy);
        let (bx, by) = d.remove_checked(dx, dy).expect("invertible");
        assert!(
            (bx - ux).abs() < 1e-5 && (by - uy).abs() < 1e-5,
            "r = {r}: ({bx}, {by})"
        );
    }
}
