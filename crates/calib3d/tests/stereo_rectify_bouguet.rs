//! `stereo_rectify_matrices` against `cv2.stereoRectify` (CALIB_ZERO_DISPARITY,
//! alpha = -1; identical K = [500 0 320; 0 500 240; 0 0 1], no distortion).
//! The left camera is the world frame; the right camera is `X_r = R X_l + T`.
//!
//! The baseline used to be forced onto +x, so a right camera mounted to the
//! left rectified 180 deg rotated and a vertical rig 90 deg, where OpenCV
//! leaves both unrotated and puts the sign into P2.

use cv_calib3d::stereo_rectify_matrices;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Matrix3, Rotation3, Vector3};

fn check(r: Matrix3<f64>, t: Vector3<f64>, r1: [f64; 9], r2: [f64; 9], p2_over_f: [f64; 3]) {
    let k = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let left = Pose::new(Matrix3::identity(), Vector3::zeros());
    let right = Pose::new(r, t);
    let m = stereo_rectify_matrices(&k, &k, &left, &right).unwrap();
    let want1 = Matrix3::from_row_slice(&r1);
    let want2 = Matrix3::from_row_slice(&r2);
    assert!((m.r1 - want1).norm() < 1e-9, "R1 {} vs cv2 {}", m.r1, want1);
    assert!((m.r2 - want2).norm() < 1e-9, "R2 {} vs cv2 {}", m.r2, want2);
    for row in 0..3 {
        assert!(
            (m.p2[(row, 3)] / m.p2[(0, 0)] - p2_over_f[row]).abs() < 1e-9,
            "P2[{row}][3]/f = {} vs cv2 {}",
            m.p2[(row, 3)] / m.p2[(0, 0)],
            p2_over_f[row]
        );
    }
}

const I: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn side_by_side_reversed_and_vertical_rigs_match_opencv() {
    check(
        Matrix3::identity(),
        Vector3::new(-0.1, 0.0, 0.0),
        I,
        I,
        [-0.1, 0.0, 0.0],
    );
    check(
        Matrix3::identity(),
        Vector3::new(0.1, 0.0, 0.0),
        I,
        I,
        [0.1, 0.0, 0.0],
    );
    check(
        Matrix3::identity(),
        Vector3::new(0.0, -0.1, 0.0),
        I,
        I,
        [0.0, -0.1, 0.0],
    );
}

#[test]
fn rotated_rig_matches_opencv() {
    let r = Rotation3::from_scaled_axis(Vector3::new(0.02, 0.09, -0.01)).into_inner();
    check(
        r,
        Vector3::new(-0.12, 0.01, 0.005),
        [
            0.996094934632872,
            -0.07292505865354865,
            0.04976763023402599,
            0.07333211671295393,
            0.9972870374856834,
            -0.006400431346940223,
            -0.04916586068739107,
            0.010025002913001341,
            0.9987403153269935,
        ],
        [
            0.9956877763740268,
            -0.0829739813645027,
            -0.04148699068225111,
            0.08331259385154388,
            0.9965022788070866,
            0.006497694822868813,
            0.04080274114656666,
            -0.00992606411457244,
            0.9991179157467449,
        ],
        [-0.1205197079319395, 0.0, 0.0],
    );
}
