//! `stereo_rectify_matrices` with non-square pixels (`fx != fy`).
//!
//! `Q` is a single-focal-length matrix: `reprojectImageTo3D` gives
//! `(u - cx, v - cy, f) / (d / B)`, so `X` and `Y` both divide by the one `f`
//! in `Q(2, 3)`. That only inverts `P1` when `P1` itself has `fx == fy`, which
//! is why `cv2.stereoRectify` gives both rectified cameras one focal length,
//! `min(K1[i, i], K2[i, i])` on the axis *across* the baseline (`fy` for a
//! side-by-side rig). The averaged `fx`/`fy` used here kept them distinct, so
//! with `fx = 500`, `fy = 650` the point `(0.3, 0.4, 2.0)` reprojected through
//! `Q` to `(0.3, 0.52, 2.0)`: `Y` off by `fy / fx`. cv2 4.13 on the same rig:
//! `P1 = [[650, 0, 320.15, 0], [0, 650, 240, 0], [0, 0, 1, 0]]`, round trip
//! `(0.3, 0.4, 2.0)`.

use cv_calib3d::stereo_rectify_matrices;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Matrix3, Point3, Rotation3, Unit, Vector3, Vector4};

fn round_trip(left: &Pose, right: &Pose, kl: &CameraIntrinsics, kr: &CameraIntrinsics) {
    let m = stereo_rectify_matrices(kl, kr, left, right).unwrap();
    let c_l = -(left.rotation_matrix().transpose() * left.translation);
    for x in [
        Point3::new(0.3, 0.4, 2.0),
        Point3::new(-0.5, 0.2, 3.5),
        Point3::new(0.1, -0.6, 1.2),
    ] {
        // Rectified left-camera frame.
        let pr = m.r1 * left.rotation_matrix() * (x.coords - c_l);
        let a = m.p1 * Vector4::new(pr.x, pr.y, pr.z, 1.0);
        let b = m.p2 * Vector4::new(pr.x, pr.y, pr.z, 1.0);
        let (ul, vl) = (a.x / a.z, a.y / a.z);
        let (ur, vr) = (b.x / b.z, b.y / b.z);
        // Disparity along whichever axis holds the baseline.
        let d = if (ul - ur).abs() > (vl - vr).abs() {
            ul - ur
        } else {
            vl - vr
        };
        let h = m.q * Vector4::new(ul, vl, d, 1.0);
        let rec = h.xyz() / h.w;
        assert!(
            (rec - pr).norm() < 1e-9 * pr.norm(),
            "Q reprojects {:?} to {:?} (P1 = {:?})",
            pr,
            rec,
            m.p1
        );
    }
}

#[test]
fn q_inverts_p1_with_non_square_pixels_horizontal_rig() {
    let kl = CameraIntrinsics::new(500.0, 650.0, 320.0, 240.0, 640, 480);
    let kr = CameraIntrinsics::new(510.0, 640.0, 318.0, 242.0, 640, 480);
    let left = Pose::new(Matrix3::identity(), Vector3::zeros());
    let right = Pose::new(Matrix3::identity(), Vector3::new(-0.1, 0.0, 0.0));
    round_trip(&left, &right, &kl, &kr);

    // cv2 gives both cameras the smaller focal across the baseline.
    let m = stereo_rectify_matrices(&kl, &kr, &left, &right).unwrap();
    for p in [m.p1, m.p2] {
        assert_eq!(p[(0, 0)], 640.0);
        assert_eq!(p[(1, 1)], 640.0);
    }
}

#[test]
fn q_inverts_p1_with_non_square_pixels_rotated_and_vertical_rigs() {
    let kl = CameraIntrinsics::new(700.0, 520.0, 320.0, 240.0, 640, 480);
    let kr = CameraIntrinsics::new(690.0, 530.0, 322.0, 238.0, 640, 480);
    let r_l = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(0.1, 0.05, 1.0)), 0.15)
        .into_inner();
    let r_r =
        Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(-0.07, 0.12, 1.0)), -0.2)
            .into_inner();
    let left = Pose::new(r_l, Vector3::new(0.02, -0.01, 0.05));
    let right = Pose::new(r_r, Vector3::new(-0.18, 0.03, 0.06));
    round_trip(&left, &right, &kl, &kr);

    let vertical = Pose::new(Matrix3::identity(), Vector3::new(0.0, -0.12, 0.0));
    round_trip(
        &Pose::new(Matrix3::identity(), Vector3::zeros()),
        &vertical,
        &kl,
        &kr,
    );
}
