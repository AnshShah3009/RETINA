//! `fisheye_stereo_rectify` must scale the baseline on whichever axis holds it.
//!
//! The function shrinks the rectified focal length by `fov_scale` to keep
//! fisheye edges from warping. That only stays consistent if P2's baseline
//! entry (`P2[idx][3] = T'[idx] * f`, where `idx` is the baseline's dominant
//! axis) is scaled by the same factor. It used to scale `P2[0][3]`
//! unconditionally, which is the horizontal rig's row; a vertical rig left its
//! baseline on row 1 untouched while `fy` shrank, so the disparity-to-depth
//! mapping in Q recovered a depth off by exactly `fov_scale`.

use cv_calib3d::stereo::fisheye_stereo_rectify;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Matrix3, Point3, Vector3, Vector4};

#[test]
fn vertical_rig_recovers_true_depth_after_fisheye_rectification() {
    let k = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let left = Pose::new(Matrix3::identity(), Vector3::zeros());
    // Vertical stereo: the right camera sits 0.1 m below the left one, so the
    // baseline is on the y axis and `stereo_rectify_matrices` puts it in
    // P2[1][3].
    let right = Pose::new(Matrix3::identity(), Vector3::new(0.0, -0.1, 0.0));

    let fish = cv_core::FisheyeDistortion::none();
    let fov_scale = 0.5;
    let m = fisheye_stereo_rectify(&k, &k, &fish, &fish, &left, &right, fov_scale)
        .expect("vertical fisheye rectification must succeed");

    // The baseline row must carry the scaled baseline, while the focal length
    // shrank by `fov_scale`.
    let unscaled_baseline = -0.1 * k.fy;
    assert!(
        (m.p2[(1, 3)] - fov_scale * unscaled_baseline).abs() < 1e-9,
        "P2[1][3] = {} not scaled by {fov_scale}: expected {}",
        m.p2[(1, 3)],
        fov_scale * unscaled_baseline
    );
    assert_eq!(m.p2[(0, 3)], 0.0, "a vertical rig has no x baseline");

    // A point 3 m away must come back at 3 m through the Q matrix.
    let p_world = Point3::new(0.4, 0.25, 3.0);
    let c_left = -(left.rotation_matrix().transpose() * left.translation);
    let p_left = m.r1 * left.rotation_matrix() * (p_world - c_left);

    let pl = m.p1 * Vector4::new(p_left.x, p_left.y, p_left.z, 1.0);
    let pr = m.p2 * Vector4::new(p_left.x, p_left.y, p_left.z, 1.0);
    let u_l = pl.x / pl.z;
    let v_l = pl.y / pl.z;
    let v_r = pr.y / pr.z;
    let disparity = v_l - v_r; // vertical disparity

    let hom = m.q * Vector4::new(u_l, v_l, disparity, 1.0);
    let recovered_z = hom.z / hom.w;
    assert!(
        (recovered_z - 3.0).abs() / 3.0 < 1e-6,
        "Q recovered depth {recovered_z}, expected 3.0 (fov_scale={fov_scale})"
    );
}
