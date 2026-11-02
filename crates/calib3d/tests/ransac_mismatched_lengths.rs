//! The E/F RANSAC entry points must reject correspondence lists of different
//! lengths, as OpenCV does.
//!
//! Both zipped `pts1` with `pts2`, which stops at the shorter list: 60 points
//! against 30 returned `Ok` with a 30-entry inlier mask, so a caller indexing
//! the mask by `pts1` either read past its end or attributed inliers to the
//! wrong points.

use cv_calib3d::{find_essential_mat_ransac, find_fundamental_mat_ransac};
use cv_core::CameraIntrinsics;
use nalgebra::{Point2, Point3, Rotation3, Vector3};

fn scene() -> (CameraIntrinsics, Vec<Point2<f64>>, Vec<Point2<f64>>) {
    let k = CameraIntrinsics::new(700.0, 700.0, 320.0, 240.0, 640, 480);
    let r = Rotation3::from_euler_angles(0.05, -0.1, 0.03).into_inner();
    let t = Vector3::new(-0.3, 0.05, 0.1);
    let project = |c: Vector3<f64>| Point2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..60 {
        let s = i as f64;
        let x = Point3::new(
            (s * 0.37).sin() * 1.5,
            (s * 0.71).cos(),
            4.0 + (s * 0.13).sin(),
        );
        a.push(project(x.coords));
        b.push(project(r * x.coords + t));
    }
    (k, a, b)
}

#[test]
fn essential_ransac_rejects_mismatched_lengths() {
    let (k, a, b) = scene();
    assert!(find_essential_mat_ransac(&a, &b, &k, 1.0, 200).is_ok());
    let r = find_essential_mat_ransac(&a, &b[..30], &k, 1.0, 200);
    assert!(
        r.is_err(),
        "60 vs 30 points returned Ok with a {}-entry mask",
        r.map(|x| x.1.len()).unwrap_or(0)
    );
}

#[test]
fn fundamental_ransac_rejects_mismatched_lengths() {
    let (_, a, b) = scene();
    assert!(find_fundamental_mat_ransac(&a, &b, 1.0, 200).is_ok());
    let r = find_fundamental_mat_ransac(&a[..30], &b, 1.0, 200);
    assert!(
        r.is_err(),
        "30 vs 60 points returned Ok with a {}-entry mask",
        r.map(|x| x.1.len()).unwrap_or(0)
    );
}
