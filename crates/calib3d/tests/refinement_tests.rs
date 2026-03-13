use cv_calib3d::{solve_pnp_dlt, solve_pnp_refine};
use cv_core::{CameraIntrinsics, Distortion, Pose};
use nalgebra::{Point2, Point3, Rotation3, Vector3};

#[test]
fn test_pnp_refine_with_distortion() {
    let k = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480);
    let d = Distortion::new(0.1, -0.05, 0.001, -0.002, 0.01);

    let gt_pose = Pose::new(
        Rotation3::from_euler_angles(0.1, -0.2, 0.3)
            .matrix()
            .clone_owned(),
        Vector3::new(0.5, -0.3, 2.0),
    );

    // Generate synthetic points
    let mut obj_pts = Vec::new();
    let mut img_pts = Vec::new();
    for i in 0..10 {
        let p = Point3::new(i as f64 * 0.1, (i % 3) as f64 * 0.2, 2.0 + i as f64 * 0.1);
        obj_pts.push(p);

        // Project with distortion
        let pc = gt_pose.rotation * p.coords + gt_pose.translation;
        let x = pc[0] / pc[2];
        let y = pc[1] / pc[2];
        let (xd, yd) = d.apply(x, y);
        img_pts.push(Point2::new(xd * k.fx + k.cx, yd * k.fy + k.cy));
    }

    // Initial guess from DLT (which doesn't know about distortion)
    // We expect DLT to be slightly off because it ignores d
    let dlt_pose = solve_pnp_dlt(&obj_pts, &img_pts, &k).unwrap();

    // 1. Refine without distortion (should still be somewhat off)
    let refined_no_d = solve_pnp_refine(&dlt_pose, &obj_pts, &img_pts, &k, None, 50).unwrap();

    // 2. Refine with distortion (should be very close to gt)
    let refined_with_d = solve_pnp_refine(&dlt_pose, &obj_pts, &img_pts, &k, Some(&d), 50).unwrap();

    let err_no_d = (refined_no_d.translation - gt_pose.translation).norm();
    let err_with_d = (refined_with_d.translation - gt_pose.translation).norm();

    println!("Error without distortion info: {}", err_no_d);
    println!("Error with distortion info: {}", err_with_d);

    assert!(err_with_d < err_no_d);
    assert!(err_with_d < 1e-3);
}
