//! PnP refinement is over-determined from four points (8 residuals, 6
//! parameters). It used to demand six, so `solve_pnp_ransac` returned the raw
//! minimal-sample P3P pose whenever it found only four or five inliers.

use cv_calib3d::solve_pnp_refine;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Point2, Point3, Rotation3, Vector3};

#[test]
fn four_points_refine_a_perturbed_pose_to_the_truth() {
    let k = CameraIntrinsics::new(600.0, 600.0, 320.0, 240.0, 640, 480);
    let r = Rotation3::from_scaled_axis(Vector3::new(0.1, -0.2, 0.05)).into_inner();
    let t = Vector3::new(0.1, -0.05, 4.0);
    let truth = Pose::new(r, t);
    let obj = [
        Point3::new(-0.5, -0.4, 0.1),
        Point3::new(0.6, -0.3, -0.2),
        Point3::new(0.4, 0.5, 0.3),
        Point3::new(-0.3, 0.4, -0.1),
    ];
    let img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| {
            let c = r * p.coords + t;
            Point2::new(600.0 * c.x / c.z + 320.0, 600.0 * c.y / c.z + 240.0)
        })
        .collect();
    let start = Pose::new(
        Rotation3::from_scaled_axis(Vector3::new(0.13, -0.17, 0.07)).into_inner(),
        t + Vector3::new(0.05, 0.03, -0.1),
    );
    let refined = solve_pnp_refine(&start, &obj, &img, &k, None, 50).expect("4 points suffice");
    assert!(
        (refined.translation - truth.translation).norm() < 1e-6,
        "{:?}",
        refined.translation
    );
    assert!((refined.rotation_matrix() - r).norm() < 1e-6);
}
