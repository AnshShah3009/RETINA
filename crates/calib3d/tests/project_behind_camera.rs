//! Points behind the camera must be rejected. `(1, 0, -5)` with `fx = 100`,
//! `cx = 50` used to come back `Ok` at `u = 30`; the same point in front
//! projects to `u = 70` and must keep doing so.

use cv_calib3d::{
    ProjectPointsOptions, project_points, project_points_with_distortion,
    project_points_with_jacobian,
};
use cv_core::{CameraIntrinsics, Distortion, Pose};
use nalgebra::Point3;

#[test]
fn behind_camera_is_rejected_and_in_front_projects_to_70() {
    let k = CameraIntrinsics::new(100.0, 80.0, 50.0, 40.0, 100, 80);
    let pose = Pose::identity();
    let distortion = Distortion::none();
    let behind = Point3::new(1.0, 0.0, -5.0);
    let front = Point3::new(1.0, 0.0, 5.0);

    assert!(
        project_points(&[behind], &k, &pose).is_err(),
        "z = -5 is behind the camera"
    );
    let projected = project_points(&[front], &k, &pose).expect("z = +5 is in front");
    assert_eq!(projected[0].x, 70.0);
    assert_eq!(projected[0].y, 40.0);

    assert!(project_points_with_distortion(&[behind], &k, &pose, &distortion).is_err());
    let projected =
        project_points_with_distortion(&[front], &k, &pose, &distortion).expect("in front");
    assert_eq!(projected[0].x, 70.0);
    assert_eq!(projected[0].y, 40.0);

    let options = ProjectPointsOptions::default();
    assert!(project_points_with_jacobian(&[behind], &k, &pose, &distortion, options).is_err());
    let projected =
        project_points_with_jacobian(&[front], &k, &pose, &distortion, options).expect("in front");
    assert_eq!(projected.image_points[0].x, 70.0);
    assert_eq!(projected.image_points[0].y, 40.0);
}
