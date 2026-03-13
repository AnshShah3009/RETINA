use cv_calib3d::*;
use cv_core::{CameraIntrinsics, Distortion, Pose};
use image::{GrayImage, Luma};
use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};

#[test]
fn test_corner_subpix_basic() {
    let mut img = GrayImage::new(20, 20);
    // Draw a sharp white square on black background
    for y in 5..15 {
        for x in 5..15 {
            img.put_pixel(x, y, Luma([255]));
        }
    }

    // Initial guess near the corner (5, 5)
    let mut corners = vec![Point2::new(5.2, 4.8)];
    corner_subpix(&img, &mut corners, 3, 10, 0.001).unwrap();

    // Should move closer to (5, 5) or at least stay valid
    assert!(corners[0].x >= 4.0 && corners[0].x <= 6.0);
    assert!(corners[0].y >= 4.0 && corners[0].y <= 6.0);
}

#[test]
fn test_project_points_roundtrip() {
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let extrinsics = Pose::new(Matrix3::identity(), Vector3::zeros());
    let distortion = Distortion::none();

    let pts3d = vec![Point3::new(0.0, 0.0, 2.0)]; // 2 meters in front
    let res =
        project_points_with_distortion(&pts3d, &intrinsics, &extrinsics, &distortion).unwrap();

    assert_eq!(res.len(), 1);
    // Should project to principal point
    assert!((res[0].x - 320.0).abs() < 1e-5);
    assert!((res[0].y - 240.0).abs() < 1e-5);
}

#[test]
fn test_undistort_rectify_map_dimensions() {
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let distortion = Distortion::none();
    let rect = Matrix3::identity();

    let (map_x, map_y) =
        init_undistort_rectify_map((640, 480), &intrinsics, &distortion, &rect, &intrinsics)
            .unwrap();

    assert_eq!(map_x.len(), 640 * 480);
    assert_eq!(map_y.len(), 640 * 480);
}

#[test]
fn test_solve_pnp_dlt_simple() {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0, 800, 600);

    // 6 points in a plane (Z=2)
    let obj_pts = vec![
        Point3::new(-1.0, -1.0, 2.0),
        Point3::new(1.0, -1.0, 2.0),
        Point3::new(1.0, 1.0, 2.0),
        Point3::new(-1.0, 1.0, 2.0),
        Point3::new(0.0, 0.5, 2.0),
        Point3::new(0.5, 0.0, 2.0),
    ];

    let img_pts = vec![
        Point2::new(0.0, 0.0), // Simplified for test
        Point2::new(800.0, 0.0),
        Point2::new(800.0, 600.0),
        Point2::new(0.0, 600.0),
        Point2::new(400.0, 500.0),
        Point2::new(600.0, 300.0),
    ];

    // DLT should at least run without error
    let res = solve_pnp_dlt(&obj_pts, &img_pts, &intrinsics);
    assert!(res.is_ok());
}
