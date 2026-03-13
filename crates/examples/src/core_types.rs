//! Example: Core Types and Frame Conventions
//!
//! Run with: cargo run --example core_types

use cv_core::{
    CameraConvention, CameraIntrinsics, FrameConvention, Handedness, KeyPoint, PointCloud,
    RigOrientation, Tensor,
};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

fn main() {
    println!("=== cv-core Example ===\n");

    println!("1. Frame Conventions:");
    frame_conventions();

    println!("\n2. Camera Intrinsics:");
    camera_intrinsics();

    println!("\n3. Point Cloud:");
    point_cloud_example();

    println!("\n4. KeyPoints:");
    keypoints_example();

    println!("\n=== Example Complete ===");
}

fn frame_conventions() {
    // Default: OpenCV (RH, +Z backward, +Y down)
    let opencv = FrameConvention::opencv();
    println!("  OpenCV: {:?}", opencv.camera_convention);
    println!("    Handedness: {:?}", opencv.handedness);

    // OpenGL (LH, +Z backward, +Y up)
    let opengl = FrameConvention::opengl();
    println!("  OpenGL: {:?}", opengl.camera_convention);
    println!("    Handedness: {:?}", opengl.handedness);

    // COLMAP (RH, +Z forward, +Y down)
    let colmap = FrameConvention::colmap();
    println!("  COLMAP: {:?}", colmap.camera_convention);
    println!("    Handedness: {:?}", colmap.handedness);

    // Custom convention
    let custom = FrameConvention::right_handed(CameraConvention::WebGPU);
    println!("  Custom (RH + WebGPU): {:?}", custom.camera_convention);

    // Convert pose between conventions
    let rotation = Matrix3::identity();
    let converted = opencv.convert_rotation(&colmap, &rotation);
    println!("  Converted identity rotation");
}

fn camera_intrinsics() {
    // Create camera intrinsics (fx, fy, cx, cy, width, height)
    let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    println!(
        "  Camera: fx={}, fy={}, cx={}, cy={}",
        cam.fx, cam.fy, cam.cx, cam.cy
    );
    println!("  Size: {}x{}", cam.width, cam.height);

    // Project 3D point to 2D
    let point_3d = Point3::new(1.0, 2.0, 5.0);
    let point_2d = cam.project(&point_3d);
    println!(
        "  Projected ({:?}) -> ({:.1}, {:.1})",
        point_3d, point_2d.x, point_2d.y
    );

    // Back-project 2D point to 3D at given depth
    use nalgebra::Point2;
    let pixel = Point2::new(320.0, 240.0);
    let ray = cam.unproject(pixel, 5.0);
    println!("  Unproject (320, 240, depth=5) -> point: {:?}", ray);
}

fn point_cloud_example() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];

    let cloud = PointCloud::new(points);
    println!("  Point cloud size: {}", cloud.len());

    // With colors
    let cloud_with_colors =
        PointCloud::new(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)])
            .with_colors(vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)])
            .unwrap();

    println!("  Point cloud with colors: {}", cloud_with_colors.len());
}

fn keypoints_example() {
    let kp = KeyPoint::new(100.0, 200.0)
        .with_size(10.0)
        .with_angle(45.0)
        .with_response(0.5);
    println!(
        "  KeyPoint: x={}, y={}, size={}, angle={}, response={}",
        kp.x, kp.y, kp.size, kp.angle, kp.response
    );
}
