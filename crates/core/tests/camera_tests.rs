use cv_core::geometry::{CameraIntrinsics, CameraModel, Distortion, PinholeModel};
use nalgebra::{Point2, Point3};

#[test]
fn test_pinhole_projection_no_distortion() {
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let distortion = Distortion::none();
    let model = PinholeModel::new(intrinsics, distortion);

    let p3 = Point3::new(1.0, 1.0, 5.0);
    let p2 = model.project(&p3);

    // x = 1.0 * 500 / 5.0 + 320 = 420
    // y = 1.0 * 500 / 5.0 + 240 = 340
    assert!((p2.x - 420.0).abs() < 1e-5);
    assert!((p2.y - 340.0).abs() < 1e-5);

    let p3_back = model.unproject(&p2, 5.0);
    assert!((p3_back.x - 1.0).abs() < 1e-5);
    assert!((p3_back.y - 1.0).abs() < 1e-5);
    assert!((p3_back.z - 5.0).abs() < 1e-5);
}

#[test]
fn test_pinhole_distortion() {
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let distortion = Distortion::new(0.1, 0.01, 0.0, 0.0, 0.0); // Small radial distortion
    let model = PinholeModel::new(intrinsics, distortion);

    let p3 = Point3::new(1.0, 1.0, 5.0);
    let p2 = model.project(&p3);

    // Without distortion it was (420, 340)
    // Normalized coords: (0.2, 0.2). r2 = 0.08.
    // Radial = 1 + 0.1*0.08 + 0.01*0.0064 = 1.008064
    // Distorted x = 0.2 * 1.008064 = 0.2016128
    // Distorted pixel x = 0.2016128 * 500 + 320 = 420.8064

    assert!(p2.x > 420.5);

    let p3_back = model.unproject(&p2, 5.0);
    assert!((p3_back.x - 1.0).abs() < 1e-3); // Iterative removal check
    assert!((p3_back.y - 1.0).abs() < 1e-3);
}
