//! `Camera::from(CameraIntrinsicsF32)` must keep `fy` and the principal point,
//! and `Gaussian::project` must use them.
//!
//! A single stored focal length and a principal point of `(width/2, height/2)`
//! map the camera-frame point `(0, 1, 2)` through `fx = 500` to `(320, 490)`
//! on a 640×480 image. The pinhole with `fy = 200`, `cx = 100`, `cy = 400` is
//! `(100, 500)`.

use cv_core::CameraIntrinsicsF32;
use cv_rendering::gaussian_splatting::{Camera, Gaussian, GaussianCloud, GaussianRasterizer};
use nalgebra::{Point3, Vector3, Vector4};

#[test]
fn anisotropic_intrinsics_project_off_centre_point() {
    let intrinsics = CameraIntrinsicsF32::new(500.0, 200.0, 100.0, 400.0, 640, 480);
    let camera = Camera::from(intrinsics);
    assert_eq!(camera.focal_length, 500.0);
    assert_eq!(camera.fy, 200.0);
    assert_eq!(camera.cx, 100.0);
    assert_eq!(camera.cy, 400.0);

    let mut cloud = GaussianCloud::new();
    // Isotropic scale so the screen covariance is s² J Jᵀ and the two focal
    // lengths show up as different diagonal entries.
    let scale = 0.05f32;
    cloud.push(Gaussian::new(
        Point3::new(0.0, 1.0, 2.0),
        Vector3::new(scale, scale, scale),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
    ));

    let projected = GaussianRasterizer::new(camera, 16, 16).project_gaussians(&cloud);
    let pg = &projected[0];

    let (px, py) = (pg.center.x, pg.center.y);
    println!("projected centre = ({px}, {py})");
    assert!(
        (px - 100.0).abs() < 1e-3 && (py - 500.0).abs() < 1e-3,
        "camera-frame (0, 1, 2) must project to the pinhole (100, 500), got ({px}, {py})"
    );

    // J row0 = (fx/z, 0, -fx x/z²) = (250, 0, 0)
    // J row1 = (0, fy/z, -fy y/z²) = (0, 100, -50)
    // Σ = s² I, so Σ_screen = s² J Jᵀ on the upper 2×2.
    let s2 = scale * scale;
    let cov_xx = s2 * 250.0 * 250.0;
    let cov_yy = s2 * (100.0 * 100.0 + 50.0 * 50.0);
    println!(
        "screen cov xx = {} (want {cov_xx}), yy = {} (want {cov_yy}), xy = {}",
        pg.covariance[(0, 0)],
        pg.covariance[(1, 1)],
        pg.covariance[(0, 1)]
    );
    assert!(
        (pg.covariance[(0, 0)] - cov_xx).abs() < 1e-2,
        "screen variance along x must use fx: got {} want {cov_xx}",
        pg.covariance[(0, 0)]
    );
    assert!(
        (pg.covariance[(1, 1)] - cov_yy).abs() < 1e-2,
        "screen variance along y must use fy: got {} want {cov_yy}",
        pg.covariance[(1, 1)]
    );
    assert!(
        pg.covariance[(0, 1)].abs() < 1e-3,
        "x = 0 makes the screen covariance diagonal, got {}",
        pg.covariance[(0, 1)]
    );
}
