//! Emit camera-projection values for `parity/parity_calib_project.py`.
//!
//! Prints only; asserts nothing. The comparison lives on the Python side so
//! `cv2.projectPoints` remains an independent reference rather than something this
//! file agrees with by construction.
//!
//! Run: `cargo run -q -p cv-calib3d --example parity_project`

use cv_calib3d::project::project_points_with_distortion;
use cv_core::geometry::{CameraIntrinsics, Distortion, Pose};
use nalgebra::{Point2, Point3, Rotation3, Vector3};

/// A deterministic point cloud in front of the camera.
///
/// Spans a range of radii so the distortion terms are exercised at several
/// strengths: at small radius the radial terms are near zero and a bug in them
/// would not show, so the cloud deliberately reaches out to r ~ 1.5.
fn world_points() -> Vec<Point3<f64>> {
    let mut pts = Vec::new();
    for i in 0..12 {
        let a = i as f64 * std::f64::consts::TAU / 12.0;
        for &r in &[0.2f64, 0.6, 1.0, 1.5] {
            pts.push(Point3::new(r * a.cos(), r * a.sin(), 2.0 + 0.1 * i as f64));
        }
    }
    pts
}

fn main() {
    let pts = world_points();

    // A real, non-trivial pose: a rotation and a translation, so the comparison
    // covers the extrinsic path and not just the intrinsic one.
    let axis = Vector3::new(0.3, -0.2, 1.0).normalize();
    let rot = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), 0.35);
    let pose = Pose {
        rotation: rot.into(),
        translation: Vector3::new(0.12, -0.07, 0.03),
    };

    let k = CameraIntrinsics::new(900.0, 905.0, 640.0, 360.0, 1280, 720);

    // Three distortion models: none (isolates the intrinsics and the pose), a
    // modest radtan, and a strongly radial one where the cubic term matters.
    let cases: [(&str, Distortion); 3] = [
        ("none", Distortion::none()),
        ("mild", Distortion::new(-0.28, 0.09, 0.001, -0.002, 0.0)),
        ("strong", Distortion::new(-0.82, 0.31, 0.004, -0.006, 0.012)),
    ];

    for (name, d) in cases {
        println!("#CASE {name}");
        println!(
            "#D {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
            d.k1, d.k2, d.p1, d.p2, d.k3
        );
        for p in &pts {
            println!("#P {:.17e} {:.17e} {:.17e}", p.x, p.y, p.z);
        }
        let img = project_points_with_distortion(&pts, &k, &pose, &d)
            .unwrap_or_else(|e| panic!("{name}: projection failed: {e}"));
        for p in &img {
            let _: Point2<f64> = *p;
            println!("#X {:.17e} {:.17e}", p.x, p.y);
        }
    }
}
