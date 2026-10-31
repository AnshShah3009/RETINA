//! Colored ICP (Park, Zhou, Koltun 2017; Open3D's
//! `TransformationEstimationForColoredICP`) must recover a known rigid offset.
//!
//! Before: the "photometric Jacobian" had no image gradient in it and the
//! geometric Jacobian was taken at the *untransformed* source point, with an
//! unsigned residual, so a 0.01 offset that point-to-plane ICP recovers exactly
//! was not recovered.

use cv_core::PointCloud;
use cv_registration::{
    evaluate_registration, registration_colored_icp, registration_icp_point_to_plane,
};
use nalgebra::{Matrix4, Point3, Rotation3, Translation3, Vector3};

/// A gently curved, smoothly coloured surface z = 0.4x^2 + 0.15y^2 + 0.2xy
/// (not a surface of revolution, so the geometry pins every DOF) with colours
/// in [0, 1] (Open3D's convention), plus analytic normals.
fn surface() -> PointCloud {
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();
    for iy in -25..=25 {
        for ix in -25..=25 {
            let x = ix as f32 * 0.02;
            let y = iy as f32 * 0.02;
            points.push(Point3::new(x, y, 0.4 * x * x + 0.15 * y * y + 0.2 * x * y));
            let r = 0.5 + 0.4 * (6.0 * x).sin() * (4.0 * y).cos();
            let g = 0.5 + 0.3 * (5.0 * y).sin();
            let b = 0.5 + 0.2 * (3.0 * (x + y)).cos();
            colors.push(Point3::new(r, g, b));
            normals.push(Vector3::new(-(0.8 * x + 0.2 * y), -(0.3 * y + 0.2 * x), 1.0).normalize());
        }
    }
    PointCloud {
        points,
        colors: Some(colors),
        normals: Some(normals),
    }
}

fn ground_truth() -> Matrix4<f32> {
    (Translation3::new(0.01, -0.006, 0.004) * Rotation3::from_euler_angles(0.004, -0.003, 0.006))
        .to_homogeneous()
}

/// Source = target moved by the inverse of the ground truth, colours carried
/// along, so `ground_truth()` maps source exactly onto target.
fn scene() -> (PointCloud, PointCloud) {
    let target = surface();
    let inv = ground_truth().try_inverse().unwrap();
    let source = PointCloud {
        points: target
            .points
            .iter()
            .map(|p| inv.transform_point(p))
            .collect(),
        colors: target.colors.clone(),
        normals: None,
    };
    (source, target)
}

fn pose_error(t: &Matrix4<f32>) -> f32 {
    (t - ground_truth()).abs().max()
}

#[test]
fn colored_icp_recovers_a_small_rigid_offset() {
    let (source, target) = scene();
    for lambda in [0.968f32, 0.5, 1.0] {
        let r = registration_colored_icp(&source, &target, 0.05, &Matrix4::identity(), 50, lambda)
            .unwrap_or_else(|| panic!("lambda={lambda}: returned None"));
        let err = pose_error(&r.transformation);
        assert!(
            err < 1e-4,
            "lambda={lambda}: pose error {err:e} (want < 1e-4)\n{}",
            r.transformation
        );
        // Open3D reports point-to-point fitness/rmse for the returned pose.
        let (fitness, rmse) = evaluate_registration(&source, &target, &r.transformation, 0.05);
        assert!((r.fitness - fitness).abs() < 1e-6);
        assert!((r.inlier_rmse - rmse).abs() <= 1e-6 * (1.0 + rmse));
    }
}

#[test]
fn point_to_plane_control_also_recovers_it() {
    // Control: the geometry alone determines the pose, so a failure above is
    // the colored estimator's, not the scene's.
    let (source, target) = scene();
    let r = registration_icp_point_to_plane(&source, &target, 0.05, &Matrix4::identity(), 50)
        .expect("point-to-plane must register the scene");
    assert!(pose_error(&r.transformation) < 1e-4, "{}", r.transformation);
}

#[test]
fn colour_resolves_a_slide_the_geometry_cannot_see() {
    // On a plane, in-plane translation is invisible to the point-to-plane term;
    // only the colour gradient can recover it.
    let mut points = Vec::new();
    let mut colors = Vec::new();
    for iy in -25..=25 {
        for ix in -25..=25 {
            let x = ix as f32 * 0.02;
            let y = iy as f32 * 0.02;
            points.push(Point3::new(x, y, 0.0));
            let r = 0.5 + 0.4 * (6.0 * x).sin() * (4.0 * y).cos();
            let g = 0.5 + 0.3 * (5.0 * y).sin();
            let b = 0.5 + 0.2 * (3.0 * x).cos();
            colors.push(Point3::new(r, g, b));
        }
    }
    let target = PointCloud {
        points: points.clone(),
        colors: Some(colors.clone()),
        normals: Some(vec![Vector3::z(); points.len()]),
    };
    // A slide of 0.005 in x and -0.004 in y: below the 0.02 grid spacing.
    let slide = Vector3::new(0.005f32, -0.004, 0.0);
    let source = PointCloud {
        points: points.iter().map(|p| p - slide).collect(),
        colors: Some(colors),
        normals: None,
    };
    let r = registration_colored_icp(&source, &target, 0.05, &Matrix4::identity(), 100, 0.5)
        .expect("colored ICP must register a textured plane");
    let t = r.transformation.fixed_view::<3, 1>(0, 3).into_owned();
    // The colour field is sampled on a 0.02 grid and linearised per point, so
    // the slide is recovered approximately, not exactly; geometry alone would
    // leave it at zero.
    assert!(
        (t - slide).norm() < 1.5e-3,
        "in-plane slide not recovered: got {t:?}, want {slide:?}"
    );
}
