use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cv_stereo::calib3d::*;
use nalgebra::{Point2, Point3, Rotation3, Vector3};

fn generate_synthetic_calibration_data() -> (Vec<Vec<Point3<f64>>>, Vec<Vec<Point2<f64>>>) {
    let pattern_size = (7, 6);
    let board = (0..pattern_size.1)
        .flat_map(|y| {
            (0..pattern_size.0).map(move |x| Point3::new(x as f64 * 0.04, y as f64 * 0.04, 0.0))
        })
        .collect::<Vec<_>>();

    let intrinsics = cv_core::CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);

    let views = [
        cv_core::Pose::new(
            Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
            Vector3::new(0.05, -0.08, 0.3),
        ),
        cv_core::Pose::new(
            Rotation3::from_euler_angles(-0.06, 0.04, -0.05).into_inner(),
            Vector3::new(-0.08, 0.02, 0.35),
        ),
        cv_core::Pose::new(
            Rotation3::from_euler_angles(0.03, 0.07, -0.02).into_inner(),
            Vector3::new(0.02, 0.06, 0.32),
        ),
        cv_core::Pose::new(
            Rotation3::from_euler_angles(-0.04, -0.05, 0.04).into_inner(),
            Vector3::new(-0.03, -0.05, 0.33),
        ),
        cv_core::Pose::new(
            Rotation3::from_euler_angles(0.07, 0.02, -0.06).into_inner(),
            Vector3::new(0.08, -0.02, 0.31),
        ),
    ];

    let mut object_points = Vec::new();
    let mut image_points = Vec::new();

    for ext in &views {
        object_points.push(board.clone());
        let projected = board
            .iter()
            .map(|p| {
                let p_cam = ext.rotation * p.coords + ext.translation;
                let u = intrinsics.fx * (p_cam[0] / p_cam[2]) + intrinsics.cx;
                let v = intrinsics.fy * (p_cam[1] / p_cam[2]) + intrinsics.cy;
                Point2::new(u, v)
            })
            .collect();
        image_points.push(projected);
    }

    (object_points, image_points)
}

fn benchmark_planar_calibration(c: &mut Criterion) {
    let (object_points, image_points) = generate_synthetic_calibration_data();

    c.bench_function("calibrate_camera_planar_5_views", |b| {
        b.iter(|| {
            black_box(calibrate_camera_planar(
                &object_points,
                &image_points,
                (640, 480),
            ))
        });
    });
}

fn benchmark_projection_with_jacobians(c: &mut Criterion) {
    let intrinsics = cv_core::CameraIntrinsics::new(820.0, 790.0, 320.0, 240.0, 640, 480);
    let extrinsics = cv_core::Pose::new(
        Rotation3::from_euler_angles(0.08, -0.03, 0.02).into_inner(),
        Vector3::new(0.05, -0.08, 0.3),
    );
    let distortion = cv_core::Distortion::default();

    let points: Vec<Point3<f64>> = (0..50)
        .map(|i| Point3::new((i as f64) * 0.1, (i as f64) * 0.05, 2.0 + (i as f64) * 0.01))
        .collect();

    let mut group = c.benchmark_group("projection");

    group.bench_function("project_points_no_jacobians", |b| {
        b.iter(|| {
            black_box(project_points_with_jacobian(
                &points,
                &intrinsics,
                &extrinsics,
                &distortion,
                ProjectPointsOptions {
                    compute_jacobians: false,
                },
            ))
        });
    });

    group.bench_function("project_points_with_jacobians", |b| {
        b.iter(|| {
            black_box(project_points_with_jacobian(
                &points,
                &intrinsics,
                &extrinsics,
                &distortion,
                ProjectPointsOptions {
                    compute_jacobians: true,
                },
            ))
        });
    });

    group.finish();
}

fn benchmark_calibration_with_flags(c: &mut Criterion) {
    let (object_points, image_points) = generate_synthetic_calibration_data();
    let mut group = c.benchmark_group("calibration_flags");

    let options_no_flags = CameraCalibrationOptions::default();
    group.bench_function("no_flags", |b| {
        b.iter(|| {
            black_box(calibrate_camera_planar_with_options(
                &object_points,
                &image_points,
                (640, 480),
                options_no_flags,
            ))
        });
    });

    let options_fix_focal = CameraCalibrationOptions {
        fix_focal_length: true,
        ..Default::default()
    };
    group.bench_function("fix_focal_length", |b| {
        b.iter(|| {
            black_box(calibrate_camera_planar_with_options(
                &object_points,
                &image_points,
                (640, 480),
                options_fix_focal,
            ))
        });
    });

    let options_multiple = CameraCalibrationOptions {
        fix_focal_length: true,
        zero_tangent_dist: true,
        fix_aspect_ratio: Some(1.0),
        ..Default::default()
    };
    group.bench_function("multiple_flags", |b| {
        b.iter(|| {
            black_box(calibrate_camera_planar_with_options(
                &object_points,
                &image_points,
                (640, 480),
                options_multiple,
            ))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_planar_calibration,
    benchmark_projection_with_jacobians,
    benchmark_calibration_with_flags
);
criterion_main!(benches);
