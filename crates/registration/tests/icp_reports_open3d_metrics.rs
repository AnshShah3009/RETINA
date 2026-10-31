//! Point-to-plane ICP must report Open3D-style metrics for the pose it returns.
//!
//! Open3D's `RegistrationICP` re-evaluates the correspondence set *after* each
//! update, so `fitness` and `inlier_rmse` describe the returned transform, and
//! the RMSE is point-to-point (the distance to the nearest target point) even for
//! point-to-plane estimation. It stops once both fitness and RMSE change by less
//! than `relative_fitness` / `relative_rmse` (1e-6) between iterations.
//!
//! Before: the metrics were accumulated during the linearisation - at the
//! pre-update pose - but stored next to the post-update pose, and the RMSE was
//! the point-to-plane residual, so it never matched `evaluate_registration` on
//! the returned transform.

use cv_core::PointCloud;
use cv_registration::{evaluate_registration, registration_icp_point_to_plane};
use nalgebra::{Matrix4, Point3, Vector3};

/// A wavy surface z = 0.15 sin(4x) + 0.1 cos(3y) + 0.2xy. Not a quadric: on a
/// quadric an in-plane translation is equivalent, to first order, to a tilt
/// plus a z shift, so point-to-plane ICP has a near-degenerate valley there.
/// `noise` perturbs the target along z deterministically so the RMSE has a
/// non-zero floor.
fn bowl(offset: Vector3<f32>, noise: f32) -> (PointCloud, PointCloud) {
    let mut source = PointCloud::new(Vec::new());
    let mut target = PointCloud::new(Vec::new());
    let mut normals = Vec::new();
    for iy in -10..=10 {
        for ix in -10..=10 {
            let x = ix as f32 * 0.05;
            let y = iy as f32 * 0.05;
            let p = Point3::new(
                x,
                y,
                0.15 * (4.0 * x).sin() + 0.1 * (3.0 * y).cos() + 0.2 * x * y,
            );
            source.points.push(p);
            let jitter = noise * (((ix * 7 + iy * 13) % 5) as f32 - 2.0) / 2.0;
            target
                .points
                .push(p + offset + Vector3::new(0.0, 0.0, jitter));
            normals.push(
                Vector3::new(
                    -(0.6 * (4.0 * x).cos() + 0.2 * y),
                    -(-0.3 * (3.0 * y).sin() + 0.2 * x),
                    1.0,
                )
                .normalize(),
            );
        }
    }
    target.normals = Some(normals);
    (source, target)
}

#[test]
fn reported_metrics_equal_evaluate_registration_on_the_returned_pose() {
    for (offset, noise, iters) in [
        (Vector3::new(0.03, -0.02, 0.01), 0.0, 1),
        (Vector3::new(0.03, -0.02, 0.01), 0.0, 3),
        (Vector3::new(0.03, -0.02, 0.01), 0.0, 50),
        (Vector3::new(0.02, 0.01, -0.01), 0.0002, 50),
    ] {
        let (source, target) = bowl(offset, noise);
        let d = 0.1;
        let r = registration_icp_point_to_plane(&source, &target, d, &Matrix4::identity(), iters)
            .expect("bowl registration must succeed");
        let (fitness, rmse) = evaluate_registration(&source, &target, &r.transformation, d);
        assert!(
            (r.fitness - fitness).abs() < 1e-6,
            "iters={iters} noise={noise}: reported fitness {} != evaluated {}",
            r.fitness,
            fitness
        );
        assert!(
            (r.inlier_rmse - rmse).abs() <= 1e-6 * (1.0 + rmse),
            "iters={iters} noise={noise}: reported rmse {:e} != evaluate_registration \
             on the returned pose {:e}",
            r.inlier_rmse,
            rmse
        );
    }
}

#[test]
fn stops_on_relative_fitness_and_rmse_change() {
    // With noise the RMSE cannot reach zero, so only a relative criterion stops
    // the loop before max_iterations.
    let (source, target) = bowl(Vector3::new(0.02, 0.01, -0.01), 0.0002);
    let r = registration_icp_point_to_plane(&source, &target, 0.1, &Matrix4::identity(), 200)
        .expect("bowl registration must succeed");
    assert!(
        r.num_iterations < 200,
        "ran all 200 iterations: no relative_fitness / relative_rmse convergence"
    );
    assert!(
        (r.transformation[(0, 3)] - 0.02).abs() < 2e-3
            && (r.transformation[(1, 3)] - 0.01).abs() < 2e-3,
        "converged to a wrong pose: {}",
        r.transformation
    );
}
