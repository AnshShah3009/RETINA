//! The CPU normals fallback must produce **one normal per point** - the thing
//! its two historical versions failed: one returned a zero vector per point
//! (fabricated), the next `Vec::new()` (0 normals for N points, silently).
//! Both only surfaced when the GPU was unavailable, i.e. never in a
//! GPU-equipped test run.

#![forbid(unsafe_code)]

use cv_hal::gpu_kernels::pointcloud::normals_cpu_analytic;
use nalgebra::Vector3;

#[test]
fn a_planar_cloud_gets_the_plane_normal() {
    let mut pts = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            pts.push(Vector3::new(i as f32, j as f32, 2.0));
        }
    }
    let normals = normals_cpu_analytic(&pts, 8);
    assert_eq!(
        normals.len(),
        pts.len(),
        "one normal per point - the fallback used to return none"
    );
    for (i, n) in normals.iter().enumerate() {
        assert!(
            (n.z.abs() - 1.0).abs() < 1e-3,
            "point {i}: a z=2 plane has its normal along z, got {n}"
        );
    }
}

#[test]
fn a_spherical_cloud_gets_radial_normals() {
    // Fibonacci sphere: spread points with no lattice structure for the PCA
    // neighbourhoods to key off.
    let n = 200usize;
    let phi = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let pts: Vec<Vector3<f32>> = (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let th = phi * i as f32;
            Vector3::new(th.cos() * r, y, th.sin() * r) * 7.5
        })
        .collect();

    let normals = normals_cpu_analytic(&pts, 12);
    assert_eq!(normals.len(), pts.len());
    for (i, (p, n)) in pts.iter().zip(&normals).enumerate() {
        let radial = p.normalize();
        let alignment = n.dot(&radial).abs();
        assert!(
            alignment > 0.98,
            "point {i}: |dot(normal, radial)| = {alignment:.4}, expected ~1 (the \
             eigenvector's sign is arbitrary; the direction is not)"
        );
    }
}

#[test]
fn degenerate_inputs_stay_finite_and_per_point() {
    // A single point and a k larger than the cloud must not panic or drop
    // points: the PCA covariance is zero, and any unit axis is an honest
    // answer for a neighbourhood with no variation.
    for (pts, k) in [
        (vec![Vector3::new(1.0, 2.0, 3.0)], 16),
        (
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)],
            8,
        ),
    ] {
        let normals = normals_cpu_analytic(&pts, k);
        assert_eq!(normals.len(), pts.len());
        assert!(normals.iter().all(|n| n.iter().all(|v| v.is_finite())));
    }
    assert!(normals_cpu_analytic(&[], 8).is_empty());
}
