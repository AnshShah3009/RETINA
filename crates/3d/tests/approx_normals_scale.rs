//! The approximate normal estimators give the same directions whatever units
//! the cloud is stored in.
//!
//! `compute_normals_approx_cross` and `compute_normals_approx_integral` build
//! the normal from cross products of neighbour offsets, whose length scales as
//! the square of the point spacing, and rejected it as degenerate below an
//! absolute `1e-10`. That is not "these neighbours are collinear", it is "the
//! spacing is below about 1e-5" (and, at unit scale, rounding noise from truly
//! collinear neighbours, about 1e-8, passed it). The degeneracy test is now
//! relative to the offsets' lengths.

use cv_3d::gpu::point_cloud::{compute_normals_approx_cross, compute_normals_approx_integral};
use nalgebra::{Point3, Vector3};

/// A jittered patch of the plane z = 0.4 x + 0.3 y, spacing about 0.05,
/// scaled by `scale`.
fn tilted_plane(scale: f32) -> Vec<Point3<f32>> {
    let mut pts = Vec::new();
    for i in 0..20 {
        for j in 0..20 {
            let t = (i * 20 + j) as f32;
            let x = i as f32 * 0.05 + 0.012 * (t * 1.7).sin();
            let y = j as f32 * 0.05 + 0.012 * (t * 2.3).cos();
            pts.push(Point3::new(
                x * scale,
                y * scale,
                (0.4 * x + 0.3 * y) * scale,
            ));
        }
    }
    pts
}

type Estimator = fn(&[Point3<f32>], f32) -> Vec<Vector3<f32>>;

/// Normals at `scale` against the same cloud at scale 1, cell size scaled too.
fn compare(name: &str, f: Estimator) {
    let reference = f(&tilted_plane(1.0), 0.1);
    let defaults = reference.iter().filter(|n| **n == Vector3::z()).count();
    assert!(
        defaults < reference.len() / 10,
        "{name}: {defaults} default normals at scale 1"
    );
    // 2^-20 (9.5e-7) scales every coordinate, offset and cross product exactly,
    // so the directions must be bit-identical; 1e-6 itself rounds the input.
    let exact = 2f32.powi(-20);
    assert_eq!(
        f(&tilted_plane(exact), 0.1 * exact),
        reference,
        "{name} at scale 2^-20"
    );
    let got = f(&tilted_plane(1e-6), 0.1 * 1e-6);
    let worst = got
        .iter()
        .zip(&reference)
        // Up to sign: rounding the input can reorder the integral's
        // neighbour ring, which reverses its (unoriented) result.
        .map(|(g, r)| g.dot(r).abs())
        .fold(f32::INFINITY, f32::min);
    let new_defaults = got
        .iter()
        .zip(&reference)
        .filter(|(g, r)| **g == Vector3::z() && **r != Vector3::z())
        .count();
    assert!(
        new_defaults == 0 && worst > 0.999,
        "{name} at scale 1e-6: {new_defaults} points fell back to (0, 0, 1); \
         worst |n . n_ref| = {worst}"
    );
}

#[test]
fn approx_cross_is_scale_invariant() {
    compare("approx_cross", compute_normals_approx_cross);
}

#[test]
fn approx_integral_is_scale_invariant() {
    compare("approx_integral", compute_normals_approx_integral);
}

#[test]
fn collinear_neighbours_give_the_default() {
    // Points on a line: every cross product is rounding noise, at any scale.
    for scale in [1.0f32, 1e-6] {
        let pts: Vec<Point3<f32>> = (0..10)
            .map(|i| Point3::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3) * scale)
            .collect();
        for (i, n) in compute_normals_approx_cross(&pts, 0.25 * scale)
            .iter()
            .enumerate()
        {
            assert_eq!(*n, Vector3::z(), "scale {scale} point {i}");
        }
    }
}
