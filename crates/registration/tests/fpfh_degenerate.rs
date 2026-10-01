//! FPFH must not panic or fabricate descriptors on degenerate input.
//!
//! Both defects were found by auditing untested code:
//!
//! - `compute_fpfh_features(cloud, 0.0)` **panicked** with an integer overflow.
//!   `p.x / voxel_size` with a zero voxel size is +/-inf, `.floor() as i32`
//!   saturates to `i32::MAX`, and the neighbour loop's `vx + dx` overflows.
//! - A cloud too sparse for PCA normals produced **all-zero histograms** and
//!   reported success, which is worse than an error: a zero descriptor looks
//!   valid and carries no information.

use cv_registration::compute_fpfh_features;
use nalgebra::{Point3, Vector3};

fn cloud(points: &[(f32, f32, f32)]) -> cv_core::PointCloud {
    let mut pc = cv_core::PointCloud::default();
    for p in points {
        pc.points.push(Point3::new(p.0, p.1, p.2));
    }
    pc
}

fn dense_grid(n: usize, step: f32) -> cv_core::PointCloud {
    let mut pc = cv_core::PointCloud::default();
    let side = (n as f32).sqrt().ceil() as usize;
    for i in 0..n {
        let x = (i % side) as f32 * step;
        let y = (i / side) as f32 * step;
        pc.points.push(Point3::new(
            x,
            y,
            ((x * 3.0 + y * 7.0) % step.max(1e-3)) * 0.5,
        ));
    }
    pc
}

/// A zero radius must be refused, not used as a divisor.
#[test]
fn a_zero_radius_is_rejected_rather_than_panicking() {
    let pc = dense_grid(64, 0.05);
    let err = compute_fpfh_features(&pc, 0.0).expect_err(
        "radius 0 must be rejected: it divides by zero, which saturated an i32 \
         voxel index to i32::MAX and then overflowed on vx + dx",
    );
    assert!(
        err.to_string().contains("radius"),
        "the error should name the radius, got: {err}"
    );
}

#[test]
fn a_negative_or_non_finite_radius_is_rejected() {
    let pc = dense_grid(64, 0.05);
    for bad in [-0.1f32, f32::NAN, f32::INFINITY] {
        assert!(
            compute_fpfh_features(&pc, bad).is_err(),
            "radius {bad} must be rejected"
        );
    }
}

/// An empty cloud is fine and produces nothing.
#[test]
fn an_empty_cloud_is_not_an_error() {
    let pc = cv_core::PointCloud::default();
    assert!(compute_fpfh_features(&pc, 0.1)
        .expect("an empty cloud is not an error")
        .is_empty());
}

/// A cloud too sparse for PCA normals must not yield zero-information
/// descriptors reported as success.
///
/// Measured before the fix: 6 points spread 0.5 m apart at radius 0.05 gave 6
/// features, **0 of 6** with any non-zero bin, and `Ok`. Every histogram summed
/// to zero.
#[test]
fn a_cloud_too_sparse_for_normals_is_reported_not_fabricated() {
    // Six isolated points: every one has zero neighbours within 0.05.
    let pts = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.0),
    ];
    let pc = cloud(&pts);
    let res = compute_fpfh_features(&pc, 0.05);
    assert!(
        res.is_err(),
        "a cloud where no point has 3 neighbours cannot produce real FPFH \
         descriptors, and all-zero histograms must not be reported as success"
    );
}

/// With normals supplied, a sparse cloud is legitimate - the caller knows the
/// normals, so no estimation is needed.
#[test]
fn supplied_normals_make_a_sparse_cloud_usable() {
    let pts = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
    ];
    let mut pc = cloud(&pts);
    pc.normals = Some(vec![Vector3::z(); pts.len()]);

    let feats = compute_fpfh_features(&pc, 0.6).expect("with normals, this is well posed");
    assert_eq!(feats.len(), pts.len());
    assert!(
        feats.iter().all(|f| f.histogram.iter().sum::<f32>() > 0.0),
        "each descriptor should carry information when normals are supplied"
    );
}

/// A dense cloud with enough neighbours produces real, non-zero descriptors.
#[test]
fn a_dense_cloud_produces_informative_descriptors() {
    let pc = dense_grid(400, 0.05);
    let feats = compute_fpfh_features(&pc, 0.15).expect("a dense cloud is well posed");
    assert_eq!(feats.len(), pc.points.len());
    let informative = feats
        .iter()
        .filter(|f| f.histogram.iter().sum::<f32>() > 0.0)
        .count();
    assert!(
        informative > pc.points.len() / 2,
        "only {informative} of {} descriptors carry any information",
        pc.points.len()
    );
}
