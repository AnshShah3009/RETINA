//! A non-finite voxel size is not `<= 0`, and `(NaN).floor() as i64` is 0, so a
//! NaN size used to run and a NaN point used to share voxel `(0, 0, 0)` with a
//! finite point. The centroid of that voxel came back NaN.

use cv_3d::filters::voxel_downsample;
use nalgebra::{Point3, Vector3};

fn finite_pair() -> Vec<Point3<f64>> {
    vec![Point3::new(0.25, 0.25, 0.25), Point3::new(1.25, 0.25, 0.25)]
}

#[test]
fn a_non_finite_voxel_size_returns_empty() {
    let points = finite_pair();
    let normals = vec![Vector3::x(), Vector3::y()];
    let colors = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)];
    for size in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        let result = voxel_downsample(&points, Some(&normals), Some(&colors), size);
        assert!(
            result.points.is_empty(),
            "size {size} must return no points, got {:?}",
            result.points
        );
        assert!(result.normals.as_ref().unwrap().is_empty());
        assert!(result.colors.as_ref().unwrap().is_empty());
    }
}

#[test]
fn a_non_finite_point_does_not_poison_the_centroid() {
    let finite = Point3::new(0.2, 0.3, 0.4);
    let points = vec![
        finite,
        Point3::new(f64::NAN, 0.0, 0.0),
        Point3::new(0.0, f64::INFINITY, 0.0),
        Point3::new(0.0, 0.0, f64::NEG_INFINITY),
    ];
    let result = voxel_downsample(&points, None, None, 1.0);
    assert_eq!(
        result.points.len(),
        1,
        "only the finite point occupies a voxel"
    );
    let p = result.points[0];
    assert!(
        (p - finite).norm() < 1e-12,
        "centroid {p:?} must be the finite point, not a NaN mixture"
    );
}

#[test]
fn a_finite_cloud_is_unchanged() {
    let points = finite_pair();
    let result = voxel_downsample(&points, None, None, 1.0);
    assert_eq!(result.points.len(), 2);
    // Keys sort by voxel index, so the point in voxel (0,0,0) comes first.
    assert!((result.points[0] - points[0]).norm() < 1e-12);
    assert!((result.points[1] - points[1]).norm() < 1e-12);
}
