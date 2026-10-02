//! Bounding boxes computed from point data must be valid.
//!
//! Both LAS builders seeded the running maximum with `f64::MIN`, which in Rust is
//! the most *negative* finite f64, not the smallest positive one. So
//! `max.max(p.x)` could never rise above it, and an empty or fully-masked cloud
//! produced an inverted box:
//!
//! ```text
//! (1.797e308, 1.797e308, 1.797e308, -1.797e308, -1.797e308, -1.797e308)
//! ```
//!
//! with min greater than max on every axis.

#![cfg(feature = "las")]

use cv_core::PointCloud;
use nalgebra::Point3;

/// `min <= max` on every axis, for any input including degenerate ones.
fn assert_valid_bounds(b: (f64, f64, f64, f64, f64, f64), what: &str) {
    let (min_x, min_y, min_z, max_x, max_y, max_z) = b;
    assert!(
        min_x <= max_x && min_y <= max_y && min_z <= max_z,
        "{what}: inverted bounding box {b:?} - min must not exceed max on any axis"
    );
    assert!(
        [min_x, min_y, min_z, max_x, max_y, max_z]
            .iter()
            .all(|v| v.is_finite()),
        "{what}: bounding box {b:?} contains a non-finite value, which a writer \
         would have to special-case"
    );
}

fn cloud_of(points: &[(f32, f32, f32)]) -> PointCloud {
    PointCloud::new(
        points
            .iter()
            .map(|(x, y, z)| Point3::new(*x, *y, *z))
            .collect(),
    )
}

/// The reported bug: an empty cloud has no extent, and must still produce a
/// box a writer can emit.
#[test]
fn an_empty_cloud_produces_a_valid_box() {
    let cloud = cloud_of(&[]);
    let b = cv_io::point_cloud_to_las(&cloud).bounds;
    assert_valid_bounds(b, "point_cloud_to_las(empty)");
}

/// All-negative coordinates, where a max seeded at 0.0 would also stay wrong.
#[test]
fn an_all_negative_cloud_produces_a_valid_box() {
    let cloud = cloud_of(&[(-10.0, -20.0, -30.0), (-1.0, -2.0, -3.0)]);
    let b = cv_io::point_cloud_to_las(&cloud).bounds;
    assert_valid_bounds(b, "point_cloud_to_las(all-negative)");
    assert_eq!(b.0, -10.0, "min_x wrong");
    assert_eq!(b.3, -1.0, "max_x wrong");
    assert_eq!(b.1, -20.0, "min_y wrong");
    assert_eq!(b.4, -2.0, "max_y wrong");
    assert_eq!(b.2, -30.0, "min_z wrong");
    assert_eq!(b.5, -3.0, "max_z wrong");
}

/// A single point gives a degenerate but valid box, not an inverted one.
#[test]
fn a_single_point_produces_a_degenerate_valid_box() {
    let cloud = cloud_of(&[(5.0, 6.0, 7.0)]);
    let b = cv_io::point_cloud_to_las(&cloud).bounds;
    assert_valid_bounds(b, "point_cloud_to_las(single)");
    assert_eq!((b.0, b.1, b.2), (5.0, 6.0, 7.0));
    assert_eq!(
        (b.3, b.4, b.5),
        (5.0, 6.0, 7.0),
        "a single point is its own bound"
    );
}

/// The ordinary case still computes the true extent.
#[test]
fn a_mixed_cloud_computes_the_true_extent() {
    let cloud = cloud_of(&[(-3.0, 0.0, 1.0), (4.0, -9.0, 2.0), (0.0, 8.0, -5.0)]);
    let b = cv_io::point_cloud_to_las(&cloud).bounds;
    assert_valid_bounds(b, "point_cloud_to_las(mixed)");
    assert_eq!((b.0, b.1, b.2), (-3.0, -9.0, -5.0), "min wrong");
    assert_eq!((b.3, b.4, b.5), (4.0, 8.0, 2.0), "max wrong");
}

/// Filtering every point away is the second route to an empty result, and had
/// the same `f64::MIN` seed.
#[test]
fn a_fully_filtered_cloud_produces_a_valid_box() {
    let cloud = cloud_of(&[(-10.0, -20.0, -30.0), (10.0, 20.0, 30.0)]);
    let data = cv_io::point_cloud_to_las(&cloud);
    let mask = vec![false; 2];
    let filtered = cv_io::filter_by_mask(&data, &mask);
    assert_eq!(filtered.points.len(), 0, "mask kept everything");
    assert_valid_bounds(filtered.bounds, "filter_by_mask(all-false)");
}

/// A mask that keeps only negative points must still bound them correctly.
#[test]
fn filtering_to_negative_points_computes_the_true_extent() {
    let cloud = cloud_of(&[(-10.0, -20.0, -30.0), (10.0, 20.0, 30.0)]);
    let data = cv_io::point_cloud_to_las(&cloud);
    let mask = vec![true, false];
    let filtered = cv_io::filter_by_mask(&data, &mask);
    assert_valid_bounds(filtered.bounds, "filter_by_mask(negatives only)");
    assert_eq!(
        (filtered.bounds.0, filtered.bounds.1, filtered.bounds.2),
        (-10.0, -20.0, -30.0)
    );
    assert_eq!(
        (filtered.bounds.3, filtered.bounds.4, filtered.bounds.5),
        (-10.0, -20.0, -30.0)
    );
}
