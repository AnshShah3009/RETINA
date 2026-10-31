//! `Aabb` must represent emptiness, not invert its bounds.
//!
//! `Aabb::empty()` seeded `max` with `f32::MIN`, which in Rust is the most
//! *negative* finite f32, not the smallest positive one. An Aabb that never
//! received a point therefore reported
//!
//! ```text
//! min = (3.403e38, 3.403e38, 3.403e38)
//! max = (-3.403e38, -3.403e38, -3.403e38)
//! ```
//!
//! with min greater than max on every axis, and nothing distinguished that from
//! a real box. Measured consequences:
//!
//! - `Aabb::from_points(&[])` returned exactly that.
//! - `longest_axis()` computed `max - min = -6.8e38`, which **overflows to -inf**
//!   in f32. Both `d.x > d.y` and `d.y > d.z` were then false, so it returned
//!   axis 2 for *every* input — including meshes lying entirely in x. In
//!   `build_recursive` a degenerate face (a repeated vertex index, so all three
//!   vertices coincide) therefore silently picked the split axis of the subtree.
//! - `intersect_ray` returned `true` for an empty Aabb from **every** origin and
//!   direction: the overflow made `tmin` and `tmax` non-finite, so the final
//!   comparison was decided by NaN ordering rather than geometry. In a traversal
//!   that is a subtree reported as hit for rays passing nowhere near it.
//!
//! Emptiness is now an explicit field, so an empty Aabb is recognisable instead
//! of being inferred from a corrupt range.

use cv_3d::spatial::bvh::{Aabb, Bvh};
use nalgebra::Point3;

/// Every non-empty Aabb must have min <= max on every axis.
fn assert_valid(b: &Aabb, what: &str) {
    assert!(
        b.is_empty || (b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z),
        "{what}: inverted bounds min={:?} max={:?}",
        b.min,
        b.max
    );
}

#[test]
fn an_empty_aabb_is_marked_empty_rather_than_inverted() {
    assert!(Aabb::empty().is_empty);
}

#[test]
fn from_points_on_an_empty_slice_is_empty() {
    assert!(
        Aabb::from_points(&[]).is_empty,
        "from_points(&[]) must be empty, not a box with min > max"
    );
}

#[test]
fn a_single_point_produces_a_valid_degenerate_box() {
    let b = Aabb::from_points(&[Point3::new(1.0, 2.0, 3.0)]);
    assert!(!b.is_empty);
    assert_valid(&b, "single point");
    assert_eq!(b.min, Point3::new(1.0, 2.0, 3.0));
    assert_eq!(b.max, Point3::new(1.0, 2.0, 3.0));
}

#[test]
fn an_all_negative_cloud_produces_a_valid_box() {
    let b = Aabb::from_points(&[
        Point3::new(-10.0, -20.0, -30.0),
        Point3::new(-1.0, -2.0, -3.0),
    ]);
    assert_valid(&b, "all negative");
    assert_eq!(b.min, Point3::new(-10.0, -20.0, -30.0));
    assert_eq!(b.max, Point3::new(-1.0, -2.0, -3.0));
}

#[test]
fn merging_an_empty_aabb_leaves_the_other_untouched() {
    let real = Aabb::from_points(&[Point3::new(4.0, 5.0, 6.0)]);

    assert_eq!(
        Aabb::empty().merge(&real),
        real,
        "empty.merge(real) must be real, not a box carrying the empty sentinel"
    );
    assert_eq!(
        real.merge(&Aabb::empty()),
        real,
        "real.merge(empty) must be real"
    );
    assert!(
        Aabb::empty().merge(&Aabb::empty()).is_empty,
        "empty.merge(empty) must stay empty"
    );
}

#[test]
fn expand_point_clears_the_empty_flag() {
    let mut b = Aabb::empty();
    b.expand_point(&Point3::new(7.0, 8.0, 9.0));
    assert!(!b.is_empty, "expanding must mark the box non-empty");
    assert_valid(&b, "after expand_point");
    assert_eq!(b.min, Point3::new(7.0, 8.0, 9.0));
}

/// The pre-fix behaviour: `true` from every origin and direction, because the
/// sentinel bounds overflowed to non-finite `tmin`/`tmax`.
#[test]
fn an_empty_aabb_is_hit_by_no_ray() {
    let e = Aabb::empty();
    let inf = f32::INFINITY;

    for (label, origin, inv) in [
        (
            "(-1,0,0) +x",
            Point3::new(-1.0, 0.0, 0.0),
            nalgebra::Vector3::new(1.0, inf, inf),
        ),
        (
            "(0,0,0) +x",
            Point3::new(0.0, 0.0, 0.0),
            nalgebra::Vector3::new(1.0, inf, inf),
        ),
        (
            "(5,5,5) +x",
            Point3::new(5.0, 5.0, 5.0),
            nalgebra::Vector3::new(1.0, inf, inf),
        ),
        (
            "(5,5,5) -x",
            Point3::new(5.0, 5.0, 5.0),
            nalgebra::Vector3::new(-1.0, inf, inf),
        ),
    ] {
        assert!(
            !e.intersect_ray(&origin, &inv),
            "an empty box must not report a hit for a ray from {label}; a \
             traversal descending into such a subtree walks boxes the ray never \
             approaches"
        );
    }
}

/// The control for the test above: the real box still answers correctly.
///
/// `intersect_ray` takes the *reciprocal* of the direction, so a ray along +x
/// passes `inv_dir = (1, inf, inf)`.
#[test]
fn a_real_aabb_still_reports_hits_and_misses() {
    let real = Aabb::from_points(&[Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)]);
    let inf = f32::INFINITY;
    let through = nalgebra::Vector3::new(1.0, inf, inf);
    let away = nalgebra::Vector3::new(-1.0, inf, inf);

    assert!(
        real.intersect_ray(&Point3::new(-1.0, 0.5, 0.5), &through),
        "a ray along +x through the box must hit it"
    );
    assert!(
        !real.intersect_ray(&Point3::new(-1.0, 0.5, 0.5), &away),
        "a ray pointing away from the box must miss it"
    );
    assert!(
        !real.intersect_ray(&Point3::new(-1.0, 9.0, 0.5), &through),
        "a ray offset outside the box must miss it"
    );
    assert!(
        real.intersect_ray(&Point3::new(0.5, 0.5, 0.5), &through),
        "a ray starting inside the box must hit it"
    );
}

/// A mesh containing a degenerate face - a repeated vertex index, so all three
/// of its vertices coincide - must still build and be queryable.
#[test]
fn a_bvh_over_a_mesh_with_a_degenerate_face_builds() {
    let verts: Vec<Point3<f32>> = (0..8)
        .map(|i| Point3::new(i as f32, (i % 2) as f32, (i % 4) as f32))
        .collect();
    let mut faces: Vec<[usize; 3]> = vec![[0, 0, 0]];
    for i in 0..7 {
        faces.push([i, i + 1, (i + 2) % 8]);
    }

    let bvh = Bvh::build(&verts, &faces);

    let dir = nalgebra::Vector3::new(1.0, 0.0, 0.0);
    let hit = bvh.intersect_ray(&Point3::new(-1.0, 0.5, 0.5), &dir, &verts, &faces);
    assert!(
        hit.is_some(),
        "a ray through the mesh must find a hit; with an inverted sentinel box \
         the traversal culls by geometry it does not have"
    );
}
