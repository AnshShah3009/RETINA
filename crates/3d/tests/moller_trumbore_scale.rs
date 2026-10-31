//! Scale invariance of the Möller-Trumbore "ray is parallel to the triangle"
//! guard.
//!
//! The guard rejects a triangle when `a = e1 · (dir × e2)` is small. `a` is
//! twice the projected triangle area: for a genuine hit it scales as **L²**,
//! the square of the edge length. Comparing it against a fixed absolute
//! constant is therefore not an angular tolerance at all — it is a test for
//! "edge length below some value", and that value moves with the world units
//! the mesh happens to be stored in.
//!
//! A point-cloud reconstruction in metres can perfectly legally produce 1 mm
//! triangles, and the guard rejected every one of them as "parallel to
//! triangle". These tests pin the property the guard is *supposed* to have: the
//! same normalised configuration, rescaled, must give the same answer.
//!
//! The control is the reason the guard exists at all: a genuinely degenerate
//! (zero-area) triangle must still be rejected at every scale, and a genuinely
//! parallel ray must still be rejected against a full-size triangle. A "fix"
//! that simply deleted the guard would pass the scale tests and fail these.

use cv_3d::mesh::TriangleMesh;
use cv_3d::raycasting::{cast_ray_mesh, point_inside_mesh, Ray};
use nalgebra::{Point3, Vector3};

/// Scales spanning five orders of magnitude. The two lower entries are the ones
/// that matter in practice: 1e-3 is a 1 mm triangle in a metre-unit mesh, and
/// 1e-5 is already at the edge of what a dense surface reconstruction emits.
const SCALES: [f32; 6] = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5];

/// `a = e1 · (dir × e2)`, computed exactly as both implementations compute it.
///
/// This is not a re-implementation under test — it is the quantity the
/// threshold is compared against, reproduced so the tests can report *how far*
/// a case is from the threshold rather than just that it failed.
fn moller_a(v0: Point3<f32>, v1: Point3<f32>, v2: Point3<f32>, dir: Vector3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    e1.dot(&dir.cross(&e2))
}

/// Equilateral triangle of edge `s` lying in the `z = 0` plane, apex up.
/// `|e1| = |e2| = s` and the included angle is 60°, so a head-on ray produces
/// `a = s²·sin(60°) ≈ 0.866·s²`.
fn equilateral_triangle(s: f32) -> TriangleMesh {
    let h = s * 3.0f32.sqrt() / 2.0; // s * sqrt(3) / 2
    TriangleMesh::with_vertices_and_faces(
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(s, 0.0, 0.0),
            Point3::new(s / 2.0, h, 0.0),
        ],
        vec![[0, 1, 2]],
    )
}

/// Closed cube of edge `s` centred on the origin, with outward-facing
/// triangles. Used to reach the brute-force path in `point_inside_mesh`,
/// which does **not** go through the BVH.
fn cube(s: f32) -> TriangleMesh {
    let h = s / 2.0;
    let v = vec![
        Point3::new(-h, -h, -h),
        Point3::new(h, -h, -h),
        Point3::new(h, h, -h),
        Point3::new(-h, h, -h),
        Point3::new(-h, -h, h),
        Point3::new(h, -h, h),
        Point3::new(h, h, h),
        Point3::new(-h, h, h),
    ];
    let f = vec![
        [0, 3, 2],
        [0, 2, 1], // -z
        [4, 5, 6],
        [4, 6, 7], // +z
        [0, 1, 5],
        [0, 5, 4], // -y
        [3, 7, 6],
        [3, 6, 2], // +y
        [0, 4, 7],
        [0, 7, 3], // -x
        [1, 2, 6],
        [1, 6, 5], // +x
    ];
    TriangleMesh::with_vertices_and_faces(v, f)
}

/// MEASUREMENT. `a` is quadratic in the edge length, which is why a fixed
/// absolute threshold cannot be an angular tolerance.
///
/// This test passes before and after the fix on purpose: it is the measurement
/// the other tests are read against. Run with `--nocapture` to see the table.
#[test]
fn the_parallel_test_quantity_is_quadratic_in_the_edge_length() {
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let reference = moller_a(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 3.0f32.sqrt() / 2.0, 0.0),
        dir,
    )
    .abs();

    println!("\n  scale    edge length   |a|             |a|/(0.866 s^2)");
    for &s in &SCALES {
        let tri = equilateral_triangle(s);
        let a = moller_a(tri.vertices[0], tri.vertices[1], tri.vertices[2], dir).abs();
        let predicted = reference * s * s;
        println!("  {:<8e} {:<13e} {:<15.6e} {:.4}", s, s, a, a / predicted);
        assert!(
            (a - predicted).abs() <= 0.01 * predicted,
            "a is not quadratic in the edge length at s={s}: got {a}, expected ~{predicted}"
        );
        // The two absolute thresholds that were in the code:
        // 1e-6 in `raycasting::ray_triangle_intersection`,
        // 1e-9 in `spatial::bvh::moller_trumbore`.
        println!(
            "             -> rejected as 'parallel' by eps=1e-6: {:<5}  by eps=1e-9: {}",
            a < 1e-6,
            a < 1e-9
        );
    }
}

/// THE PROPERTY UNDER TEST. The same normalised ray against the same
/// normalised mesh must hit at every scale, and at the same *normalised*
/// distance `t / s`.
#[test]
fn the_same_normalised_ray_hits_the_same_mesh_at_every_scale() {
    for &s in &SCALES {
        let mesh = equilateral_triangle(s);
        // Normalised configuration, rescaled: origin one edge-length below the
        // triangle, straight up the z axis through an interior point.
        let ray = Ray::new(
            Point3::new(s * 0.25, s * 0.25, -s),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let a = moller_a(
            mesh.vertices[0],
            mesh.vertices[1],
            mesh.vertices[2],
            ray.direction,
        );

        let hit = cast_ray_mesh(&ray, &mesh).unwrap_or_else(|| {
            panic!(
                "MISS at scale {s:e} (edge length {s:e}): the same normalised ray \
                 hits the same normalised mesh at every scale. a = {a:.6e}, and \
                 |a| = 0.866 s^2, so a fixed |a| < 1e-6 / 1e-9 threshold is \
                 rejecting a triangle by its SIZE, not by its orientation."
            )
        });
        assert!(
            (hit.distance - s).abs() <= 1e-4 * s,
            "scale {s:e}: hit distance {} should be {s} (the ray is one edge length away)",
            hit.distance
        );
        assert!(
            (hit.normal.z - 1.0).abs() < 1e-5,
            "scale {s:e}: normal {:?} should be +z",
            hit.normal
        );
    }
}

/// The same property for an oblique ray, where `a` is reduced by the cosine of
/// the incidence angle on top of the L² factor. Measured per scale so the
/// report can quote it.
#[test]
fn an_oblique_ray_hits_the_same_mesh_at_every_scale() {
    // Pointing *along* -z toward the origin from below, so it crosses the
    // z = 0 plane at the same normalised point as the head-on case.
    let oblique = Vector3::new(-0.05, -0.02, 1.0).normalize();
    let expected = 1.0 / oblique.z; // distance to the z = 0 plane
    for &s in &SCALES {
        let mesh = equilateral_triangle(s);
        let ray = Ray::new(Point3::new(s * 0.25, s * 0.25, -s), oblique);
        let a = moller_a(
            mesh.vertices[0],
            mesh.vertices[1],
            mesh.vertices[2],
            ray.direction,
        );

        let hit = cast_ray_mesh(&ray, &mesh).unwrap_or_else(|| {
            panic!(
                "MISS at scale {s:e} from an oblique ray: a = {a:.6e}. The \
                 obliquity and the edge length are both encoded in `a`, so a \
                 fixed threshold couples two unrelated things."
            )
        });
        assert!(
            (hit.distance - s * expected).abs() <= 0.01 * s * expected,
            "scale {s:e}: oblique hit distance {} should be {:.6} s",
            hit.distance,
            expected
        );
    }
}

/// The brute-force path. `point_inside_mesh` does not use the BVH; it calls
/// `ray_triangle_intersection` in `raycasting/mod.rs` directly, so this reaches
/// the second copy of the guard.
///
/// The query point is deliberately **off-centre**. A ray cast from the exact
/// centre of a closed cube along +x runs *edge to edge* of the ±y and ±z
/// faces, and `a` is exactly zero on those — independent of scale, so it is a
/// separate numerical property and not this one.
#[test]
fn point_inside_a_closed_cube_is_scale_invariant() {
    for &s in &SCALES {
        let mesh = cube(s);
        assert!(
            point_inside_mesh(&Point3::new(s * 0.1, s * 0.2, s * 0.3), &mesh),
            "a point inside a closed cube of edge {s:e} is inside it, whatever units \
             the cube is expressed in"
        );
        assert!(
            !point_inside_mesh(&Point3::new(s, 0.0, 0.0), &mesh),
            "a point one edge length outside a cube of edge {s:e} is outside it"
        );
    }
}

/// CONTROL 1: a zero-area triangle. The guard's whole job is to reject these,
/// and the fix must keep doing so at every scale — including where the triangle
/// is *tiny*, which is exactly the case a sloppy normalised guard forgets.
#[test]
fn a_zero_area_triangle_is_rejected_at_every_scale() {
    for &s in &SCALES {
        // Collinear: three distinct points, no area.
        let collinear = TriangleMesh::with_vertices_and_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(s, 0.0, 0.0),
                Point3::new(2.0 * s, 0.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let ray = Ray::new(
            Point3::new(s * 0.25, s * 0.25, -s),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert!(
            cast_ray_mesh(&ray, &collinear).is_none(),
            "collinear (zero-area) triangle of size {s:e} must be rejected, not hit"
        );

        // Fully collapsed: a repeated vertex.
        let collapsed = TriangleMesh::with_vertices_and_faces(
            vec![
                Point3::new(s, s, 0.0),
                Point3::new(s, s, 0.0),
                Point3::new(s, s, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let ray = Ray::new(Point3::new(s, s, -s), Vector3::new(0.0, 0.0, 1.0));
        assert!(
            cast_ray_mesh(&ray, &collapsed).is_none(),
            "collapsed (zero-area) triangle of size {s:e} must be rejected, not hit"
        );
    }
}

/// CONTROL 2: the ordinary reason the guard exists — a ray that is actually
/// parallel to a perfectly good, full-size triangle must still miss, and a ray
/// that passes beside it must still miss.
#[test]
fn a_ray_parallel_to_a_full_size_triangle_is_still_rejected() {
    let mesh = equilateral_triangle(1.0);

    // In-plane: dir x e2 is in-plane and perpendicular to e1, so a == 0 exactly.
    let parallel = Ray::new(Point3::new(-1.0, 0.25, 0.0), Vector3::new(1.0, 0.0, 0.0));
    let a = moller_a(
        mesh.vertices[0],
        mesh.vertices[1],
        mesh.vertices[2],
        parallel.direction,
    );
    assert!(
        a.abs() == 0.0,
        "an exactly in-plane ray should give a == 0, got {a}"
    );
    assert!(
        cast_ray_mesh(&parallel, &mesh).is_none(),
        "a ray lying in the plane of the triangle is parallel to it and must miss"
    );

    // A ray that simply passes beside the triangle still misses, so the guard
    // was never the only thing doing the rejecting.
    let beside = Ray::new(Point3::new(5.0, 5.0, -1.0), Vector3::new(0.0, 0.0, 1.0));
    assert!(cast_ray_mesh(&beside, &mesh).is_none());
}

/// CONTROL 2b: the guard is a *relative* test, so the same ray against the
/// same normalised triangle gets the same verdict at every scale — and now for
/// a geometrically legible reason.
///
/// Note what `a` actually measures. `a = e1 · (dir × e2)` and `e2` is in-plane,
/// so `|a|/(|e1||e2|) = |dir · n|` with `n` the unit normal: the **normal**
/// component of the ray, the true sine of the incidence angle. An earlier
/// draft of this control tilted the ray's normal component instead, and
/// measured `|a|/denom` pinned at 0.866 all the way down to 1e-12 — the
/// `x`-component of `dir × e2` cancels it, so that configuration measures
/// nothing at all.
///
/// The ray below travels mostly in-plane with a small normal component, so the
/// ratio really is the sine of the angle, and the ray never crosses the plane.
/// Measured after the fix, identical at every scale:
///
/// ```text
///   normal   s=1.0        s=1e-3        s=1e-5        verdict
///   0.0      0.0          0.0           0.0           miss (in-plane)
///   1e-3     8.66e-4      8.66e-4       8.66e-4       miss
///   1e-6     8.66e-7      8.66e-7       8.66e-7       miss (past `raycasting`'s 1e-6)
///   1e-7     8.66e-8      8.66e-8       8.66e-8       miss (past the BVH's 1e-9)
/// ```
///
/// Before the fix the same ray's `|a|` was 8.66e-4, 8.66e-10 and 8.66e-14 at
/// those three scales, so the small copies were additionally rejected by the
/// absolute `1e-6` while the large one was not.
#[test]
fn a_nearly_parallel_ray_gets_the_same_verdict_at_every_scale() {
    for normal_component in [0.0_f32, 1e-3, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10] {
        let dir = Vector3::new(1.0, 0.0, normal_component).normalize();
        let mut verdicts = Vec::new();
        for &s in &SCALES {
            let mesh = equilateral_triangle(s);
            let ray = Ray::new(Point3::new(s * 0.25, s * 0.25, -s), dir);
            let a = moller_a(mesh.vertices[0], mesh.vertices[1], mesh.vertices[2], dir).abs();
            let denom = (mesh.vertices[1] - mesh.vertices[0]).norm()
                * (mesh.vertices[2] - mesh.vertices[0]).norm();
            verdicts.push(cast_ray_mesh(&ray, &mesh).is_some());
            // Relative comparison: both sides carry the same s^2, so the ratio
            // must agree to a few ulp no matter how small the scale gets.
            let ratio = a / denom;
            let expected = 0.866_025 * normal_component;
            assert!(
                (ratio - expected).abs() <= 1e-3 * expected.abs() + f32::MIN_POSITIVE,
                "s={s:e}, normal component {normal_component:e}: the guard's ratio \
                 |a|/(|e1||e2|) must be the sine of the incidence angle, and must \
                 not depend on the edge length (got {ratio:e}, expected {expected:e})"
            );
        }
        assert!(
            verdicts.iter().all(|v| !v),
            "a ray travelling {normal_component:e} off the plane has no forward \
             crossing of it; it must miss at every scale, got {verdicts:?}"
        );
    }
}

/// A ray tilted by a fixed angle off the *normal* axis — the ordinary case,
/// which does cross the plane — hits at the same normalised distance at every
/// scale.
#[test]
fn a_shallowly_tilted_ray_hits_at_the_same_normalised_distance_at_every_scale() {
    for tilt in [0.0_f32, 1e-2, 1e-4, 1e-6] {
        let dir = Vector3::new(tilt, 0.0, 1.0).normalize();
        let expected = 1.0 / dir.z; // distance to the z = 0 plane, in edge lengths
        for &s in &SCALES {
            let mesh = equilateral_triangle(s);
            let ray = Ray::new(Point3::new(s * 0.25, s * 0.25, -s), dir);
            let hit = cast_ray_mesh(&ray, &mesh).unwrap_or_else(|| {
                panic!(
                    "MISS at scale {s:e} for a ray tilted {tilt:e} off the normal; \
                     the same normalised ray hits the same normalised mesh at \
                     every scale"
                )
            });
            assert!(
                (hit.distance / s - expected).abs() <= 1e-3 * expected,
                "scale {s:e}, tilt {tilt:e}: normalised distance {} should be {expected}",
                hit.distance / s
            );
        }
    }
}

/// DOCUMENTED, DELIBERATELY NOT FIXED: the `t > eps` near-plane test.
///
/// `t` is a world-space *distance*, so `t > 1e-6` is a test in absolute units,
/// exactly the same class of problem as the `a` guard — and it survives the fix
/// below because it is a different, deliberate policy decision (it exists to
/// stop a ray from hitting geometry it is sitting on).
///
/// Measured here rather than changed: with the whole scene rescaled, `t = s`,
/// so at `s = 1e-6` the hit lands exactly on the near plane. This is recorded
/// as `#[ignore]`d rather than asserted, because the *desired* behaviour is that
/// it hits; asserting the current behaviour would pin the defect in place, and
/// asserting the desired behaviour would fail the suite. Run it with
/// `--ignored --nocapture` to see the measurement.
#[test]
#[ignore = "documents the measured near-plane scale dependence; not fixed here"]
fn the_near_plane_test_is_still_in_absolute_world_units() {
    for &s in &[1e-3_f32, 1e-5, 1e-6, 1e-7] {
        let mesh = equilateral_triangle(s);
        let ray = Ray::new(
            Point3::new(s * 0.25, s * 0.25, -s),
            Vector3::new(0.0, 0.0, 1.0),
        );
        match cast_ray_mesh(&ray, &mesh) {
            Some(hit) => println!("  s = {s:e}: hit at t = {:.6e} (t > 1e-6)", hit.distance),
            None => {
                println!("  s = {s:e}: MISS — t would have been {s:e}, below the 1e-6 near plane")
            }
        }
    }
}
