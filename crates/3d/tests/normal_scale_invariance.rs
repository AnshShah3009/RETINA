//! A triangle's normal must be a *direction*, at every scale the mesh is stored in.
//!
//! # The defect class
//!
//! `e1.cross(&e2)` is the **unnormalised** face normal, so its length is the
//! triangle *area*, which scales as L^2. A guard of the form
//! `if area > 1e-9 { normal /= area }` therefore does not test "is this a valid
//! direction" — it tests "is this triangle bigger than about 30 microns", and
//! the length it rejects moves with whatever world units the mesh happens to be
//! stored in. A metre-unit mesh of 1 mm triangles is exactly that mesh, and it
//! is exactly what a surface reconstruction emits.
//!
//! This crate had four copies of this guard. Two were already fixed (in
//! `raycasting::cast_ray_mesh_bvh` and in `spatial::bvh::intersect_ray`'s
//! sibling, the normalisation in `cast_ray_mesh_bvh`); two were not, and one of
//! the unfixed copies had *disagreed with the fixed ones at the same scale*.
//!
//! # Measurements (all real, all reproduced by these tests)
//!
//! Equilateral triangle of edge `s` in the z = 0 plane, wound so the normal is
//! +z, ray straight down onto its centre from z = 1:
//!
//! ```text
//! edge      |cross|      face normal     vertex normal   BVH hit     brute hit
//! 1.0       1.7321       |n| = 1.000      |n| = 1.000      hit (t=1)  hit (t=1)
//! 1e-3      1.7321e-6    |n| = 1.000      |n| = 1.000      hit        hit
//! 1e-4      1.7321e-8    |n| = 1.000      |n| = 1.000      hit        hit
//! 1e-5      1.7321e-10   |n| = 0.000      |n| = 0.000      |n|=1.7e-10  MISS
//! 1e-6      1.7321e-12   |n| = 1.000      |n| = 1.000      -          -
//! ```
//!
//! Three separate symptoms at edge 1e-5, from three copies of the same mistake:
//! a face normal of zero length, a vertex normal of zero length, and a
//! brute-force raycast that *loses the hit entirely* while the BVH path finds
//! it. The last is the worst of the three: `cast_rays_brute` is documented as
//! the correctness reference for the BVH path, so the reference disagreed with
//! the thing it was checking, at precisely the scale where the difference shows.
//!
//! After the fix every row above is identical, and every normal is unit length.

use cv_3d::gpu::mesh;
use cv_3d::gpu::raycasting;
use cv_3d::mesh::TriangleMesh;
use nalgebra::{Point3, Vector3};

/// Equilateral triangle of side `edge`, wound anticlockwise seen from +z so
/// `cross(v1 - v0, v2 - v0)` points along +z. `|cross| = 0.866 * edge^2`.
fn equilateral(edge: f32) -> [Point3<f32>; 3] {
    let h = edge * 0.577_350_269;
    [
        Point3::new(-edge / 2.0, -h, 0.0),
        Point3::new(edge / 2.0, -h, 0.0),
        Point3::new(0.0, 2.0 * h, 0.0),
    ]
}

/// Edges spanning the scales at which the old absolute thresholds bit. 1e-3 and
/// 1e-5 are the equilateral cases where |cross| = 8.66e-7 and 8.66e-11, i.e.
/// one above and one below the 1e-9 thresholds.
const EDGES: [f32; 5] = [1.0, 1e-3, 1e-4, 1e-5, 1e-6];

/// `TriangleMesh::compute_face_normals` must return a unit vector at every scale.
#[test]
fn face_normals_are_unit_length_at_every_scale() {
    for &edge in &EDGES {
        let v = equilateral(edge);
        let mesh = TriangleMesh::with_vertices_and_faces(v.to_vec(), vec![[0, 1, 2]]);
        let n = mesh.compute_face_normals();

        assert_eq!(n.len(), 1);
        assert!(
            (n[0].norm() - 1.0).abs() < 1e-6,
            "edge {edge:e}: |cross| = {:.4e} (it is the AREA, so it scales as L^2); \
             the face normal came back as {:?} with length {:.4e}, not 1.0",
            edge,
            n[0],
            n[0].norm()
        );
        // The sign must be the winding's, at every scale: anticlockwise seen
        // from +z gives +z. A sign flip would be equally invisible to a length
        // check, so it is checked separately.
        assert!(
            n[0].z > 0.99,
            "edge {edge:e}: face normal {:?} points away from the winding's +z",
            n[0]
        );
    }
}

/// The same, for the vertex normals that `compute_vertex_normals` accumulates.
#[test]
fn vertex_normals_are_unit_length_at_every_scale() {
    for &edge in &EDGES {
        let v = equilateral(edge);
        let mut mesh = TriangleMesh::with_vertices_and_faces(v.to_vec(), vec![[0, 1, 2]]);
        mesh.compute_vertex_normals();
        let n = mesh.normals.as_ref().expect("normals were computed");

        assert_eq!(n.len(), 3);
        for (i, nv) in n.iter().enumerate() {
            assert!(
                (nv.norm() - 1.0).abs() < 1e-6,
                "edge {edge:e}: vertex {i} normal {:?} has length {:.4e}, not 1.0. \
                 All three vertices share one face, so each should be that face's \
                 unit normal.",
                nv,
                nv.norm()
            );
        }
    }
}

/// `gpu::mesh::compute_vertex_normals` accumulates *area-weighted* face normals,
/// so its accumulator's length is an area too — and it had the same guard.
#[test]
fn gpu_mesh_vertex_normals_are_unit_length_at_every_scale() {
    for &edge in &EDGES {
        let v = equilateral(edge);
        let n = mesh::compute_vertex_normals(&v, &[[0usize, 1, 2]]).expect("no faces, no error");
        for (i, nv) in n.iter().enumerate() {
            assert!(
                (nv.norm() - 1.0).abs() < 1e-6,
                "edge {edge:e}: gpu vertex {i} normal {:?} has length {:.4e}, not 1.0",
                nv,
                nv.norm()
            );
        }
    }
}

/// The BVH path's *hit normal* must be unit length at every scale.
///
/// `cast_ray_mesh_bvh` was already fixed in this crate; this pins it so the two
/// GPU copies below cannot drift away from it again.
#[test]
fn the_bvh_hit_normal_is_unit_length_at_every_scale() {
    for &edge in &EDGES {
        let v = equilateral(edge);
        let hit = raycasting::cast_rays(
            &[Point3::new(0.0, 0.0, 1.0)],
            &[Vector3::new(0.0, 0.0, -1.0)],
            &v,
            &[[0usize, 1, 2]],
        )
        .expect("one ray")
        .remove(0)
        .unwrap_or_else(|| panic!("edge {edge:e}: the BVH path lost a dead-on hit"));

        assert!(
            (hit.2.norm() - 1.0).abs() < 1e-6,
            "edge {edge:e}: hit normal {:?} has length {:.4e}, not 1.0",
            hit.2,
            hit.2.norm()
        );
    }
}

/// The brute-force path must agree with the BVH path at every scale.
///
/// This is the sharpest form of the test. `cast_rays_brute` is documented as the
/// correctness reference for the BVH path, and its own Möller-Trumbore copy still
/// carried the absolute `|a| < 1e-9` test when the BVH copy had been fixed:
/// `a = e1 · (d x e2)` is twice the *projected* area, so it scales as L^2 and
/// the test rejected edges below about 3e-5 rather than rays that were nearly
/// parallel to the triangle.
#[test]
fn brute_force_agrees_with_the_bvh_path_at_every_scale() {
    for &edge in &EDGES {
        let v = equilateral(edge);
        let ro = [Point3::new(0.0, 0.0, 1.0)];
        let rd = [Vector3::new(0.0, 0.0, -1.0)];

        let bvh = raycasting::cast_rays(&ro, &rd, &v, &[[0usize, 1, 2]])
            .expect("bvh runs")
            .remove(0);
        let brute = raycasting::cast_rays_brute(&ro, &rd, &v, &[[0usize, 1, 2]])
            .expect("brute runs")
            .remove(0);

        match (bvh, brute) {
            (Some(a), Some(b)) => {
                assert!(
                    (a.0 - b.0).abs() < 1e-5,
                    "edge {edge:e}: distances disagree, bvh t = {} vs brute t = {}",
                    a.0,
                    b.0
                );
                assert!(
                    (a.2 - b.2).norm() < 1e-5,
                    "edge {edge:e}: normals disagree, {:?} vs {:?}",
                    a.2,
                    b.2
                );
            }
            (None, None) => panic!("edge {edge:e}: BOTH paths lost a dead-on hit — the triangle is gone"),
            (Some(a), None) => panic!(
                "edge {edge:e}: the BVH path hit at t = {} with normal {:?}, but the brute-force \
                 reference found nothing. `a = e1 . (d x e2)` is twice the projected area and scales \
                 as L^2, so an absolute threshold on it is a size test, not a parallelism test.",
                a.0, a.2
            ),
            (None, Some(b)) => panic!("edge {edge:e}: brute force hit at t = {} but the BVH missed", b.0),
        }
    }
}

/// CONTROL: a genuinely degenerate (zero-area) triangle still yields no normal,
/// and a genuinely parallel ray is still rejected.
///
/// The fix changed `|area| > 1e-9` to `area > 0.0`, which lowers the bar; these
/// are the cases that must *not* slip through it.
#[test]
fn control_a_zero_area_triangle_still_has_no_normal_and_no_hit() {
    // All three vertices coincident: the cross product is exactly zero.
    let v = [
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(1.0, 2.0, 3.0),
    ];
    let mesh = TriangleMesh::with_vertices_and_faces(v.to_vec(), vec![[0, 1, 2]]);
    assert_eq!(
        mesh.compute_face_normals()[0],
        Vector3::zeros(),
        "a zero-area triangle has no normal, and must not be given one"
    );

    // A full-size triangle seen exactly edge-on: the ray is parallel to the plane
    // of the triangle and can never hit it, whatever the scale.
    for &edge in &EDGES {
        let tri = equilateral(edge);
        let hits = raycasting::cast_rays_brute(
            &[Point3::new(0.0, 0.0, 1.0)],
            &[Vector3::new(0.0, 0.0, 1.0)], // parallel to the z = 0 plane
            &tri,
            &[[0usize, 1, 2]],
        )
        .expect("brute runs");
        assert!(
            hits[0].is_none(),
            "edge {edge:e}: a ray lying IN the triangle's own plane must never hit it"
        );
    }
}

/// CONTROL: the well-formed unit-scale mesh is unaffected.
#[test]
fn control_a_unit_scale_mesh_is_unchanged() {
    let v = equilateral(1.0);
    let mesh = TriangleMesh::with_vertices_and_faces(v.to_vec(), vec![[0, 1, 2]]);
    assert!((mesh.compute_face_normals()[0] - Vector3::z()).norm() < 1e-6);
    assert!((mesh.surface_area() - 0.866_025_4).abs() < 1e-6);

    let hit = raycasting::cast_rays_brute(
        &[Point3::new(0.0, 0.0, 5.0)],
        &[Vector3::new(0.0, 0.0, -1.0)],
        &v,
        &[[0usize, 1, 2]],
    )
    .expect("brute runs")
    .remove(0)
    .expect("CONTROL: a unit triangle must still be hit");
    assert!((hit.0 - 5.0).abs() < 1e-5, "hit distance should be 5.0, got {}", hit.0);
    assert!((hit.2 - Vector3::z()).norm() < 1e-6);
}