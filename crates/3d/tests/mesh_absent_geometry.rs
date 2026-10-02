//! `closest_point_on_mesh` and `mesh_to_mesh_distance` must report absence of
//! geometry instead of fabricating a confident answer.
//!
//! # 2. `mesh_to_mesh_distance` returned a PERFECT 0.0 for an empty mesh
//!
//! `crates/3d/src/raycasting/mod.rs:343`
//! ```ignore
//! if source.vertices.is_empty() || target.vertices.is_empty() {
//!     return (0.0, 0.0);
//! }
//! ```
//! `(0.0, 0.0)` is exactly what two *identical, fully matching* meshes produce,
//! so "no geometry here" was indistinguishable from "perfect match".
//! Verified: `mesh_to_mesh_distance(&TriangleMesh::new(), &cube)` returned
//! `(0.0, 0.0)`.
//!
//! The NaN-avoidance intent was right (it replaced a `0/0` mean) but the
//! replacement was wrong. The convention already adopted by
//! `closest_point_on_mesh` in the same file — `f32::INFINITY` for the empty
//! case — is what is used now. The in-module test at `raycasting/mod.rs:446`
//! asserted only `is_finite()`, locking the wrong value in; it now asserts the
//! absence is distinguishable from a perfect match.
//!
//! # 3. `closest_point_on_mesh` fabricated triangle index 0 with no faces
//!
//! `crates/3d/src/raycasting/mod.rs:225` guarded only `mesh.vertices.is_empty()`.
//! For a mesh with vertices but `faces.len() == 0` the search loop never ran, so
//! the function returned `(vertices[0], dist_to_vertex_0, 0)` — a triangle index
//! naming a triangle that does not exist. A caller indexing `mesh.faces[tri]`
//! got a wrong face or panicked.

use cv_3d::mesh::TriangleMesh;
use cv_3d::raycasting::{closest_point_on_mesh, mesh_to_mesh_distance};
use nalgebra::Point3;

/// Axis-aligned unit cube centred on the origin: 8 vertices, 12 faces.
fn unit_cube() -> TriangleMesh {
    let v = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    let vertices = v.iter().map(|(x, y, z)| Point3::new(*x, *y, *z)).collect();
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    TriangleMesh::with_vertices_and_faces(vertices, faces)
}

/// Vertices but NO faces: the case that made `closest_point_on_mesh` return
/// triangle index 0 while `mesh.faces` was empty.
fn vertices_without_faces() -> TriangleMesh {
    TriangleMesh::with_vertices_and_faces(
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 5.0, 0.0),
            Point3::new(0.0, 0.0, 5.0),
        ],
        vec![],
    )
}

// ── Defect 2: empty-mesh distance must not look like a perfect match ──────────

/// The empty source must not report the same distance as an exact match.
#[test]
fn an_empty_source_mesh_does_not_report_a_perfect_match() {
    let empty = TriangleMesh::new();
    let cube = unit_cube();

    let (hausdorff, mean) = mesh_to_mesh_distance(&empty, &cube);

    // CONTROL: this is what a genuine perfect match produces.
    let (exact_hausdorff, exact_mean) = mesh_to_mesh_distance(&cube, &cube);

    assert!(
        !(hausdorff == 0.0 && mean == 0.0),
        "an empty source mesh reported the perfect-match value (0.0, 0.0), which is \
         exactly what two identical fully-overlapping meshes report; absence of \
         geometry is indistinguishable from an exact match"
    );
    assert!(
        hausdorff.is_infinite() && mean.is_infinite(),
        "no vertices means no distances are defined, so both components must be \
         INFINITY (the convention closest_point_on_mesh already uses), got \
         hausdorff={hausdorff} mean={mean}"
    );
    assert!(
        exact_hausdorff == 0.0 && exact_mean == 0.0,
        "CONTROL: a cube measured against itself must still report (0.0, 0.0), got \
         ({exact_hausdorff}, {exact_mean})"
    );
}

/// Symmetric case: an empty *target* mesh.
#[test]
fn an_empty_target_mesh_does_not_report_a_perfect_match() {
    let empty = TriangleMesh::new();
    let cube = unit_cube();

    let (hausdorff, mean) = mesh_to_mesh_distance(&cube, &empty);

    assert!(
        hausdorff.is_infinite() && mean.is_infinite(),
        "an empty target mesh must report absence, not (0.0, 0.0); got \
         hausdorff={hausdorff} mean={mean}"
    );
}

/// Both meshes empty.
#[test]
fn two_empty_meshes_report_absence_not_a_match() {
    let empty = TriangleMesh::new();
    let (hausdorff, mean) = mesh_to_mesh_distance(&empty, &empty);

    assert!(
        hausdorff.is_infinite() && mean.is_infinite(),
        "two empty meshes must report absence, not (0.0, 0.0); got \
         hausdorff={hausdorff} mean={mean}"
    );
}

/// CONTROL: real, distinct meshes still produce finite, positive, correct
/// distances. This is the case that already worked, so the fix cannot have
/// turned everything into INFINITY.
#[test]
fn real_distinct_meshes_still_report_a_finite_positive_distance() {
    let cube = unit_cube();

    // A cube of the same shape scaled 1.5x about the origin: surface points move
    // outwards by 0.5 on the faces and by 0.5*sqrt(3) at the corners.
    let mut bigger = unit_cube();
    for v in bigger.vertices.iter_mut() {
        *v = Point3::new(v.x * 1.5, v.y * 1.5, v.z * 1.5);
    }

    let (hausdorff, mean) = mesh_to_mesh_distance(&cube, &bigger);

    assert!(
        hausdorff.is_finite() && mean.is_finite(),
        "CONTROL: two real meshes must give finite distances; got hausdorff={hausdorff} \
         mean={mean}"
    );
    assert!(
        hausdorff > 0.0 && mean > 0.0,
        "CONTROL: a cube and a strictly larger cube are not coincident, so the \
         distance must be strictly positive; got hausdorff={hausdorff} mean={mean}"
    );
    // The mean forward distance: a vertex of the unit cube, (1,1,1), sits on the
    // +X/+Y/+Z face planes of the larger cube, each at x=1.5, so its nearest point
    // is 0.5 away along a single axis. Every vertex behaves that way, so the
    // mean is exactly 0.5 and the Hausdorff distance (the worst corner, where the
    // three faces meet) is 0.5 * sqrt(3).
    assert!(
        (mean - 0.5).abs() < 1e-5,
        "each cube vertex is 0.5 from the nearer of the three faces it touches, so the \
         mean forward distance must be ~0.5; got mean={mean}"
    );
    assert!(
        (hausdorff - 0.5 * 3.0_f32.sqrt()).abs() < 1e-4,
        "the worst case is a corner, 0.5 out along each axis: the Hausdorff distance \
         must be 0.5*sqrt(3) = {:.5}; got {hausdorff}",
        0.5 * 3.0_f32.sqrt()
    );
}

/// CONTROL: a mesh whose vertices exist but whose faces do not has no
/// *surface*, so distances measured against it are also undefined and must be
/// reported as absent — and crucially must not read as 0.0.
#[test]
fn a_mesh_with_vertices_but_no_faces_is_also_reported_as_absent() {
    let open = vertices_without_faces();
    let cube = unit_cube();

    let (hausdorff, mean) = mesh_to_mesh_distance(&cube, &open);

    assert!(
        !(hausdorff == 0.0 && mean == 0.0),
        "a mesh with no faces has no surface to measure against; reporting (0.0, 0.0) \
         would claim a perfect match, got hausdorff={hausdorff} mean={mean}"
    );
    assert!(
        hausdorff.is_infinite() && mean.is_infinite(),
        "expected absence via INFINITY; got hausdorff={hausdorff} mean={mean}"
    );
}

// ── Defect 3: no triangle index without faces ────────────────────────────────

/// A mesh with vertices but zero faces must not produce a usable triangle index.
#[test]
fn a_mesh_without_faces_does_not_report_a_triangle_index() {
    let open = vertices_without_faces();
    let query = Point3::new(1.0, 1.0, 1.0);

    let (point, dist, tri) = closest_point_on_mesh(&query, &open);

    assert!(
        open.faces.is_empty(),
        "fixture precondition: the mesh must have no faces"
    );
    assert!(
        !(tri < open.faces.len()),
        "returned triangle index {tri} but the mesh has {} faces: the loop over faces \
         never ran, so the function fabricated index 0 and a caller indexing \
         `mesh.faces[{tri}]` gets the wrong face or panics",
        open.faces.len()
    );
    assert!(
        dist.is_infinite(),
        "with no triangles there is no surface, so the distance must be INFINITY \
         (the empty-mesh convention), got {dist}"
    );
    assert!(
        point == query,
        "with no surface the query point itself is echoed back unchanged, got {point:?}"
    );
}

/// The fully empty mesh already reported INFINITY; it must keep doing so, and
/// must not report a valid index either.
#[test]
fn a_fully_empty_mesh_reports_no_triangle() {
    let empty = TriangleMesh::new();
    let query = Point3::new(1.0, 2.0, 3.0);

    let (point, dist, tri) = closest_point_on_mesh(&query, &empty);

    assert!(point == query);
    assert!(dist.is_infinite(), "expected INFINITY, got {dist}");
    assert!(
        !(tri < empty.faces.len()),
        "an empty mesh must not report triangle index {tri}"
    );
}

/// CONTROL: a real mesh with faces still returns a valid index and a finite
/// distance. Without this the assertions above could pass by breaking the
/// working path.
#[test]
fn a_real_mesh_still_returns_a_usable_triangle_index_and_distance() {
    let cube = unit_cube();

    // A point outside the cube, diagonally beyond the (1,1,1) corner.
    let query = Point3::new(2.0, 2.0, 2.0);
    let (point, dist, tri) = closest_point_on_mesh(&query, &cube);

    assert!(
        tri < cube.faces.len(),
        "CONTROL: a real mesh must return an index into its faces; got {tri} of {}",
        cube.faces.len()
    );
    assert!(
        dist.is_finite() && dist > 0.0,
        "CONTROL: distance to a real mesh must be finite and positive; got {dist}"
    );
    // The closest point on the cube is the corner (1,1,1); distance sqrt(3).
    assert!(
        (dist - 3.0_f32.sqrt()).abs() < 1e-5,
        "CONTROL: the closest point on a unit cube to (2,2,2) is the corner (1,1,1) at \
         distance sqrt(3); got {dist} at {point:?}"
    );
    assert!((point - Point3::new(1.0, 1.0, 1.0)).norm() < 1e-5);
}

/// CONTROL: a point inside the cube is 1.0 from the nearest face, and the
/// index must name a face that actually contains that closest point.
#[test]
fn the_reported_face_actually_owns_the_closest_point() {
    let cube = unit_cube();
    let query = Point3::new(0.0, 0.0, 0.0);

    let (point, dist, tri) = closest_point_on_mesh(&query, &cube);

    assert!(tri < cube.faces.len(), "index {tri} out of range");
    assert!(
        (dist - 1.0).abs() < 1e-5,
        "CONTROL: the origin is 1.0 from every face of a unit cube; got {dist}"
    );
    assert!(
        (point.z - 1.0).abs() < 1e-5 || point.z == -1.0,
        "CONTROL: the closest point must lie on the cube's surface; got {point:?}"
    );
}
