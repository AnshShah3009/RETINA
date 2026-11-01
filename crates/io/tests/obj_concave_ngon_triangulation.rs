//! `ObjMesh::to_triangle_mesh` fan-triangulates every n-gon, which is only
//! correct for a **convex** polygon.
//!
//! OBJ faces are arbitrary planar polygons - many exporters emit an L- or
//! U-shaped face for a single planar region rather than splitting it - and the
//! fan is built from `face[0]`, so for a concave polygon the triangles it emits
//! cover area *outside* the polygon and the shape it describes is the convex
//! hull of the face, not the face.
//!
//! This is an inter-crate defect, which is why it is invisible to an OBJ
//! round-trip test: `ObjMesh::read` returns the n-gon correctly (its
//! `faces` are the face's own indices), and the vertices are untouched. The
//! geometry only goes wrong at the cv-io -> cv-3d boundary, when the caller
//! asks for the `TriangleMesh` that `cv_3d` then measures and ray-casts.
//!
//! The example is the U-shaped planar face on the left, area **10**:
//!
//! ```text
//!   (0,4) +-----------+ (4,4)
//!         |   +---+   |
//!         |   |   |   |      the notch is (1,1)-(3,4): 3 x 3 = 9 of empty
//!         |   |   |   |      space that is NOT part of the face
//!   (0,1) +---+   +---+ (4,1)
//!         |           |
//!   (0,0) +-----------+ (4,0)
//! ```
//!
//! Measured before the fix: `surface_area` 19 instead of 10, and a ray fired
//! straight down the middle of the notch at `(2, 2)` hit the "surface" at
//! distance 10 - a point where the polygon has no geometry at all.

use cv_io::ObjMesh;
use nalgebra::{Point3, Vector3};

/// The U-shaped concave face above, as `v` lines followed by one `f` line.
const U_FACE: &str = "\
v 0 0 0
v 4 0 0
v 4 4 0
v 3 4 0
v 3 1 0
v 1 1 0
v 1 4 0
v 0 4 0
f 1 2 3 4 5 6 7 8
";

/// Area of the U-shape: the 4x4 square minus the 2x3 notch.
const U_AREA: f32 = 16.0 - 6.0;

fn u_mesh() -> cv_3d::mesh::TriangleMesh {
    ObjMesh::read(std::io::Cursor::new(U_FACE))
        .expect("the fixture is a well-formed OBJ")
        .to_triangle_mesh()
}

/// The area the cv-3d mesh reports must be the polygon's own area.
#[test]
fn concave_obj_ngon_keeps_its_area_in_cv_3d() {
    let mesh = u_mesh();
    let area = mesh.surface_area();

    // CONTROL: the same 8 vertices as a closed convex 4x4 square do measure 16,
    // so the assertion below cannot pass by `surface_area` being broken.
    let convex = cv_3d::mesh::TriangleMesh::with_vertices_and_faces(
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(4.0, 4.0, 0.0),
            Point3::new(0.0, 4.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    );
    assert!(
        (convex.surface_area() - 16.0).abs() < 1e-4,
        "CONTROL: a 4x4 square must measure 16, got {}",
        convex.surface_area()
    );

    assert!(
        (area - U_AREA).abs() < 1e-4,
        "the U-shaped face has area {U_AREA}, but the mesh cv-3d measures says {area}: \
         fan triangulation covered the 2x3 notch, so the loaded geometry is the convex \
         hull of the face rather than the face"
    );
}

/// A ray down through the notch must miss. Before the fix it hit the fan
/// triangle that spans the notch.
#[test]
fn a_ray_through_the_notch_does_not_hit_the_concave_face() {
    let mesh = u_mesh();

    for (x, y) in [(2.0f32, 2.0f32), (2.0, 3.0), (1.5, 3.5)] {
        let ray = cv_3d::Ray::new(Point3::new(x, y, 10.0), Vector3::new(0.0, 0.0, -1.0));
        assert!(
            cv_3d::cast_ray_mesh(&ray, &mesh).is_none(),
            "({x}, {y}) lies in the notch, where the face has no surface, but a ray from \
             above hit it at {:?}: the triangulation filled the notch in",
            cv_3d::cast_ray_mesh(&ray, &mesh).map(|h| h.distance)
        );
    }

    // CONTROL: the three arms of the U must still be hit.
    for (x, y) in [(0.5f32, 2.0f32), (3.5, 2.0), (2.0, 0.5), (0.5, 3.8)] {
        let ray = cv_3d::Ray::new(Point3::new(x, y, 10.0), Vector3::new(0.0, 0.0, -1.0));
        let hit = cv_3d::cast_ray_mesh(&ray, &mesh)
            .unwrap_or_else(|| panic!("({x}, {y}) is on the face, so the ray must hit it"));
        assert!(
            (hit.distance - 10.0).abs() < 1e-4,
            "({x}, {y}): the face is the z = 0 plane, so the hit is 10 away, got {}",
            hit.distance
        );
    }

    // A point inside the notch is outside the (closed, planar) surface, and the
    // point-inside query must agree with the raycast above.
    assert!(
        !cv_3d::point_inside_mesh(&Point3::new(2.0, 2.0, 0.0), &mesh),
        "the notch centre is not part of the face, so it must not report inside"
    );
}

/// CONTROL: a convex n-gon and a plain triangle must be unchanged, so the fix
/// cannot be "stop triangulating".
#[test]
fn convex_ngons_and_triangles_still_triangulate() {
    // A planar convex hexagon (regular, radius 1, in z = 0), area 3*sqrt(3)/2.
    let mut obj = String::new();
    for i in 0..6 {
        let a = std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::TAU / 6.0;
        obj.push_str(&format!("v {} {} 0\n", a.cos(), a.sin()));
    }
    obj.push_str("f 1 2 3 4 5 6\n");

    let mesh = ObjMesh::read(std::io::Cursor::new(obj))
        .expect("convex hexagon")
        .to_triangle_mesh();
    assert_eq!(mesh.faces.len(), 4, "a hexagon fans into 4 triangles");
    let expected = 3.0 * 3.0f32.sqrt() / 2.0;
    assert!(
        (mesh.surface_area() - expected).abs() < 1e-4,
        "convex hexagon area should be {expected}, got {}",
        mesh.surface_area()
    );

    // A triangle is a triangle.
    let tri = ObjMesh::read(std::io::Cursor::new("v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n"))
        .expect("triangle")
        .to_triangle_mesh();
    assert_eq!(tri.faces.len(), 1);
    assert!((tri.surface_area() - 2.0).abs() < 1e-5);
}

/// A concave face wound the other way must triangulate to the same shape: the
/// ear-clipping passes must not depend on the file's winding direction.
#[test]
fn a_clockwise_concave_face_keeps_its_area_too() {
    // The same U, listed clockwise. Its own area is still 10.
    let clockwise = "\
v 0 0 0
v 0 4 0
v 1 4 0
v 1 1 0
v 3 1 0
v 3 4 0
v 4 4 0
v 4 0 0
f 1 2 3 4 5 6 7 8
";
    let mesh = ObjMesh::read(std::io::Cursor::new(clockwise))
        .expect("clockwise U")
        .to_triangle_mesh();
    assert!(
        (mesh.surface_area() - U_AREA).abs() < 1e-4,
        "the winding direction must not change the area; got {} for {U_AREA}",
        mesh.surface_area()
    );
}

/// Exporters routinely split a straight edge with an extra collinear vertex.
/// Such a corner turns by zero, so it is never an ear - the clipper must still
/// find ears around it rather than giving up and falling back to the fan.
#[test]
fn collinear_corners_do_not_defeat_the_clipper() {
    // The same U with a midpoint added to the bottom edge and to the left wall.
    let with_midpoints = "\
v 0 0 0
v 2 0 0
v 4 0 0
v 4 4 0
v 3 4 0
v 3 1 0
v 1 1 0
v 1 4 0
v 0 4 0
v 0 2 0
f 1 2 3 4 5 6 7 8 9 10
";
    let mesh = ObjMesh::read(std::io::Cursor::new(with_midpoints))
        .expect("U with midpoints")
        .to_triangle_mesh();
    assert_eq!(mesh.faces.len(), 8, "a 10-gon has 8 triangles");
    assert!(
        (mesh.surface_area() - U_AREA).abs() < 1e-4,
        "collinear corners must not change the area; got {} for {U_AREA}",
        mesh.surface_area()
    );
}

/// The same U, rotated out of the xy plane and scaled to millimetres. The
/// clipper projects onto the coordinate plane the face's normal faces least, so
/// it must not be fooled by an axis-aligned-looking face and must not be
/// scale-sensitive.
#[test]
fn a_concave_face_off_the_xy_plane_and_in_millimetres_keeps_its_area() {
    // Rotate the U about the x axis by 90 degrees (it now lies in the xz plane)
    // and shrink it by 1e-3.
    let corners = [
        (0.0f32, 0.0f32),
        (4.0, 0.0),
        (4.0, 4.0),
        (3.0, 4.0),
        (3.0, 1.0),
        (1.0, 1.0),
        (1.0, 4.0),
        (0.0, 4.0),
    ];
    let mut obj = String::new();
    for (x, y) in corners {
        obj.push_str(&format!("v {} 0 {}\n", x * 1e-3, y * 1e-3));
    }
    obj.push_str("f 1 2 3 4 5 6 7 8\n");

    let mesh = ObjMesh::read(std::io::Cursor::new(obj))
        .expect("rotated U")
        .to_triangle_mesh();
    let expected = U_AREA * 1e-6;
    assert!(
        (mesh.surface_area() - expected).abs() < 1e-10,
        "the area must scale with the face; expected {expected}, got {}",
        mesh.surface_area()
    );
}

/// The 2-D position of a triangle's centroid, for checking which triangles were
/// emitted without depending on the clip order.
fn area_of_triangle(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() * 0.5
}

/// Every emitted triangle of the U must lie inside the U. This is the property
/// the fan violated: its first triangle spanned the notch.
#[test]
fn every_triangle_of_a_concave_face_lies_inside_it() {
    let mesh = u_mesh();
    let inside = |p: (f32, f32)| -> bool {
        // The U is the 4x4 square minus the open notch (1,3) x (1,4).
        let in_square = p.0 >= 0.0 && p.0 <= 4.0 && p.1 >= 0.0 && p.1 <= 4.0;
        let in_notch = p.0 > 1.0 && p.0 < 3.0 && p.1 > 1.0 && p.1 <= 4.0;
        in_square && !in_notch
    };

    let total: f32 = mesh
        .faces
        .iter()
        .map(|f| {
            let p: Vec<(f32, f32)> = f
                .iter()
                .map(|&i| {
                    let v = mesh.vertices[i];
                    (v.x, v.y)
                })
                .collect();
            // The centroid of a triangle inside the polygon is inside it.
            let centroid = (
                (p[0].0 + p[1].0 + p[2].0) / 3.0,
                (p[0].1 + p[1].1 + p[2].1) / 3.0,
            );
            assert!(
                inside(centroid),
                "triangle {:?} of the U has centroid {:?}, which is in the notch",
                f,
                centroid
            );
            area_of_triangle(p[0], p[1], p[2])
        })
        .sum();

    assert!(
        (total - U_AREA).abs() < 1e-4,
        "the triangles must tile the U exactly; their total area is {total}, expected {U_AREA}"
    );
}
