//! Mesh processing on an open mesh: Loop subdivision's boundary rule,
//! Laplacian smoothing's neighbour set, and edge-collapse's face budget.
//!
//! The mesh is a 4x4 grid of vertices on the height field
//! `z = 0.1 (x^2 + y^2)`, two triangles per cell - open, so it has boundary
//! vertices and corners, which is where all three defects lived.

use cv_3d::mesh::TriangleMesh;
use cv_3d::mesh::processing::{laplacian_smooth, loop_subdivision, simplify_edge_collapse};
use nalgebra::Point3;

fn grid() -> TriangleMesh {
    let mut v = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            let (x, y) = (i as f32, j as f32);
            v.push(Point3::new(x, y, 0.1 * (x * x + y * y)));
        }
    }
    let mut f = Vec::new();
    for j in 0..3 {
        for i in 0..3 {
            let a = j * 4 + i;
            let b = a + 1;
            let c = a + 4;
            let d = c + 1;
            f.push([a, b, d]);
            f.push([a, d, c]);
        }
    }
    TriangleMesh::with_vertices_and_faces(v, f)
}

fn close(a: &Point3<f32>, b: [f64; 3]) -> bool {
    (a.x as f64 - b[0]).abs() < 1e-5
        && (a.y as f64 - b[1]).abs() < 1e-5
        && (a.z as f64 - b[2]).abs() < 1e-5
}

/// Open3D 0.20.0 `TriangleMesh.subdivide_loop(1)` on the same mesh: the 16
/// original vertices, in order.
const OPEN3D_LOOP_OLD: [[f64; 3]; 16] = [
    [0.125, 0.125, 0.025],
    [1.0, 0.0, 0.125],
    [2.0, 0.0, 0.425],
    [2.875, 0.125, 0.85],
    [0.0, 1.0, 0.125],
    [1.0, 1.0, 0.25],
    [2.0, 1.0, 0.55],
    [3.0, 1.0, 1.025],
    [0.0, 2.0, 0.425],
    [1.0, 2.0, 0.55],
    [2.0, 2.0, 0.85],
    [3.0, 2.0, 1.325],
    [0.125, 2.875, 0.85],
    [1.0, 3.0, 1.025],
    [2.0, 3.0, 1.325],
    [2.875, 2.875, 1.675],
];

/// ...and the 33 edge vertices, as a set (the two libraries number them
/// differently).
const OPEN3D_LOOP_NEW: [[f64; 3]; 33] = [
    [0.0, 0.5, 0.05],
    [0.0, 1.5, 0.25],
    [0.0, 2.5, 0.65],
    [0.5, 0.0, 0.05],
    [0.5, 0.5, 0.1],
    [0.5, 1.0, 0.175],
    [0.5, 1.5, 0.3],
    [0.5, 2.0, 0.475],
    [0.5, 2.5, 0.7],
    [0.5, 3.0, 0.95],
    [1.0, 0.5, 0.175],
    [1.0, 1.5, 0.375],
    [1.0, 2.5, 0.775],
    [1.5, 0.0, 0.25],
    [1.5, 0.5, 0.3],
    [1.5, 1.0, 0.375],
    [1.5, 1.5, 0.5],
    [1.5, 2.0, 0.675],
    [1.5, 2.5, 0.9],
    [1.5, 3.0, 1.15],
    [2.0, 0.5, 0.475],
    [2.0, 1.5, 0.675],
    [2.0, 2.5, 1.075],
    [2.5, 0.0, 0.65],
    [2.5, 0.5, 0.7],
    [2.5, 1.0, 0.775],
    [2.5, 1.5, 0.9],
    [2.5, 2.0, 1.075],
    [2.5, 2.5, 1.3],
    [2.5, 3.0, 1.55],
    [3.0, 0.5, 0.95],
    [3.0, 1.5, 1.15],
    [3.0, 2.5, 1.55],
];

#[test]
fn loop_subdivision_matches_open3d_on_an_open_mesh() {
    let mut m = grid();
    loop_subdivision(&mut m);
    assert_eq!(m.vertices.len(), 49);
    assert_eq!(m.faces.len(), 72);
    for (i, want) in OPEN3D_LOOP_OLD.iter().enumerate() {
        assert!(
            close(&m.vertices[i], *want),
            "original vertex {i}: {:?}, Open3D {want:?}",
            m.vertices[i]
        );
    }
    for want in OPEN3D_LOOP_NEW.iter() {
        assert!(
            m.vertices[16..].iter().any(|v| close(v, *want)),
            "edge vertex {want:?} (Open3D) is missing"
        );
    }
}

#[test]
fn loop_subdivision_keeps_a_straight_border_straight() {
    // Every vertex of the y = 0 border stays on y = 0: the boundary mask
    // only mixes boundary neighbours, which all lie on that line.
    let mut m = grid();
    loop_subdivision(&mut m);
    for i in [1usize, 2] {
        assert!(
            m.vertices[i].y.abs() < 1e-6,
            "border vertex {i} left the border: {:?}",
            m.vertices[i]
        );
    }
}

#[test]
fn laplacian_is_the_uniform_average_of_distinct_neighbours() {
    let mut m = grid();
    laplacian_smooth(&mut m, 1, 0.5);

    // The same smoothing in the GPU module, which builds unique neighbours.
    let reference = grid();
    let mut v = reference.vertices.clone();
    cv_3d::gpu::mesh::laplacian_smooth(&mut v, &reference.faces, 1, 0.5).unwrap();

    // Corner (0, 0): neighbours (1, 0, 0.1), (1, 1, 0.2), (0, 1, 0.1), whose
    // mean is (2/3, 2/3, 0.4/3); half way there is (1/3, 1/3, 0.0667).
    assert!(
        close(&m.vertices[0], [1.0 / 3.0, 1.0 / 3.0, 0.2 / 3.0]),
        "corner moved to {:?}",
        m.vertices[0]
    );
    for (i, (a, b)) in m.vertices.iter().zip(&v).enumerate() {
        assert!(
            (a - b).norm() < 1e-6,
            "vertex {i}: processing {a:?} vs gpu::mesh {b:?}"
        );
    }
}

#[test]
fn edge_collapse_keeps_the_requested_fraction_of_faces() {
    for (ratio, want) in [(0.5f32, 9usize), (0.25, 4), (0.75, 13)] {
        let mut m = grid();
        simplify_edge_collapse(&mut m, ratio);
        assert_eq!(
            m.faces.len(),
            want,
            "target_ratio {ratio} of 18 faces left {}",
            m.faces.len()
        );
    }
}

#[test]
fn edge_collapse_is_reproducible() {
    let mut a = grid();
    let mut b = grid();
    simplify_edge_collapse(&mut a, 0.5);
    simplify_edge_collapse(&mut b, 0.5);
    assert_eq!(a.faces, b.faces);
    assert_eq!(a.vertices, b.vertices);
}
