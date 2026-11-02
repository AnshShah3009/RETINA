//! `taubin_smooth`'s `mu` follows Open3D's `filter_smooth_taubin` (and
//! Taubin 1995): the second pass is a Laplacian step with factor `mu`, and
//! `mu` is *negative* (Open3D's default `-0.53`) so that pass inflates.
//!
//! It used to apply `-mu`, the opposite convention, undocumented: Open3D's own
//! parameters (`lambda = 0.5, mu = -0.53`) made both passes shrink.
//!
//! Reference: Open3D 0.20.0 on the octahedron with vertices `+-e_i`. Every
//! vertex has four neighbours at equal distance, so Open3D's inverse-distance
//! Laplacian and this crate's uniform one coincide, and the neighbour centroid
//! is the origin: one iteration scales the mesh by `(1 - lambda)(1 - mu)`.
//! Open3D: vertex 0 goes to `(0.765, 0, 0)` after one iteration and to
//! `(0.44769713, 0, 0)` after three. Before the fix: `(0.235, 0, 0)`.

use cv_3d::mesh::TriangleMesh;
use cv_3d::mesh::processing::taubin_smooth;
use nalgebra::Point3;

fn octahedron() -> TriangleMesh {
    let v = vec![
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, -1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, -1.0),
    ];
    let f = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    TriangleMesh::with_vertices_and_faces(v, f)
}

#[test]
fn taubin_mu_is_open3ds_negative_inflation_factor() {
    for (iterations, open3d_x) in [(1usize, 0.765f32), (3, 0.447_697_13)] {
        let mut mesh = octahedron();
        taubin_smooth(&mut mesh, iterations, 0.5, -0.53);
        let v0 = mesh.vertices[0];
        assert!(
            (v0.x - open3d_x).abs() < 1e-6 && v0.y.abs() < 1e-6 && v0.z.abs() < 1e-6,
            "{iterations} iteration(s): vertex 0 at {v0:?}, Open3D ({open3d_x}, 0, 0)"
        );
    }
}
