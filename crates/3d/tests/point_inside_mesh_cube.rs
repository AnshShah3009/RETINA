//! Inside/outside on a 12-triangle unit cube. A single +x ray from the cube's
//! centre runs through the shared diagonal of a face, which the inclusive
//! ray/triangle test counted twice: the centre came back *outside*.

use cv_3d::mesh::TriangleMesh;
use cv_3d::raycasting::point_inside_mesh;
use nalgebra::Point3;

fn unit_cube() -> TriangleMesh {
    let v = |x: f32, y: f32, z: f32| Point3::new(x, y, z);
    let vertices = vec![
        v(0., 0., 0.),
        v(1., 0., 0.),
        v(1., 1., 0.),
        v(0., 1., 0.),
        v(0., 0., 1.),
        v(1., 0., 1.),
        v(1., 1., 1.),
        v(0., 1., 1.),
    ];
    let faces = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];
    let mut mesh = TriangleMesh::new();
    mesh.vertices = vertices;
    mesh.faces = faces;
    mesh
}

#[test]
fn cube_inside_and_outside() {
    let cube = unit_cube();
    for p in [
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.9, 0.1, 0.5],
        [0.5, 0.5, 0.01],
    ] {
        assert!(
            point_inside_mesh(&Point3::new(p[0], p[1], p[2]), &cube),
            "{p:?} should be inside"
        );
    }
    for p in [
        [1.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 2.0],
    ] {
        assert!(
            !point_inside_mesh(&Point3::new(p[0], p[1], p[2]), &cube),
            "{p:?} should be outside"
        );
    }
}
