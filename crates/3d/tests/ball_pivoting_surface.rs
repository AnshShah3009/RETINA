//! Ball pivoting reconstructs a surface, not a pile of overlapping triangles.
//!
//! A BPA triangle is one on which a ball of the given radius rests touching
//! its three vertices with **no other point inside**, on the side the normals
//! face. The implementation never checked emptiness and put the ball on
//! whichever side the vertex order gave, so on a jittered planar grid it
//! returned 845 triangles covering 19.27 m^2 of a 0.86 m^2 patch, half of
//! them facing down.
//!
//! References are Open3D 0.20.0's `create_from_point_cloud_ball_pivoting` on
//! the same clouds, with the same normals.

use cv_3d::mesh::TriangleMesh;
use cv_3d::mesh::reconstruction::{ball_pivoting, create_sphere_point_cloud};
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

/// A 10x10 grid at 0.1 spacing with a deterministic in-plane jitter, z = 0,
/// normals +z.
fn jittered_plane() -> PointCloud {
    let mut pts = Vec::new();
    for j in 0..10 {
        for i in 0..10 {
            let jitter = ((i * 7 + j * 13) % 5) as f32 * 0.01;
            pts.push(Point3::new(
                i as f32 * 0.1 + jitter,
                j as f32 * 0.1 - jitter,
                0.0,
            ));
        }
    }
    let n = pts.len();
    PointCloud {
        points: pts,
        normals: Some(vec![Vector3::new(0.0, 0.0, 1.0); n]),
        colors: None,
    }
}

/// Triangles per undirected edge.
fn edge_uses(m: &TriangleMesh) -> HashMap<(usize, usize), usize> {
    let mut edges = HashMap::new();
    for f in &m.faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    edges
}

#[test]
fn a_planar_patch_is_tiled_once_and_faces_up() {
    // (radius, Open3D triangle count, Open3D area). The convex hull of the
    // points has area 0.864, an upper bound for any non-overlapping tiling.
    for (radius, o3d_faces, o3d_area) in [
        (0.08f32, 162usize, 0.809f32),
        (0.1, 162, 0.809),
        (0.2, 176, 0.844),
    ] {
        let m = ball_pivoting(&jittered_plane(), radius);
        let area = m.surface_area();
        assert!(
            area <= 0.864 + 1e-4,
            "r {radius}: area {area} exceeds the hull's 0.864 - triangles overlap"
        );
        assert!(
            (area - o3d_area).abs() < 1e-4,
            "r {radius}: area {area}, Open3D {o3d_area}"
        );
        assert_eq!(m.faces.len(), o3d_faces, "r {radius}: triangle count");
        for (f, n) in m.faces.iter().zip(m.compute_face_normals()) {
            assert!(
                n.z > 0.99,
                "r {radius}: face {f:?} faces {n:?}, the normals face +z"
            );
        }
        assert!(
            edge_uses(&m).values().all(|&c| c <= 2),
            "r {radius}: an edge is shared by more than two triangles"
        );
    }
}

#[test]
fn a_sphere_closes_with_outward_triangles() {
    // 500 Fibonacci points on the unit sphere with radial normals. Open3D:
    // 996 triangles, area 12.4878, no boundary edge at radius 0.15 and 0.3;
    // 65 triangles at radius 0.1, which is too small to bridge the samples.
    let cloud = create_sphere_point_cloud(Point3::origin(), 1.0, 500);
    for (radius, o3d_faces, o3d_area) in [
        (0.15f32, 996usize, 12.487839f32),
        (0.3, 996, 12.487839),
        (0.1, 65, 0.764942),
    ] {
        let m = ball_pivoting(&cloud, radius);
        assert_eq!(m.faces.len(), o3d_faces, "r {radius}: triangle count");
        let area = m.surface_area();
        assert!(
            (area - o3d_area).abs() < 1e-3,
            "r {radius}: area {area}, Open3D {o3d_area}"
        );
        for (f, n) in m.faces.iter().zip(m.compute_face_normals()) {
            assert!(
                n.dot(&m.vertices[f[0]].coords) > 0.0,
                "r {radius}: face {f:?} points into the sphere"
            );
        }
        let uses = edge_uses(&m);
        assert!(uses.values().all(|&c| c <= 2));
        if o3d_faces == 996 {
            assert!(
                uses.values().all(|&c| c == 2),
                "r {radius}: the closed sphere has a boundary edge"
            );
        }
    }
}
