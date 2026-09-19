//! Mesh Processing Operations
//!
//! Implements:
//! - Laplacian smoothing
//! - Edge collapse simplification
//! - Loop subdivision
//! - Vertex clustering

use super::TriangleMesh;
use nalgebra::Point3;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Laplacian smoothing: Move each vertex toward the average of its neighbors
pub fn laplacian_smooth(mesh: &mut TriangleMesh, iterations: usize, lambda: f32) {
    let num_vertices = mesh.vertices.len();

    for _ in 0..iterations {
        // Build adjacency list
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
        for face in &mesh.faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                neighbors[v0].push(v1);
                neighbors[v1].push(v0);
            }
        }

        // Compute new positions
        let mut new_positions = mesh.vertices.clone();

        new_positions
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, vertex)| {
                if neighbors[i].is_empty() {
                    return;
                }

                let mut centroid = Point3::origin();
                for &neighbor_idx in &neighbors[i] {
                    centroid += mesh.vertices[neighbor_idx].coords;
                }
                centroid /= neighbors[i].len() as f32;

                // Move vertex toward centroid
                let displacement = centroid - mesh.vertices[i];
                *vertex = mesh.vertices[i] + displacement * lambda;
            });

        mesh.vertices = new_positions;
    }

    // Recompute normals after smoothing
    mesh.compute_vertex_normals();
}

/// Taubin smoothing: Two-step Laplacian with shrinkage compensation
pub fn taubin_smooth(mesh: &mut TriangleMesh, iterations: usize, lambda: f32, mu: f32) {
    for _ in 0..iterations {
        // Forward Laplacian (shrink)
        laplacian_smooth(mesh, 1, lambda);
        // Backward Laplacian (inflate)
        laplacian_smooth(mesh, 1, -mu);
    }
}

/// Edge collapse simplification: Reduce triangle count
/// target_ratio: 0.0 to 1.0 (fraction of faces to keep)
pub fn simplify_edge_collapse(mesh: &mut TriangleMesh, target_ratio: f32) {
    let target_faces = (mesh.faces.len() as f32 * target_ratio.clamp(0.0, 1.0)) as usize;
    if target_faces >= mesh.faces.len() {
        return;
    }

    let num_faces_to_remove = mesh.faces.len() - target_faces;

    // Build edge set
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i].min(face[(i + 1) % 3]);
            let v1 = face[i].max(face[(i + 1) % 3]);
            edges.insert((v0, v1));
        }
    }

    // Simple approach: Collapse short edges
    let mut collapsed = 0;
    let mut vertex_remap: HashMap<usize, usize> =
        (0..mesh.vertices.len()).map(|i| (i, i)).collect();

    // Path-compressing resolve: follow remap chains to the final vertex
    let resolve = |remap: &HashMap<usize, usize>, mut v: usize| -> usize {
        while let Some(&next) = remap.get(&v) {
            if next == v {
                break;
            }
            v = next;
        }
        v
    };

    for (v0, v1) in edges {
        if collapsed >= num_faces_to_remove {
            break;
        }

        let rv0 = resolve(&vertex_remap, v0);
        let rv1 = resolve(&vertex_remap, v1);

        if rv0 == rv1 {
            continue;
        }

        // Collapse: move v1 to midpoint
        let midpoint = (mesh.vertices[rv0] + mesh.vertices[rv1].coords) * 0.5;
        mesh.vertices[rv0] = midpoint;
        vertex_remap.insert(rv1, rv0);

        collapsed += 1;
    }

    // Remap faces and remove degenerate
    let mut new_faces = Vec::new();
    for face in &mesh.faces {
        let new_face = [
            resolve(&vertex_remap, face[0]),
            resolve(&vertex_remap, face[1]),
            resolve(&vertex_remap, face[2]),
        ];

        // Keep only non-degenerate faces
        if new_face[0] != new_face[1] && new_face[1] != new_face[2] && new_face[2] != new_face[0] {
            new_faces.push(new_face);
        }
    }

    mesh.faces = new_faces;
    mesh.compute_vertex_normals();
}

/// Vertex clustering simplification using uniform grid
pub fn simplify_vertex_clustering(mesh: &mut TriangleMesh, voxel_size: f32) {
    if voxel_size <= 0.0 {
        return;
    }

    // Compute bounds
    let (min, _max) = mesh.bounds();

    // Cluster vertices
    let mut cluster_map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, v) in mesh.vertices.iter().enumerate() {
        let key = (
            ((v.x - min.x) / voxel_size) as i32,
            ((v.y - min.y) / voxel_size) as i32,
            ((v.z - min.z) / voxel_size) as i32,
        );
        cluster_map.entry(key).or_default().push(i);
    }

    // Create representative vertices for each cluster
    let mut cluster_vec: Vec<Vec<usize>> = cluster_map.into_values().collect();
    let mut vertex_remap: Vec<usize> = vec![0; mesh.vertices.len()];

    let new_vertices: Vec<Point3<f32>> = cluster_vec
        .par_iter_mut()
        .map(|cluster_indices| {
            let mut centroid = Point3::origin();
            for &idx in cluster_indices.iter() {
                centroid += mesh.vertices[idx].coords;
            }
            centroid /= cluster_indices.len() as f32;

            // We can't easily update vertex_remap in parallel without unsafe or a separate pass
            // So we'll do it in a separate pass or just accept this part being parallel.
            centroid
        })
        .collect();

    // Fill vertex_remap (serial pass is fast enough for remapping)
    for (new_idx, cluster_indices) in cluster_vec.iter().enumerate() {
        for &idx in cluster_indices {
            vertex_remap[idx] = new_idx;
        }
    }

    // Remap faces
    let mut new_faces: Vec<[usize; 3]> = Vec::new();
    let mut seen_faces: HashSet<[usize; 3]> = HashSet::new();

    for face in &mesh.faces {
        let new_face = [
            vertex_remap[face[0]],
            vertex_remap[face[1]],
            vertex_remap[face[2]],
        ];

        // Skip degenerate and duplicate faces
        if new_face[0] != new_face[1] && new_face[1] != new_face[2] && new_face[2] != new_face[0] {
            let sorted = {
                let mut s = new_face;
                s.sort_unstable();
                s
            };
            if seen_faces.insert(sorted) {
                new_faces.push(new_face);
            }
        }
    }

    mesh.vertices = new_vertices;
    mesh.faces = new_faces;
    mesh.compute_vertex_normals();
}

/// Loop subdivision for triangle meshes
pub fn loop_subdivision(mesh: &mut TriangleMesh) {
    let num_vertices = mesh.vertices.len();
    let num_faces = mesh.faces.len();

    // Precompute edge→opposite vertices map (O(f) instead of O(f²) per edge)
    let mut edge_opposites: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let opp = face[(i + 2) % 3];
            let edge = (v0.min(v1), v0.max(v1));
            edge_opposites.entry(edge).or_default().push(opp);
        }
    }

    // Precompute vertex→neighbors map (O(f) instead of O(v×f))
    let mut vertex_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];
    for face in &mesh.faces {
        for i in 0..3 {
            vertex_neighbors[face[i]].insert(face[(i + 1) % 3]);
            vertex_neighbors[face[i]].insert(face[(i + 2) % 3]);
        }
    }

    // Compute new edge vertices using precomputed adjacency
    let mut edge_vertices: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_vertices = mesh.vertices.clone();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = (v0.min(v1), v0.max(v1));

            if let std::collections::hash_map::Entry::Vacant(e) = edge_vertices.entry(edge) {
                let opposite_vertices = edge_opposites.get(&edge).cloned().unwrap_or_default();

                let new_pos = if opposite_vertices.len() == 2 {
                    let p0 = mesh.vertices[v0];
                    let p1 = mesh.vertices[v1];
                    let o0 = mesh.vertices[opposite_vertices[0]];
                    let o1 = mesh.vertices[opposite_vertices[1]];
                    Point3::from(
                        (p0.coords + p1.coords) * (3.0 / 8.0)
                            + (o0.coords + o1.coords) * (1.0 / 8.0),
                    )
                } else {
                    let p0 = mesh.vertices[v0];
                    let p1 = mesh.vertices[v1];
                    (p0 + p1.coords) * 0.5
                };

                let new_idx = new_vertices.len();
                new_vertices.push(new_pos);
                e.insert(new_idx);
            }
        }
    }

    // Update existing vertices using precomputed neighbors
    let mut smoothed_vertices = new_vertices.clone();

    for i in 0..num_vertices {
        let neighbors = &vertex_neighbors[i];
        if !neighbors.is_empty() {
            let n = neighbors.len() as f32;
            let beta = if n > 3.0 { 3.0 / (8.0 * n) } else { 3.0 / 16.0 };

            let mut new_pos = new_vertices[i].coords * (1.0 - n * beta);
            for &neighbor in neighbors {
                new_pos += new_vertices[neighbor].coords * beta;
            }
            smoothed_vertices[i] = Point3::from(new_pos);
        }
    }

    // Create new faces (each triangle becomes 4)
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(num_faces * 4);

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        let e01 = *edge_vertices.get(&(v0.min(v1), v0.max(v1))).unwrap();
        let e12 = *edge_vertices.get(&(v1.min(v2), v1.max(v2))).unwrap();
        let e20 = *edge_vertices.get(&(v2.min(v0), v2.max(v0))).unwrap();

        new_faces.push([v0, e01, e20]);
        new_faces.push([v1, e12, e01]);
        new_faces.push([v2, e20, e12]);
        new_faces.push([e01, e12, e20]);
    }

    mesh.vertices = smoothed_vertices;
    mesh.faces = new_faces;
    mesh.compute_vertex_normals();
}
