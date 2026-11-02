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
        // Build adjacency list: each neighbour once. An interior edge belongs to
        // two triangles, so pushing per face listed most neighbours twice and
        // boundary neighbours once, and the "average" became a weighted one -
        // measured on an open grid, the corner (0, 0) with neighbours (1, 0),
        // (1, 1), (0, 1) moved to (0.375, 0.375) at lambda 0.5 instead of the
        // uniform average's (0.333, 0.333), which `gpu::mesh::laplacian_smooth`
        // (unique neighbours) returns for the same mesh.
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
        for face in &mesh.faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                neighbors[v0].push(v1);
                neighbors[v1].push(v0);
            }
        }
        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
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
///
/// Edges are collapsed shortest first (to their midpoint) until the mesh has
/// `floor(faces * target_ratio)` faces; a collapse that would remove more faces
/// than are left to remove is skipped in favour of a later one.
pub fn simplify_edge_collapse(mesh: &mut TriangleMesh, target_ratio: f32) {
    let target_faces = (mesh.faces.len() as f32 * target_ratio.clamp(0.0, 1.0)) as usize;
    if target_faces >= mesh.faces.len() {
        return;
    }

    // Build edge set, shortest first. The edges used to be taken in `HashSet`
    // iteration order - neither "short edges first" nor reproducible from one
    // run to the next.
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i].min(face[(i + 1) % 3]);
            let v1 = face[i].max(face[(i + 1) % 3]);
            if v0 != v1 {
                edge_set.insert((v0, v1));
            }
        }
    }
    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let len_sq = |e: &(usize, usize)| (mesh.vertices[e.0] - mesh.vertices[e.1]).norm_squared();
    edges.sort_by(|a, b| len_sq(a).total_cmp(&len_sq(b)).then(a.cmp(b)));

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

    // Faces still standing, and the faces around each surviving vertex.
    //
    // The stop test used to compare the number of *edges* collapsed with the
    // number of *faces* to remove. An interior collapse removes two faces, so
    // the mesh overshot: measured on an 18-face grid, `target_ratio = 0.5`
    // left 3 faces instead of 9.
    let mut alive = vec![true; mesh.faces.len()];
    let mut alive_count = mesh.faces.iter().filter(|f| !is_degenerate(f)).count();
    for (fi, f) in mesh.faces.iter().enumerate() {
        if is_degenerate(f) {
            alive[fi] = false;
        }
    }
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); mesh.vertices.len()];
    for (fi, f) in mesh.faces.iter().enumerate() {
        if alive[fi] {
            for &v in f {
                if !incident[v].contains(&fi) {
                    incident[v].push(fi);
                }
            }
        }
    }

    for (v0, v1) in edges {
        if alive_count <= target_faces {
            break;
        }

        let rv0 = resolve(&vertex_remap, v0);
        let rv1 = resolve(&vertex_remap, v1);

        if rv0 == rv1 {
            continue;
        }

        // The faces this collapse destroys: those around `rv1` that also
        // contain `rv0`.
        let doomed: Vec<usize> = incident[rv1]
            .iter()
            .copied()
            .filter(|&fi| {
                alive[fi]
                    && mesh.faces[fi]
                        .iter()
                        .any(|&v| resolve(&vertex_remap, v) == rv0)
            })
            .collect();
        if alive_count - doomed.len() < target_faces {
            continue;
        }

        // Collapse: move rv0 to the midpoint and merge rv1 into it.
        let midpoint = (mesh.vertices[rv0] + mesh.vertices[rv1].coords) * 0.5;
        mesh.vertices[rv0] = midpoint;
        vertex_remap.insert(rv1, rv0);

        for fi in doomed {
            alive[fi] = false;
            alive_count -= 1;
        }
        let moved = std::mem::take(&mut incident[rv1]);
        for fi in moved {
            if alive[fi] && !incident[rv0].contains(&fi) {
                incident[rv0].push(fi);
            }
        }
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
        if !is_degenerate(&new_face) {
            new_faces.push(new_face);
        }
    }

    mesh.faces = new_faces;
    mesh.compute_vertex_normals();
}

fn is_degenerate(f: &[usize; 3]) -> bool {
    f[0] == f[1] || f[1] == f[2] || f[2] == f[0]
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

    // A vertex on the boundary is moved along the boundary only: Loop's
    // boundary mask is `3/4 v + 1/8 (b0 + b1)` over its two neighbours across
    // boundary edges (edges with a single triangle), which is also what Open3D's
    // `subdivide_loop` does. The interior mask, applied to a boundary vertex,
    // pulls it off the boundary curve towards the interior: measured on an open
    // 4x4 height-field grid, the vertex at (1, 0) on the straight y = 0 border
    // went to (1.094, 0.188) instead of staying on y = 0 at (1.0, 0.0), and
    // the corner (0, 0) went to (0.375, 0.375) instead of (0.125, 0.125).
    let is_boundary_edge = |a: usize, b: usize| -> bool {
        edge_opposites
            .get(&(a.min(b), a.max(b)))
            .is_some_and(|opp| opp.len() == 1)
    };
    for i in 0..num_vertices {
        let neighbors = &vertex_neighbors[i];
        if neighbors.is_empty() {
            continue;
        }
        let boundary: Vec<usize> = neighbors
            .iter()
            .copied()
            .filter(|&j| is_boundary_edge(i, j))
            .collect();
        let (beta, ring): (f32, Vec<usize>) = if boundary.len() >= 2 {
            (1.0 / 8.0, boundary)
        } else {
            let n = neighbors.len() as f32;
            let beta = if n > 3.0 { 3.0 / (8.0 * n) } else { 3.0 / 16.0 };
            (beta, neighbors.iter().copied().collect())
        };
        let mut new_pos = mesh.vertices[i].coords * (1.0 - ring.len() as f32 * beta);
        for &neighbor in &ring {
            new_pos += mesh.vertices[neighbor].coords * beta;
        }
        smoothed_vertices[i] = Point3::from(new_pos);
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
