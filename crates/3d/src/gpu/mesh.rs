use nalgebra::{Point3, Vector3};

/// Laplacian mesh smoothing with uniform weights.
///
/// Iteratively moves each vertex towards the centroid of its neighbors.
/// `lambda` controls the smoothing strength (0..1, typically 0.5).
pub fn laplacian_smooth(
    v: &mut [Point3<f32>],
    f: &[[usize; 3]],
    iters: usize,
    lambda: f32,
) -> Result<(), String> {
    if v.is_empty() || f.is_empty() {
        return Ok(());
    }
    // Build adjacency: for each vertex, collect unique neighbor indices
    let n = v.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for face in f {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if !adj[a].contains(&b) {
                adj[a].push(b);
            }
            if !adj[b].contains(&a) {
                adj[b].push(a);
            }
        }
    }
    // Iterative Laplacian smoothing
    for _ in 0..iters {
        let old = v.to_vec();
        for i in 0..n {
            if adj[i].is_empty() {
                continue;
            }
            let centroid: Vector3<f32> =
                adj[i].iter().map(|&j| old[j].coords).sum::<Vector3<f32>>()
                    / adj[i].len() as f32;
            v[i] = Point3::from(old[i].coords * (1.0 - lambda) + centroid * lambda);
        }
    }
    Ok(())
}

/// Compute vertex normals by averaging adjacent face normals.
pub fn compute_vertex_normals(
    v: &[Point3<f32>],
    f: &[[usize; 3]],
) -> Result<Vec<Vector3<f32>>, String> {
    let mut normals = vec![Vector3::zeros(); v.len()];
    for face in f {
        let v0 = v[face[0]];
        let v1 = v[face[1]];
        let v2 = v[face[2]];
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let fn_ = e1.cross(&e2); // area-weighted normal
        for &idx in face {
            normals[idx] += fn_;
        }
    }
    for n in &mut normals {
        let len = n.norm();
        if len > 1e-9 {
            *n /= len;
        }
    }
    Ok(normals)
}
