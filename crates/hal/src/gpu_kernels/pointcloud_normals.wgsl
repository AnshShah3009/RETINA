// Point Cloud Normal Computation
// Computes surface normals from point cloud using neighbor search

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> neighbor_indices: array<u32>; // Flattened k-NN indices
@group(0) @binding(3) var<uniform> num_points: u32;
@group(0) @binding(4) var<uniform> k_neighbors: u32;

// Compute normal using PCA on neighbors
fn compute_normal(point_idx: u32) -> vec3<f32> {
    let p = points[point_idx].xyz;
    
    // Compute centroid of neighbors
    var centroid = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < k_neighbors; i = i + 1u) {
        let neighbor_idx = neighbor_indices[point_idx * k_neighbors + i];
        centroid = centroid + points[neighbor_idx].xyz;
    }
    centroid = centroid / f32(k_neighbors);
    
    // Compute covariance matrix (simplified - just use cross product of two vectors)
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    
    if (k_neighbors >= 3u) {
        let idx0 = neighbor_indices[point_idx * k_neighbors + 0u];
        let idx1 = neighbor_indices[point_idx * k_neighbors + 1u];
        let idx2 = neighbor_indices[point_idx * k_neighbors + 2u];
        
        let v0 = points[idx0].xyz - p;
        let v1 = points[idx1].xyz - p;
        
        normal = cross(v0, v1);
        let len = length(normal);
        if (len > 0.0001) {
            normal = normal / len;
        }
    }
    
    return normal;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_points) {
        return;
    }
    
    normals[idx] = vec4<f32>(compute_normal(idx), 0.0);
}
