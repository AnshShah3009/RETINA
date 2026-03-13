// Mesh Laplacian Smoothing Shader
// Performs Laplacian smoothing on mesh vertices

@group(0) @binding(0) var<storage, read> vertices_in: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> vertices_out: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> adjacency_list: array<u32>; // Flattened neighbor indices
@group(0) @binding(3) var<storage, read> adjacency_offsets: array<u32>; // Start index for each vertex
@group(0) @binding(4) var<uniform> num_vertices: u32;
@group(0) @binding(5) var<uniform> lambda: f32; // Smoothing factor

// Compute Laplacian (average of neighbors)
fn laplacian(vertex_idx: u32) -> vec3<f32> {
    let start = adjacency_offsets[vertex_idx];
    let end = adjacency_offsets[vertex_idx + 1u];
    
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0u;
    
    for (var i = start; i < end; i = i + 1u) {
        let neighbor_idx = adjacency_list[i];
        sum = sum + vertices_in[neighbor_idx];
        count = count + 1u;
    }
    
    if (count == 0u) {
        return vertices_in[vertex_idx];
    }
    
    return sum / f32(count);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_vertices) {
        return;
    }
    
    let current = vertices_in[idx];
    let neighbor_avg = laplacian(idx);
    
    // Laplacian update: move toward neighbor average
    vertices_out[idx] = current + lambda * (neighbor_avg - current);
}
