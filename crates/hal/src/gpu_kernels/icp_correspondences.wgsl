// ICP Correspondence Finding Shader
// Finds nearest neighbors between source and target point clouds

@group(0) @binding(0) var<storage, read> source_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> target_points: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> correspondences: array<u32>; // Target indices
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

@group(1) @binding(0) var<uniform> transform: mat4x4<f32>;
@group(1) @binding(1) var<uniform> num_source: u32;
@group(1) @binding(2) var<uniform> num_target: u32;
@group(1) @binding(3) var<uniform> max_distance: f32;

// Brute-force nearest neighbor (can be optimized with KDTree on GPU)
fn find_nearest(query: vec3<f32>) -> vec2<f32> { // Returns (index, distance)
    var best_idx = 0u;
    var best_dist = 999999.0;
    
    for (var i = 0u; i < num_target; i = i + 1u) {
        let dist = length(query - target_points[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return vec2<f32>(f32(best_idx), best_dist);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_source) {
        return;
    }
    
    // Transform source point
    let p = source_points[idx];
    let p_h = vec4<f32>(p, 1.0);
    let transformed = (transform * p_h).xyz;
    
    // Find nearest neighbor
    let result = find_nearest(transformed);
    let best_idx = u32(result.x);
    let best_dist = result.y;
    
    // Store correspondence if within threshold
    if (best_dist <= max_distance) {
        correspondences[idx] = best_idx;
        distances[idx] = best_dist;
    } else {
        correspondences[idx] = 0xFFFFFFFFu; // Invalid marker
        distances[idx] = -1.0;
    }
}
