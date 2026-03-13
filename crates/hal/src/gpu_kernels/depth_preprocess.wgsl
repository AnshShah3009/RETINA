// Depth Preprocessing Shader
// Computes vertex and normal maps from a 2D depth image

@group(0) @binding(0) var<storage, read> depth_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertex_map: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> normal_map: array<vec3<f32>>;

@group(1) @binding(0) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(1) var<uniform> size: vec2<u32>;

// Backproject depth to 3D
fn depth_to_3d(u: u32, v: u32, depth: f32) -> vec3<f32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;
    
    let x = (f32(u) - cx) * depth / fx;
    let y = (f32(v) - cy) * depth / fy;
    
    return vec3<f32>(x, y, depth);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    
    if (u >= size.x || v >= size.y) {
        return;
    }
    
    let idx = v * size.x + u;
    let depth = depth_image[idx];
    
    if (depth <= 0.0) {
        vertex_map[idx] = vec3<f32>(0.0, 0.0, 0.0);
        normal_map[idx] = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }
    
    // 1. Compute vertex
    let p = depth_to_3d(u, v, depth);
    vertex_map[idx] = p;
    
    // 2. Compute normal using central differences
    if (u > 0u && u < size.x - 1u && v > 0u && v < size.y - 1u) {
        let dz_dx = (depth_image[idx + 1u] - depth_image[idx - 1u]) * 0.5;
        let dz_dy = (depth_image[idx + size.x] - depth_image[idx - size.x]) * 0.5;
        
        // Approx normal from gradients (simplified)
        let n = normalize(vec3<f32>(-dz_dx, -dz_dy, 1.0));
        normal_map[idx] = n;
    } else {
        normal_map[idx] = vec3<f32>(0.0, 0.0, 1.0);
    }
}
