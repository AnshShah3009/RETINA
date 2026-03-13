// RGBD Odometry Shader
// Computes visual odometry from consecutive RGBD frames

@group(0) @binding(0) var<storage, read> source_depth: array<f32>;
@group(0) @binding(1) var<storage, read> target_depth: array<f32>;
@group(0) @binding(2) var<storage, read> target_vertices: array<vec3<f32>>; // Precomputed from target
@group(0) @binding(3) var<storage, read> target_normals: array<vec3<f32>>;

@group(1) @binding(0) var<storage, read_write> jacobian_buffer: array<vec3<f32>>; // JTJ rows (simplified)
@group(1) @binding(1) var<storage, read_write> residual_buffer: array<f32>;
@group(1) @binding(2) var<uniform> transform: mat4x4<f32>;
@group(1) @binding(3) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(4) var<uniform> image_size: vec2<u32>;
@group(1) @binding(5) var<uniform> max_distance: f32;

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

// Project 3D to image coordinates
fn project_3d(point: vec3<f32>) -> vec2<i32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;
    
    let u = i32(point.x * fx / point.z + cx);
    let v = i32(point.y * fy / point.z + cy);
    
    return vec2<i32>(u, v);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    
    if (u >= image_size.x || v >= image_size.y) {
        return;
    }
    
    let pixel_idx = v * image_size.x + u;
    let depth = source_depth[pixel_idx];
    
    // Skip invalid depth
    if (depth <= 0.0) {
        residual_buffer[pixel_idx] = 0.0;
        return;
    }
    
    // Backproject to 3D in source frame
    let source_point = depth_to_3d(u, v, depth);
    
    // Transform to target frame
    let p_h = vec4<f32>(source_point, 1.0);
    let target_point = (transform * p_h).xyz;
    
    // Project to target image
    let proj = project_3d(target_point);
    
    // Check bounds
    if (proj.x < 0 || proj.x >= i32(image_size.x) || proj.y < 0 || proj.y >= i32(image_size.y)) {
        residual_buffer[pixel_idx] = 0.0;
        return;
    }
    
    let target_idx = u32(proj.y) * image_size.x + u32(proj.x);
    let target_v = target_vertices[target_idx];
    let target_n = target_normals[target_idx];
    
    // Check if target vertex is valid
    if (target_v.z <= 0.0) {
        residual_buffer[pixel_idx] = 0.0;
        return;
    }
    
    // Point-to-plane error
    let residual = dot(target_point - target_v, target_n);
    
    // Check distance threshold
    if (length(target_point - target_v) > max_distance) {
        residual_buffer[pixel_idx] = 0.0;
        return;
    }
    
    // Compute Jacobian (simplified - just store residual for now)
    residual_buffer[pixel_idx] = residual;
    
    // Full Jacobian would include SE(3) derivatives
    // jacobian_buffer[pixel_idx] = ...
}
