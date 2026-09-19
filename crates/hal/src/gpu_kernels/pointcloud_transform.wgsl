// Point Cloud Transform Shader
// Transforms 3D points using a 4x4 transformation matrix

@group(0) @binding(0) var<storage, read> points_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> points_out: array<f32>;
@group(0) @binding(2) var<uniform> transform: mat4x4<f32>;
@group(0) @binding(3) var<uniform> num_points: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_points) {
        return;
    }
    
    let base = idx * 4u;
    let p = vec4<f32>(points_in[base], points_in[base+1u], points_in[base+2u], points_in[base+3u]);
    let transformed = transform * p;
    
    points_out[base] = transformed.x;
    points_out[base+1u] = transformed.y;
    points_out[base+2u] = transformed.z;
    points_out[base+3u] = 1.0;
}
