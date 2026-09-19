// Point Cloud Transform Shader
// Transforms Nx4 point cloud using a 4x4 transformation matrix
//
// Input: array of f32 (Nx4 point cloud, stored as [x0,y0,z0,w0, x1,y1,z1,w1, ...])
// Params: 4x4 transformation matrix (16 floats) + num_points
// Output: transformed points (same layout)

struct Params {
    num_points: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    transform: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> input_points: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_points: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) {
        return;
    }

    let base = idx * 4u;
    let p = vec4<f32>(
        input_points[base],
        input_points[base + 1u],
        input_points[base + 2u],
        input_points[base + 3u],
    );

    let transformed = params.transform * p;

    output_points[base] = transformed.x;
    output_points[base + 1u] = transformed.y;
    output_points[base + 2u] = transformed.z;
    output_points[base + 3u] = transformed.w;
}
