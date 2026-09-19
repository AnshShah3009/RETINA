// RGBD Odometry Shader — per-pixel linear system accumulation
// Computes J^T*J (21 upper-triangle) + J^T*r (6) = 27 floats per pixel.
// The Jacobian is the analytical point-to-plane form:
//   J = [n_x, n_y, n_z, (p x n)_x, (p x n)_y, (p x n)_z]
// where p is the transformed source point and n is the target normal.

struct OdometryParams {
    width: u32,
    height: u32,
    max_iterations: u32,
    min_depth: f32,
    max_depth: f32,
    // Pad to 48 bytes (3 x 16) for proper WGSL uniform alignment
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> source_depth: array<f32>;
@group(0) @binding(1) var<storage, read> target_vertices: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> target_normals: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> output_linear_system: array<f32>; // 27 floats per pixel

@group(1) @binding(0) var<uniform> transform: mat4x4<f32>;
@group(1) @binding(1) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(2) var<uniform> params: OdometryParams;

fn depth_to_3d(u: u32, v: u32, depth: f32) -> vec3<f32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;

    let x = (f32(u) - cx) * depth / fx;
    let y = (f32(v) - cy) * depth / fy;

    return vec3<f32>(x, y, depth);
}

fn project_3d(point: vec3<f32>) -> vec2<i32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;

    let u = i32(point.x * fx / point.z + cx);
    let v = i32(point.y * fy / point.z + cy);

    return vec2<i32>(u, v);
}

fn set_zero(pixel_idx: u32) {
    let out_base = pixel_idx * 27u;
    for (var i = 0u; i < 27u; i++) {
        output_linear_system[out_base + i] = 0.0;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;

    if (u >= params.width || v >= params.height) {
        return;
    }

    let pixel_idx = v * params.width + u;
    let depth = source_depth[pixel_idx];

    // Skip invalid depth
    if (depth <= 0.0) {
        set_zero(pixel_idx);
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
    if (proj.x < 0 || proj.x >= i32(params.width) || proj.y < 0 || proj.y >= i32(params.height)) {
        set_zero(pixel_idx);
        return;
    }

    let target_idx = u32(proj.y) * params.width + u32(proj.x);
    let target_v = target_vertices[target_idx];
    let target_n = target_normals[target_idx];

    // Check if target vertex and normal are valid
    if (target_v.z <= 0.0 || dot(target_n, target_n) < 1e-6) {
        set_zero(pixel_idx);
        return;
    }

    // Point-to-plane residual: r = n . (p_transformed - p_target)
    let diff = target_point - target_v;
    let residual = dot(diff, target_n);

    // Analytical Jacobian: J = [n_x, n_y, n_z, (p x n)_x, (p x n)_y, (p x n)_z]
    let p = target_point;
    let n = target_n;
    let cross_p_n = cross(p, n);

    var J: array<f32, 6>;
    J[0] = n.x; J[1] = n.y; J[2] = n.z;
    J[3] = cross_p_n.x; J[4] = cross_p_n.y; J[5] = cross_p_n.z;

    // Compute J^T * J (21 upper-triangle elements) and J^T * r (6 elements) = 27 total
    let out_base = pixel_idx * 27u;
    var out_idx = 0u;

    // Upper triangle of J^T * J (symmetric)
    for (var i = 0u; i < 6u; i++) {
        for (var j = i; j < 6u; j++) {
            output_linear_system[out_base + out_idx] = J[i] * J[j];
            out_idx++;
        }
    }

    // J^T * r
    for (var i = 0u; i < 6u; i++) {
        output_linear_system[out_base + out_idx] = J[i] * residual;
        out_idx++;
    }
}
