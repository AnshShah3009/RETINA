// Dense Point-to-Plane ICP Tracking
// Computes local Hessian (J^T J) and Gradient (J^T r) contributions per pixel.

struct Params {
    width: u32,
    height: u32,
    max_dist: f32,
    max_angle: f32, // cos(angle)
}

@group(0) @binding(0) var<storage, read> source_depth: array<f32>;
@group(0) @binding(1) var<storage, read> target_data: array<vec4<f32>>; // x: depth, yzw: normal
@group(0) @binding(2) var<storage, read_write> output_linear_system: array<f32>; // 27 floats per pixel
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(0) @binding(5) var<uniform> intrinsics_inv: vec4<f32>; // 1/fx, 1/fy, cx, cy
@group(0) @binding(6) var<uniform> curr_to_prev: mat4x4<f32>; // Current estimate

fn project(p: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        p.x * intrinsics.x / p.z + intrinsics.z,
        p.y * intrinsics.y / p.z + intrinsics.w
    );
}

fn unproject(u: f32, v: f32, d: f32) -> vec3<f32> {
    return vec3<f32>(
        (u - intrinsics_inv.z) * intrinsics_inv.x * d,
        (v - intrinsics_inv.w) * intrinsics_inv.y * d,
        d
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    
    if (u >= params.width || v >= params.height) { return; }
    
    let pixel_idx = v * params.width + u;
    let d_curr = source_depth[pixel_idx];
    
    if (d_curr <= 0.0) {
        set_zero(pixel_idx);
        return;
    }
    
    // 1. Point in current camera frame
    let p_curr = unproject(f32(u), f32(v), d_curr);
    
    // 2. Transform to previous camera frame (target)
    let p_prev = (curr_to_prev * vec4<f32>(p_curr, 1.0)).xyz;
    
    if (p_prev.z <= 0.0) {
        set_zero(pixel_idx);
        return;
    }
    
    // 3. Project to target image coordinates
    let uv_prev = project(p_prev);
    let u_p = u32(round(uv_prev.x));
    let v_p = u32(round(uv_prev.y));
    
    if (u_p >= params.width || v_p >= params.height) {
        set_zero(pixel_idx);
        return;
    }
    
    // 4. Get target depth and normal
    let target_sample = target_data[v_p * params.width + u_p];
    let d_prev = target_sample.x;
    let n_prev = target_sample.yzw;
    
    if (d_prev <= 0.0 || length(n_prev) < 0.5) {
        set_zero(pixel_idx);
        return;
    }
    
    // 5. Correspondence check
    let dist = abs(p_prev.z - d_prev);
    if (dist > params.max_dist) {
        set_zero(pixel_idx);
        return;
    }
    
    // 6. Compute Point-to-Plane linear system contribution
    // residual r = (p_prev - q_prev) . n_prev
    // Jacobian J = [ n_prev^T , (p_prev x n_prev)^T ] (6-dim row vector)
    
    let q_prev = unproject(f32(u_p), f32(v_p), d_prev);
    let residual = dot(p_prev - q_prev, n_prev);
    
    let cross_p_n = cross(p_prev, n_prev);
    
    var J: array<f32, 6>;
    J[0] = n_prev.x; J[1] = n_prev.y; J[2] = n_prev.z;
    J[3] = cross_p_n.x; J[4] = cross_p_n.y; J[5] = cross_p_n.z;
    
    // Compute J^T * J (21 elements) and J^T * r (6 elements)
    let out_base = pixel_idx * 27u;
    var out_idx = 0u;
    
    // Symmetric J^T J
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

fn set_zero(idx: u32) {
    let out_base = idx * 27u;
    for (var i = 0u; i < 27u; i++) {
        output_linear_system[out_base + i] = 0.0;
    }
}
