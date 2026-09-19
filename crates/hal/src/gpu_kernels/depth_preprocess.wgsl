// Depth Preprocessing Shader
// Computes vertex and normal maps from a 2D depth image
// Normal computation uses central differences on vertex positions (not raw depth).

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

    // 2. Compute normal using central differences on vertex positions
    //    This matches the CPU reference: dx = right - left, dy = down - up, n = normalize(dx x dy)
    if (u > 0u && u < size.x - 1u && v > 0u && v < size.y - 1u) {
        let d_left  = depth_image[idx - 1u];
        let d_right = depth_image[idx + 1u];
        let d_up    = depth_image[idx - size.x];
        let d_down  = depth_image[idx + size.x];

        if (d_left > 0.0 && d_right > 0.0 && d_up > 0.0 && d_down > 0.0) {
            let p_left  = depth_to_3d(u - 1u, v, d_left);
            let p_right = depth_to_3d(u + 1u, v, d_right);
            let p_up    = depth_to_3d(u, v - 1u, d_up);
            let p_down  = depth_to_3d(u, v + 1u, d_down);

            let dx = p_right - p_left;
            let dy = p_down - p_up;
            let n = normalize(cross(dx, dy));
            normal_map[idx] = n;
        } else {
            normal_map[idx] = vec3<f32>(0.0, 0.0, 0.0);
        }
    } else {
        normal_map[idx] = vec3<f32>(0.0, 0.0, 0.0);
    }
}
