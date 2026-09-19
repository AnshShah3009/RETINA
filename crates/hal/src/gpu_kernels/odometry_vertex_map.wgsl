struct Params {
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

@group(0) @binding(0) var<storage, read> depth: array<f32>;
@group(0) @binding(1) var<storage, read_write> vertex_map: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let d = depth[idx];

    // Invalid depth (0 or negative)
    if (d <= 0.001) {
        vertex_map[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Backproject to 3D
    let px = f32(x);
    let py = f32(y);
    let vx = (px - params.cx) * d / params.fx;
    let vy = (py - params.cy) * d / params.fy;
    let vz = d;

    vertex_map[idx] = vec4<f32>(vx, vy, vz, 1.0);
}
