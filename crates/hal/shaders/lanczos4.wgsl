// Lanczos-4 resize of a planar CHW f32 tensor.
//
// Element (ch, y, x) of a `TensorShape::new(c, h, w)` tensor is at
// `ch * h * w + y * w + x`. Each plane is an independent image; the channel is
// `global_id.z` (the host dispatches `channels` workgroups in z).

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
}

fn lanczos_weight(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.0001) {
        return 1.0;
    }
    if (ax >= 4.0) {
        return 0.0;
    }
    
    let pi = 3.14159265359;
    let pi_x = pi * ax;
    let pi_x_4 = pi * ax / 4.0;
    
    return sin(pi_x) * sin(pi_x_4) / (pi_x * pi_x / 4.0);
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_dst = global_id.x;
    let y_dst = global_id.y;
    let ch = global_id.z;

    // The host never dispatches an empty source; the clamps below would wrap.
    if (x_dst >= params.dst_w || y_dst >= params.dst_h || ch >= params.channels
        || params.src_w == 0u || params.src_h == 0u) {
        return;
    }

    let src_width_f = f32(params.src_w) - 1.0;
    let src_height_f = f32(params.src_h) - 1.0;
    let dst_width_f = max(f32(params.dst_w) - 1.0, 1.0);
    let dst_height_f = max(f32(params.dst_h) - 1.0, 1.0);

    let src_x_f = f32(x_dst) * src_width_f / dst_width_f;
    let src_y_f = f32(y_dst) * src_height_f / dst_height_f;

    let kernel_radius = 4;

    let x0 = i32(floor(src_x_f)) - kernel_radius + 1;
    let y0 = i32(floor(src_y_f)) - kernel_radius + 1;

    let dx0 = f32(x0) - src_x_f;
    let dy0 = f32(y0) - src_y_f;

    let src_plane = ch * params.src_h * params.src_w;
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var j = 0; j < 8; j = j + 1) {
        let dy = dy0 + f32(j);
        let wy = lanczos_weight(dy);

        for (var i = 0; i < 8; i = i + 1) {
            let dx = dx0 + f32(i);
            let wx = lanczos_weight(dx);

            let weight = wx * wy;

            let src_x_clamped = clamp(x0 + i32(i), 0, i32(params.src_w) - 1);
            let src_y_clamped = clamp(y0 + i32(j), 0, i32(params.src_h) - 1);

            let idx = src_plane + u32(src_y_clamped) * params.src_w + u32(src_x_clamped);
            sum = sum + input_data[idx] * weight;
            weight_sum = weight_sum + weight;
        }
    }

    let dst_plane = ch * params.dst_h * params.dst_w;
    output_data[dst_plane + y_dst * params.dst_w + x_dst] = sum / weight_sum;
}
