struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_f32(idx: u32) -> f32 {
    return input_data[idx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_dst = global_id.x;
    let y_dst = global_id.y;
    
    if (x_dst >= params.dst_w || y_dst >= params.dst_h) {
        return;
    }

    let src_width_f = f32(params.src_w) - 1.0;
    let src_height_f = f32(params.src_h) - 1.0;
    let dst_width_f = f32(params.dst_w) - 1.0;
    let dst_height_f = f32(params.dst_h) - 1.0;

    let src_x_f = f32(x_dst) * src_width_f / dst_width_f;
    let src_y_f = f32(y_dst) * src_height_f / dst_height_f;
    
    let x0 = u32(max(0.0, floor(src_x_f)));
    let y0 = u32(max(0.0, floor(src_y_f)));
    let x1 = min(params.src_w - 1u, x0 + 1u);
    let y1 = min(params.src_h - 1u, y0 + 1u);
    
    let dx = src_x_f - f32(x0);
    let dy = src_y_f - f32(y0);
    
    let p00 = get_f32(y0 * params.src_w + x0);
    let p10 = get_f32(y0 * params.src_w + x1);
    let p01 = get_f32(y1 * params.src_w + x0);
    let p11 = get_f32(y1 * params.src_w + x1);
    
    let val_f = mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
    
    output_data[y_dst * params.dst_w + x_dst] = val_f;
}
