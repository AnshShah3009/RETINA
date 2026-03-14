struct Params {
    width: u32,
    height: u32,
    ksize: u32,
    border_mode: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> gy_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_f32(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    return input_data[idx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    let p00 = get_f32(x - 1, y - 1);
    let p01 = get_f32(x,     y - 1);
    let p02 = get_f32(x + 1, y - 1);
    let p10 = get_f32(x - 1, y);
    let p12 = get_f32(x + 1, y);
    let p20 = get_f32(x - 1, y + 1);
    let p21 = get_f32(x,     y + 1);
    let p22 = get_f32(x + 1, y + 1);

    let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
    let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);

    let idx = u32(y) * params.width + u32(x);
    gx_data[idx] = gx;
    gy_data[idx] = gy;
}
