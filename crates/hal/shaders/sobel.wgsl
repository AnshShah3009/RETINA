struct Params {
    width: u32,
    height: u32,
    ksize: u32,
    border_mode: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> gx_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> gy_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_x = global_id.x; // Processes 4 pixels horizontally
    let y = i32(global_id.y);

    if (u32_x * 4u >= params.width || u32(y) >= params.height) {
        return;
    }

    var res_gx = 0u;
    var res_gy = 0u;

    for (var i = 0u; i < 4u; i++) {
        let x = i32(u32_x * 4u + i);
        if (u32(x) >= params.width) { break; }

        let p00 = get_pixel(x - 1, y - 1);
        let p01 = get_pixel(x,     y - 1);
        let p02 = get_pixel(x + 1, y - 1);
        let p10 = get_pixel(x - 1, y);
        let p12 = get_pixel(x + 1, y);
        let p20 = get_pixel(x - 1, y + 1);
        let p21 = get_pixel(x,     y + 1);
        let p22 = get_pixel(x + 1, y + 1);

        // Compute signed gradients (no abs), then bias by 128 to store in u8.
        // This preserves sign information: values < 128 are negative gradients,
        // values > 128 are positive gradients, 128 is zero.
        // For absolute-value gradient magnitude, use the f32 Sobel output path.
        let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
        let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);

        let gx_out = clamp(gx + 128.0, 0.0, 255.0);
        let gy_out = clamp(gy + 128.0, 0.0, 255.0);

        res_gx = res_gx | ((u32(gx_out) & 0xFFu) << (i * 8u));
        res_gy = res_gy | ((u32(gy_out) & 0xFFu) << (i * 8u));
    }

    gx_data[u32_x + u32(y) * ((params.width + 3u) / 4u)] = res_gx;
    gy_data[u32_x + u32(y) * ((params.width + 3u) / 4u)] = res_gy;
}
