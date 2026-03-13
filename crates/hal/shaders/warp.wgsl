// GPU Warp (Affine/Perspective) Kernel
// Performs image warping using bilinear interpolation.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    warp_type: u32, // 0: Affine, 1: Perspective
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>; // Packed u8
@group(0) @binding(2) var<uniform> matrix: mat3x3<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_u8(idx: u32) -> f32 {
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return f32((input_data[u32_idx] >> shift) & 0xFFu);
}

fn get_pixel_bilinear(x: f32, y: f32) -> f32 {
    if (x < 0.0 || x >= f32(params.src_w - 1u) || y < 0.0 || y >= f32(params.src_h - 1u)) {
        return 0.0;
    }
    
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let p00 = get_u8(y0 * params.src_w + x0);
    let p10 = get_u8(y0 * params.src_w + x1);
    let p01 = get_u8(y1 * params.src_w + x0);
    let p11 = get_u8(y1 * params.src_w + x1);
    
    return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x; // Processes 4 pixels
    let y_dst = global_id.y;
    
    if (x_u32 * 4u >= params.dst_w || y_dst >= params.dst_h) {
        return;
    }

    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x_dst = x_u32 * 4u + i;
        if (x_dst >= params.dst_w) { break; }

        var src_x = 0.0;
        var src_y = 0.0;
        
        if (params.warp_type == 0u) { // Affine
            src_x = matrix[0][0] * f32(x_dst) + matrix[1][0] * f32(y_dst) + matrix[2][0];
            src_y = matrix[0][1] * f32(x_dst) + matrix[1][1] * f32(y_dst) + matrix[2][1];
        } else { // Perspective
            let w = matrix[0][2] * f32(x_dst) + matrix[1][2] * f32(y_dst) + matrix[2][2];
            let inv_w = 1.0 / select(w, 1.0, abs(w) < 1e-10);
            src_x = (matrix[0][0] * f32(x_dst) + matrix[1][0] * f32(y_dst) + matrix[2][0]) * inv_w;
            src_y = (matrix[0][1] * f32(x_dst) + matrix[1][1] * f32(y_dst) + matrix[2][1]) * inv_w;
        }
        
        let val = u32(clamp(get_pixel_bilinear(src_x, src_y) + 0.5, 0.0, 255.0));
        res_combined = res_combined | (val << (i * 8u));
    }

    output_data[y_dst * ((params.dst_w + 3u) / 4u) + x_u32] = res_combined;
}
