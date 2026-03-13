// GPU Morphology (Erode/Dilate) Kernel
// Performs morphological operations on packed u8 grayscale images.

struct Params {
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    typ: u32, // 0: Erode, 1: Dilate
    iterations: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read> kernel_mask: array<u32>; // Non-packed mask (0 or 1)
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>; // Packed u8
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> u32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        if (params.typ == 0u) { return 255u; } else { return 0u; }
    }
    let idx = u32(y) * params.width + u32(x);
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return (input_data[u32_idx] >> shift) & 0xFFu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x; // Processes 4 pixels horizontally
    let y = global_id.y;
    
    if (x_u32 * 4u >= params.width || y >= params.height) {
        return;
    }

    let cx = i32(params.kw / 2u);
    let cy = i32(params.kh / 2u);

    var res_combined = 0u;
    
    for (var i = 0u; i < 4u; i++) {
        let x = i32(x_u32 * 4u + i);
        if (u32(x) >= params.width) { break; }

        var val = if (params.typ == 0u) { 255u } else { 0u };

        for (var ky = 0u; ky < params.kh; ky++) {
            for (var kx = 0u; kx < params.kw; kx++) {
                if (kernel_mask[ky * params.kw + kx] == 0u) {
                    continue;
                }
                
                let src_x = x + i32(kx) - cx;
                let src_y = i32(y) + i32(ky) - cy;
                
                let pixel = get_pixel(src_x, src_y);
                
                if (params.typ == 0u) {
                    val = min(val, pixel);
                } else {
                    val = max(val, pixel);
                }
            }
        }
        
        res_combined = res_combined | (val << (i * 8u));
    }

    output_data[y * ((params.width + 3u) / 4u) + x_u32] = res_combined;
}
