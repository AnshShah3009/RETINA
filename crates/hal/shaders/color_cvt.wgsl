struct Params {
    len: u32,
    code: u32, // 0: RgbToGray, 1: BgrToGray, 2: GrayToRgb
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_u8(idx: u32) -> u32 {
    let u32_idx = idx / 4u;
    let shift = (idx % 4u) * 8u;
    return (input_data[u32_idx] >> shift) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_u32_idx = global_id.x;
    
    if (params.code == 0u || params.code == 1u) { // RgbToGray or BgrToGray
        // Input: 3 bytes per pixel. Output: 1 byte per pixel.
        // Each thread produces 4 grayscale pixels -> 1 u32.
        
        var res_combined = 0u;
        for (var i = 0u; i < 4u; i++) {
            let out_pixel_idx = out_u32_idx * 4u + i;
            if (out_pixel_idx >= params.len) { break; }
            
            let src_base = out_pixel_idx * 3u;
            let c0 = get_u8(src_base);
            let c1 = get_u8(src_base + 1u);
            let c2 = get_u8(src_base + 2u);
            
            var gray: u32 = 0u;
            if (params.code == 0u) { // RGB
                gray = u32(0.299 * f32(c0) + 0.587 * f32(c1) + 0.114 * f32(c2));
            } else { // BGR
                gray = u32(0.114 * f32(c0) + 0.587 * f32(c1) + 0.299 * f32(c2));
            }
            res_combined = res_combined | ((gray & 0xFFu) << (i * 8u));
        }
        
        if (out_u32_idx * 4u < params.len) {
            output_data[out_u32_idx] = res_combined;
        }
    } else if (params.code == 2u) { // GrayToRgb
        // Input: 1 byte per pixel. Output: 3 bytes per pixel.
        // This is complex because 1 output pixel (3 bytes) doesn't align with u32 (4 bytes).
        // For now, let's just implement RgbToGray as it's most common for CV.
    }
}
