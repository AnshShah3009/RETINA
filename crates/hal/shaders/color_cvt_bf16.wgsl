struct Params {
    len: u32,
    code: u32, // 0: RgbToGray, 1: BgrToGray, 2: GrayToRgb
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_u32(idx: u32) -> u32 {
    return input_data[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_pixel_idx = global_id.x;
    if (out_pixel_idx >= params.len) { return; }
    
    if (params.code == 0u || params.code == 1u) { // RgbToGray or BgrToGray
        // Input: 3 floats per pixel. Output: 1 float per pixel.
        let x = i32(out_pixel_idx % params.width);
        let y = i32(out_pixel_idx / params.width);
        let c0 = get_f32(x, y, 0u);
        let c1 = get_f32(x, y, 1u);
        let c2 = get_f32(x, y, 2u);
        
        var gray: f32 = 0.0;
        if (params.code == 0u) { // RGB
            gray = 0.299 * c0 + 0.587 * c1 + 0.114 * c2;
        } else { // BGR
            gray = 0.114 * c0 + 0.587 * c1 + 0.299 * c2;
        }
        
        output_data[out_pixel_idx] = bitcast<u32>(gray);
    } else if (params.code == 2u) { // GrayToRgb
        // Input: 1 byte per pixel. Output: 3 bytes per pixel.
        // This is complex because 1 output pixel (3 bytes) doesn't align with u32 (4 bytes).
        // For now, let's just implement RgbToGray as it's most common for CV.
    }
}
