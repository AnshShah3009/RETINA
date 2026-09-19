// Type casting kernels

// Shared bindings for all kernels in this module
@group(0) @binding(0) var<storage, read> input_u8: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_f32: array<f32>;

@group(0) @binding(2) var<storage, read> input_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_u8: array<u32>;

@compute @workgroup_size(256)
fn u8_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_pixels = arrayLength(&output_f32);
    if (idx >= num_pixels) {
        return;
    }
    
    let word_idx = idx / 4u;
    let byte_offset = (idx % 4u) * 8u;
    let val = (input_u8[word_idx] >> byte_offset) & 0xFFu;
    
    output_f32[idx] = f32(val) / 255.0;
}

@compute @workgroup_size(256)
fn f32_to_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; // Word index (4 pixels)
    let num_words = arrayLength(&output_u8);
    let num_pixels = arrayLength(&input_f32);
    
    if (idx >= num_words) {
        return;
    }
    
    var packed: u32 = 0u;
    for (var i = 0u; i < 4u; i++) {
        let pixel_idx = idx * 4u + i;
        if (pixel_idx < num_pixels) {
            let val = u32(clamp(input_f32[pixel_idx] * 255.0 + 0.5, 0.0, 255.0));
            packed |= (val << (i * 8u));
        }
    }
    
    output_u8[idx] = packed;
}
