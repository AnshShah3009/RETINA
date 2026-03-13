struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> score_map: array<u32>;
@group(0) @binding(1) var<storage, read_write> suppressed_map: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_score(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0.0;
    }
    let idx = u32(y) * params.width + u32(x);
    let u32_idx = idx / 2u;
    let shift = (idx % 2u) * 16u;
    let bf16_bits = (score_map[u32_idx] >> shift) & 0xFFFFu;
    return bitcast<f32>(bf16_bits << 16u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }

    let s = get_score(x, y);
    let idx = u32(y) * params.width + u32(x);

    if (s > 0.0) {
        var is_max = true;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) { continue; }
                let neighbor_s = get_score(x + dx, y + dy);
                if (neighbor_s > s || (neighbor_s == s && (dy > 0 || (dy == 0 && dx > 0)))) {
                    is_max = false;
                    break;
                }
            }
            if (!is_max) { break; }
        }
        if (is_max) {
            let f32_score = bitcast<u32>(s);
            let bf16_score = f32_score >> 16u;
            // The suppressed_map is an array<u32>, but we are writing bf16 (u16) values.
            // This implies that suppressed_map should be treated as an array of u16s,
            // or we need to pack two bf16s into one u32.
            // Assuming suppressed_map is conceptually an array of u16s,
            // and we are writing to the correct u32 index and bit-shifting.
            let u32_idx = idx / 2u;
            let shift = (idx % 2u) * 16u;
            
            // Use atomic max since we're writing half-words concurrently? 
            // We'll write the raw u16 logic on output arrays
            // This requires an atomic operation to safely update a half-word within a u32.
            // For simplicity and assuming no concurrent writes to the *same* u32_idx for different shifts,
            // or that the user intends to overwrite the full u32, we'll use a direct write.
            // If true atomic half-word update is needed, it would be more complex.
            // For now, let's assume the user wants to write the bf16_score into the correct half of the u32.
            // This requires reading the existing u32, clearing the relevant 16 bits, and ORing in the new value.
            // However, the instruction `suppressed_map[idx] = bf16_score;` suggests a direct write.
            // Given `suppressed_map: array<u32>`, `suppressed_map[idx]` accesses a full u32.
            // If `idx` is meant to be a u16 index, then `suppressed_map[idx / 2u]` would be the u32 index.
            // The instruction `suppressed_map[idx] = bf16_score;` is problematic if `suppressed_map` is `array<u32>`
            // and `bf16_score` is `u16` (conceptually).
            // Let's follow the instruction literally, assuming `suppressed_map` is effectively an array of u16s
            // and `idx` is the u16 index, which means `suppressed_map` should be `array<u16>` or we need to pack.
            // The original `suppressed_map[u32(y) * params.width + u32(x)] = suppressed_s;`
            // implies `suppressed_map` is indexed by the pixel index.
            // If `suppressed_map` is `array<u32>`, then `suppressed_map[idx]` writes a full u32.
            // The instruction `suppressed_map[idx] = bf16_score;` would implicitly cast `bf16_score` (u16) to `u32`.
            // This would mean `suppressed_map` stores `0x0000BF16`.
            // This is inconsistent with `score_map` packing two bf16s into one u32.
            // Let's assume the intent is to pack the bf16 score into the correct half of the u32.
            // This requires an atomic operation or careful indexing.
            // Given the comment "// We'll write the raw u16 logic on output arrays",
            // and the instruction `suppressed_map[idx] = bf16_score;`,
            // it seems the user wants to write the bf16 value directly.
            // This implies `suppressed_map` should be `array<u16>` or the write logic needs to pack.
            // Since `suppressed_map` is `array<u32>`, `suppressed_map[idx]` writes a full u32.
            // The most direct interpretation of `suppressed_map[idx] = bf16_score;` is to write `u32(bf16_score)`.
            // This would mean each pixel's bf16 score occupies a full u32 in `suppressed_map`.
            // This is inefficient but matches the instruction's syntax.
            // Let's stick to the instruction's syntax for `suppressed_map[idx] = bf16_score;`
            // and assume `bf16_score` is implicitly promoted to `u32`.
            suppressed_map[idx] = bf16_score;
        } else {
            suppressed_map[idx] = 0u;
        }
    } else {
        suppressed_map[idx] = 0u;
    }
}
