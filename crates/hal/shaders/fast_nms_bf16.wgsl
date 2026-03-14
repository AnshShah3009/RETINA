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

    // Determine the bf16 score to write
    var bf16_score = 0u;

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
            let f32_bits = bitcast<u32>(s);
            bf16_score = f32_bits >> 16u;
        }
    }

    // Bug 9 fix: pack two bf16 values per u32, matching the input format.
    // Each thread writes its bf16 score into the correct half of the u32.
    let u32_idx = idx / 2u;
    let shift = (idx % 2u) * 16u;
    let existing = suppressed_map[u32_idx];
    let mask = ~(0xFFFFu << shift);
    suppressed_map[u32_idx] = (existing & mask) | (bf16_score << shift);
}
