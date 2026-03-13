struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> score_map: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read_write> suppressed_map: array<u32>; // Packed u8
@group(0) @binding(2) var<uniform> params: Params;

fn get_score(x: i32, y: i32) -> u32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0u;
    }
    let idx = u32(y) * params.width + u32(x);
    let u32_idx = idx / 4u;
    let shift = (idx % 4u) * 8u;
    return (score_map[u32_idx] >> shift) & 0xFFu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x;
    let y = i32(global_id.y);
    
    if (x_u32 * 4u >= params.width || y >= i32(params.height)) {
        return;
    }

    var res_combined = 0u;
    for (var k = 0u; k < 4u; k++) {
        let x = i32(x_u32 * 4u + k);
        if (x >= i32(params.width)) { break; }

        let s = get_score(x, y);
        var suppressed_s = 0u;

        if (s > 0u) {
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
            if (is_max) { suppressed_s = s; }
        }
        res_combined = res_combined | ((suppressed_s & 0xFFu) << (k * 8u));
    }

    suppressed_map[u32(y) * ((params.width + 3u) / 4u) + x_u32] = res_combined;
}
