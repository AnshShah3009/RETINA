enable f16;
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> score_map: array<f16>;
@group(0) @binding(1) var<storage, read_write> suppressed_map: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_score(x: i32, y: i32) -> f16 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0.0h;
    }
    let idx = u32(y) * params.width + u32(x);
    return score_map[idx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    
    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }

    let s = get_score(x, y);
    var suppressed_s = 0.0h;

    if (s > 0.0h) {
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

    suppressed_map[u32(y) * params.width + u32(x)] = suppressed_s;
}
