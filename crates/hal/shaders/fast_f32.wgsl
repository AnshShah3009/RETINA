struct Params {
    width: u32,
    height: u32,
    threshold: u32, // bitcast to f32 for float threshold
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_f32(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    return input_data[idx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }

    // Bug 4 fix: skip 3px border to match CPU
    if (x < 3 || x >= i32(params.width) - 3 || y < 3 || y >= i32(params.height) - 3) {
        output_data[u32(y) * params.width + u32(x)] = 0.0;
        return;
    }

    let p = get_f32(x, y);
    // Bug 3 fix: interpret threshold as f32 bits via bitcast
    let thresh = bitcast<f32>(params.threshold);
    let high = p + thresh;
    let low = max(p - thresh, 0.0);

    let v0 = get_f32(x, y - 3);
    let v1 = get_f32(x + 1, y - 3);
    let v2 = get_f32(x + 2, y - 2);
    let v3 = get_f32(x + 3, y - 1);
    let v4 = get_f32(x + 3, y);
    let v5 = get_f32(x + 3, y + 1);
    let v6 = get_f32(x + 2, y + 2);
    let v7 = get_f32(x + 1, y + 3);
    let v8 = get_f32(x, y + 3);
    let v9 = get_f32(x - 1, y + 3);
    let v10 = get_f32(x - 2, y + 2);
    let v11 = get_f32(x - 3, y + 1);
    let v12 = get_f32(x - 3, y);
    let v13 = get_f32(x - 3, y - 1);
    let v14 = get_f32(x - 2, y - 2);
    let v15 = get_f32(x - 1, y - 3);

    var b_mask = 0u;
    var d_mask = 0u;

    if (v0 > high) { b_mask |= 1u; } else if (v0 < low) { d_mask |= 1u; }
    if (v1 > high) { b_mask |= 2u; } else if (v1 < low) { d_mask |= 2u; }
    if (v2 > high) { b_mask |= 4u; } else if (v2 < low) { d_mask |= 4u; }
    if (v3 > high) { b_mask |= 8u; } else if (v3 < low) { d_mask |= 8u; }
    if (v4 > high) { b_mask |= 16u; } else if (v4 < low) { d_mask |= 16u; }
    if (v5 > high) { b_mask |= 32u; } else if (v5 < low) { d_mask |= 32u; }
    if (v6 > high) { b_mask |= 64u; } else if (v6 < low) { d_mask |= 64u; }
    if (v7 > high) { b_mask |= 128u; } else if (v7 < low) { d_mask |= 128u; }
    if (v8 > high) { b_mask |= 256u; } else if (v8 < low) { d_mask |= 256u; }
    if (v9 > high) { b_mask |= 512u; } else if (v9 < low) { d_mask |= 512u; }
    if (v10 > high) { b_mask |= 1024u; } else if (v10 < low) { d_mask |= 1024u; }
    if (v11 > high) { b_mask |= 2048u; } else if (v11 < low) { d_mask |= 2048u; }
    if (v12 > high) { b_mask |= 4096u; } else if (v12 < low) { d_mask |= 4096u; }
    if (v13 > high) { b_mask |= 8192u; } else if (v13 < low) { d_mask |= 8192u; }
    if (v14 > high) { b_mask |= 16384u; } else if (v14 < low) { d_mask |= 16384u; }
    if (v15 > high) { b_mask |= 32768u; } else if (v15 < low) { d_mask |= 32768u; }

    let b_ext = b_mask | (b_mask << 16u);
    let d_ext = d_mask | (d_mask << 16u);

    var is_corner = false;
    for (var i = 0u; i < 16u; i++) {
        if (((b_ext >> i) & 0x1FFu) == 0x1FFu) { is_corner = true; break; }
        if (((d_ext >> i) & 0x1FFu) == 0x1FFu) { is_corner = true; break; }
    }

    // Bug 2 fix: compute min-diff score to match CPU
    var score = 0.0;
    if (is_corner) {
        let diffs = array<f32, 16>(
            abs(v0 - p), abs(v1 - p), abs(v2 - p), abs(v3 - p),
            abs(v4 - p), abs(v5 - p), abs(v6 - p), abs(v7 - p),
            abs(v8 - p), abs(v9 - p), abs(v10 - p), abs(v11 - p),
            abs(v12 - p), abs(v13 - p), abs(v14 - p), abs(v15 - p)
        );
        var min_d = diffs[0];
        for (var i = 1u; i < 16u; i++) {
            min_d = min(min_d, diffs[i]);
        }
        score = min_d;
    }

    output_data[u32(y) * params.width + u32(x)] = score;
}
