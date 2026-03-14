enable f16;
struct Params {
    width: u32,
    height: u32,
    threshold: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f16>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_f16(x: i32, y: i32) -> f16 {
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
        output_data[u32(y) * params.width + u32(x)] = 0.0h;
        return;
    }

    let p = get_f16(x, y);
    let thresh = f16(bitcast<f32>(params.threshold));
    let high = p + thresh;
    // Bug 8 fix: clamp low threshold at zero to prevent underflow
    let low = max(p - thresh, f16(0.0));

    let v0 = get_f16(x, y - 3);
    let v1 = get_f16(x + 1, y - 3);
    let v2 = get_f16(x + 2, y - 2);
    let v3 = get_f16(x + 3, y - 1);
    let v4 = get_f16(x + 3, y);
    let v5 = get_f16(x + 3, y + 1);
    let v6 = get_f16(x + 2, y + 2);
    let v7 = get_f16(x + 1, y + 3);
    let v8 = get_f16(x, y + 3);
    let v9 = get_f16(x - 1, y + 3);
    let v10 = get_f16(x - 2, y + 2);
    let v11 = get_f16(x - 3, y + 1);
    let v12 = get_f16(x - 3, y);
    let v13 = get_f16(x - 3, y - 1);
    let v14 = get_f16(x - 2, y - 2);
    let v15 = get_f16(x - 1, y - 3);

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
    var score = 0.0h;
    if (is_corner) {
        var min_d = abs(v0 - p);
        min_d = min(min_d, abs(v1 - p));
        min_d = min(min_d, abs(v2 - p));
        min_d = min(min_d, abs(v3 - p));
        min_d = min(min_d, abs(v4 - p));
        min_d = min(min_d, abs(v5 - p));
        min_d = min(min_d, abs(v6 - p));
        min_d = min(min_d, abs(v7 - p));
        min_d = min(min_d, abs(v8 - p));
        min_d = min(min_d, abs(v9 - p));
        min_d = min(min_d, abs(v10 - p));
        min_d = min(min_d, abs(v11 - p));
        min_d = min(min_d, abs(v12 - p));
        min_d = min(min_d, abs(v13 - p));
        min_d = min(min_d, abs(v14 - p));
        min_d = min(min_d, abs(v15 - p));
        score = min_d;
    }

    output_data[u32(y) * params.width + u32(x)] = score;
}
