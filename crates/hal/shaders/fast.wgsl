struct Params {
    width: u32,
    height: u32,
    threshold: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>; // Packed u8
@group(0) @binding(2) var<uniform> params: Params;

fn get_u8(x: i32, y: i32) -> u32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    let u32_idx = idx / 4u;
    let shift = (idx % 4u) * 8u;
    return (input_data[u32_idx] >> shift) & 0xFFu;
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

        // Bug 4 fix: skip 3px border to match CPU
        if (x < 3 || x >= i32(params.width) - 3 || y < 3 || y >= i32(params.height) - 3) {
            continue;
        }

        let p = get_u8(x, y);
        let high = p + params.threshold;
        let low = select(0u, p - params.threshold, p >= params.threshold);

        let v0 = get_u8(x, y - 3);
        let v1 = get_u8(x + 1, y - 3);
        let v2 = get_u8(x + 2, y - 2);
        let v3 = get_u8(x + 3, y - 1);
        let v4 = get_u8(x + 3, y);
        let v5 = get_u8(x + 3, y + 1);
        let v6 = get_u8(x + 2, y + 2);
        let v7 = get_u8(x + 1, y + 3);
        let v8 = get_u8(x, y + 3);
        let v9 = get_u8(x - 1, y + 3);
        let v10 = get_u8(x - 2, y + 2);
        let v11 = get_u8(x - 3, y + 1);
        let v12 = get_u8(x - 3, y);
        let v13 = get_u8(x - 3, y - 1);
        let v14 = get_u8(x - 2, y - 2);
        let v15 = get_u8(x - 1, y - 3);

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
        var score = 0u;
        if (is_corner) {
            var min_d = 255u;
            min_d = min(min_d, u32(abs(i32(v0) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v1) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v2) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v3) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v4) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v5) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v6) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v7) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v8) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v9) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v10) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v11) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v12) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v13) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v14) - i32(p))));
            min_d = min(min_d, u32(abs(i32(v15) - i32(p))));
            score = min_d;
        }
        res_combined = res_combined | ((score & 0xFFu) << (k * 8u));
    }

    output_data[u32(y) * ((params.width + 3u) / 4u) + x_u32] = res_combined;
}
