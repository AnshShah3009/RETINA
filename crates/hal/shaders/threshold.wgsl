struct Params {
    thresh: u32,
    max_value: u32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn apply_thresh(val: u32) -> u32 {
    var res: u32 = 0u;
    if (params.typ == 0u) { // Binary
        if (val > params.thresh) { res = params.max_value; } else { res = 0u; }
    } else if (params.typ == 1u) { // BinaryInv
        if (val > params.thresh) { res = 0u; } else { res = params.max_value; }
    } else if (params.typ == 2u) { // Trunc
        res = min(val, params.thresh);
    } else if (params.typ == 3u) { // ToZero
        if (val > params.thresh) { res = val; } else { res = 0u; }
    } else if (params.typ == 4u) { // ToZeroInv
        if (val > params.thresh) { res = 0u; } else { res = val; }
    }
    return res;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_idx = global_id.x;
    if (u32_idx * 4u >= params.len) {
        return;
    }

    let combined = input_data[u32_idx];
    var res_combined = 0u;

    for (var i = 0u; i < 4u; i++) {
        let pixel_idx = u32_idx * 4u + i;
        if (pixel_idx >= params.len) {
            break;
        }

        let val = (combined >> (i * 8u)) & 0xFFu;
        let res = apply_thresh(val);
        res_combined = res_combined | ((res & 0xFFu) << (i * 8u));
    }

    output_data[u32_idx] = res_combined;
}
