struct Params {
    width: u32,
    height: u32,
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
    let thresh_u32 = params.thresh;
    let max_val_u32 = params.max_value;

    if (params.typ == 0u) { // Binary
        if (val > thresh_u32) { res = max_val_u32; } else { res = 0u; }
    } else if (params.typ == 1u) { // BinaryInv
        if (val > thresh_u32) { res = 0u; } else { res = max_val_u32; }
    } else if (params.typ == 2u) { // Trunc
        res = min(val, thresh_u32);
    } else if (params.typ == 3u) { // ToZero
        if (val > thresh_u32) { res = val; } else { res = 0u; }
    } else if (params.typ == 4u) { // ToZeroInv
        if (val > thresh_u32) { res = 0u; } else { res = val; }
    }
    return res;
}

fn get_f32(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    let u32_idx = idx / 2u;
    let shift = (idx % 2u) * 16u;
    let bf16_bits = (input_data[u32_idx] >> shift) & 0xFFFFu;
    return bitcast<f32>(bf16_bits << 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }

    let val = input_data[idx];
    output_data[idx] = apply_thresh(val);
}
