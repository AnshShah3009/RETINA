enable f16;
struct Params {
    width: u32,
    height: u32,
    thresh: u32,
    max_value: u32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f16>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

fn apply_thresh(val: f16) -> f16 {
    var res: f16 = 0.0h;
    let thresh_f16 = bitcast<f16>(params.thresh);
    let max_val_f16 = bitcast<f16>(params.max_value);
    
    if (params.typ == 0u) { // Binary
        if (val > thresh_f16) { res = max_val_f16; } else { res = 0.0h; }
    } else if (params.typ == 1u) { // BinaryInv
        if (val > thresh_f16) { res = 0.0h; } else { res = max_val_f16; }
    } else if (params.typ == 2u) { // Trunc
        res = min(val, thresh_f16);
    } else if (params.typ == 3u) { // ToZero
        if (val > thresh_f16) { res = val; } else { res = 0.0h; }
    } else if (params.typ == 4u) { // ToZeroInv
        if (val > thresh_f16) { res = 0.0h; } else { res = val; }
    }
    return res;
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
