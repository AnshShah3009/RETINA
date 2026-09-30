struct Params {
    width: u32,
    height: u32,
    radius: i32,
    // The host writes f32 here (`-0.5 / (sigma * sigma)`); declaring them u32
    // made `exp(f32 * u32)` a type error, so this shader never compiled at all.
    // The f32 sibling has them as f32.
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// bfloat16 -> f32. A bf16 value is the high 16 bits of an f32, so widening is a
// shift into the exponent position.
//
// This shader was reading the raw u32 and using it as a value: the range term
// became `(bits - bits)^2` rather than a colour difference, so the range weight
// was computed from integer encodings. The three sibling shaders that do this
// correctly - fast, fast_nms and threshold - unpack with a shift and a mask;
// this one was missing it. Two bf16 per u32, matching those.
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16);
}

fn get_val(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    return bf16_to_f32(input_data[idx]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x;
    let y = i32(global_id.y);
    
    if (x_u32 >= params.width || y >= i32(params.height)) {
        return;
    }

    let x = i32(x_u32);
    let center_val = get_val(x, y);
    var sum = 0.0;
    var norm = 0.0;

    for (var j = -params.radius; j <= params.radius; j++) {
        for (var i = -params.radius; i <= params.radius; i++) {
            let val = get_val(x + i, y + j);
            
            let dist_sq = f32(i * i + j * j);
            let range_sq = (val - center_val) * (val - center_val);
            
            let weight = exp(dist_sq * params.sigma_space_sq_inv + range_sq * params.sigma_color_sq_inv);
            sum += val * weight;
            norm += weight;
        }
    }
    
    let final_val = sum / norm;
    // Narrow back to bf16 for storage. Assigning the f32 directly to a `u32`
    // binding is a type error; truncating to u32 is wrong for a value that is
    // not already an integer.
    let narrowed = u32(bitcast<u32>(final_val) >> 16);
    output_data[u32(y) * params.width + x_u32] = narrowed;
}
