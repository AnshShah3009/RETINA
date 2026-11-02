// Bilateral filter on an f16 or bf16 tensor, any device.
//
// The arithmetic is `bilateral_f32.wgsl`'s, in f32, on the exactly widened
// inputs; only the result is rounded to the tensor's format. So the output is
// the f32 shader's result on the same values, narrowed once.
//
// Replaces `bilateral_f16.wgsl` (native f16 arithmetic: needed SHADER_F16 and
// accumulated the weights in f16) and `bilateral_bf16.wgsl` (read one bf16 per
// u32 word and truncated the result), neither of which the host could run.

struct Params {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
    format: u32, // 0: f16, 1: bf16
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two values per word
// Two invocations share each output word, so each ORs its own half into a
// zero-initialised buffer.
@group(0) @binding(1) var<storage, read_write> output_data: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

// ---- Packed 16-bit helpers, identical in every `*_packed16.wgsl` ----
//
// A 16-bit tensor is bound as u32 words: element `i` sits in the low half of
// word `i >> 1` when `i` is even and in the high half when it is odd. WGSL has
// no bf16 type and `f16` needs SHADER_F16, so values are widened to f32 with
// integer operations, which is exact for both formats on every device, and
// narrowed with round-to-nearest-even, the rounding `half::f16::from_f32` and
// `half::bf16::from_f32` use. `params.format` is 0 for f16 and 1 for bf16.

// f16 bits (low 16) -> f32, exact.
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1Fu;
    let m = h & 0x3FFu;
    if (e == 0u) {
        // Zero or subnormal: m * 2^-24, exact and normal in f32.
        let mag = f32(m) * bitcast<f32>(0x33800000u);
        return bitcast<f32>(sign | bitcast<u32>(mag));
    }
    if (e == 31u) {
        return bitcast<f32>(sign | 0x7F800000u | (m << 13u));
    }
    return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 13u));
}

// f32 -> f16 bits, round to nearest even: half's `f32_to_f16_fallback`,
// NaN payload handling included.
fn f32_to_f16(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    let sign = (x >> 16u) & 0x8000u;
    let e = (x >> 23u) & 0xFFu;
    let m = x & 0x7FFFFFu;
    if (e == 0xFFu) {
        return sign | 0x7C00u | select(0u, 0x200u, m != 0u) | (m >> 13u);
    }
    // The f16 exponent is e - 112.
    if (e >= 143u) {
        return sign | 0x7C00u;
    }
    if (e <= 112u) {
        // Subnormal or zero in f16; s is 14..=126.
        let s = 126u - e;
        if (s > 24u) {
            return sign;
        }
        let mm = m | 0x800000u;
        var hm = mm >> s;
        let rb = 1u << (s - 1u);
        if ((mm & rb) != 0u && (mm & (3u * rb - 1u)) != 0u) {
            hm += 1u;
        }
        return sign | hm;
    }
    let bits = sign | ((e - 112u) << 10u) | (m >> 13u);
    // A carry out of the mantissa bumps the exponent, up to infinity.
    if ((m & 0x1000u) != 0u && (m & 0x2FFFu) != 0u) {
        return bits + 1u;
    }
    return bits;
}

// f32 -> bf16 bits, round to nearest even: half's `f32_to_bf16`.
fn f32_to_bf16(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
        return (x >> 16u) | 0x40u;
    }
    if ((x & 0x8000u) != 0u && (x & 0x17FFFu) != 0u) {
        return (x >> 16u) + 1u;
    }
    return x >> 16u;
}

fn decode(bits: u32) -> f32 {
    if (params.format == 1u) {
        return bitcast<f32>(bits << 16u);
    }
    return f16_to_f32(bits);
}

fn encode(v: f32) -> u32 {
    if (params.format == 1u) {
        return f32_to_bf16(v);
    }
    return f32_to_f16(v);
}
// ---- end of packed 16-bit helpers ----

fn load(idx: u32) -> f32 {
    return decode((input_data[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu);
}

fn get_val(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    return load(u32(cl_y) * params.width + u32(cl_x));
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

    let idx = u32(y) * params.width + x_u32;
    atomicOr(&output_data[idx >> 1u], encode(sum / norm) << ((idx & 1u) << 4u));
}
