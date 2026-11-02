// FAST-9 corner scores on an f16 or bf16 tensor, any device.
//
// Bit-identical to the CPU backend's `fast_detect` for the same type. The CPU
// does each operation on f16 / bf16 values, which `half` evaluates as the f32
// operation rounded back to the type; the inputs here are widened exactly and
// every intermediate the CPU rounds (`p + t`, `p - t`, each `|v - p|`, each
// partial sum and the final `/ 16`) is rounded with `r16`. An f32 add or
// subtract of two such values rounded once more to f16 / bf16 is the
// correctly rounded result (f32 has at least 2p + 2 bits for p = 11 and 8),
// so this matches the CPU exactly. The f32 operations are done on bits
// (`add_bits`, `greater`, `div16_bits`) so that a device which flushes f32
// subnormals cannot change them: bf16 subnormals widen to f32 subnormals.
//
// Replaces `fast_f16.wgsl` (native f16 arithmetic, needed SHADER_F16, and
// clamped the low threshold at 0, unlike the CPU and f32 shaders) and
// `fast_bf16.wgsl` (truncated the score, and two invocations did a
// non-atomic read-modify-write of the same output word).

struct Params {
    width: u32,
    height: u32,
    // The threshold as the tensor type, widened to f32 (exact), read as
    // bits.
    threshold_bits: u32,
    format: u32,    // 0: f16, 1: bf16
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two values per word
// Two invocations share each output word, so each ORs its own half into a
// zero-initialised buffer. Non-corners and the border stay 0.
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

// ---- Exact f32 add, subnormals included ----
//
// A bf16 value below 2^-126 widens to an f32 subnormal, and the device
// flushes those in float arithmetic and comparisons (the RTX 5070 Ti on
// Vulkan does), so plain `a + b` and `a < b` went wrong on bf16 subnormals
// and on sums and differences that land below 2^-126. Comparisons go through
// `order_key` on the bits instead, and adds through `add_bits`, which is the
// IEEE f32 sum, subnormal or not. The f32 shader `color_cvt_f32.wgsl` has the
// same functions and the argument for them.

// x * 2^64 for |x| < 2^-60 (bits), exact and normal unless x is zero.
fn scale_up(x: u32) -> f32 {
    let mag = x & 0x7FFFFFFFu;
    if (mag >= 0x00800000u) {
        return bitcast<f32>(x + (64u << 23u));
    }
    let v = f32(mag) * bitcast<f32>(0x15000000u); // mag * 2^-85
    return bitcast<f32>(bitcast<u32>(v) | (x & 0x80000000u));
}

// s * 2^-64 as f32 bits, for an `s` that is a multiple of 2^-85.
fn scale_down(s: f32) -> u32 {
    let bits = bitcast<u32>(s);
    if ((bits & 0x7FFFFFFFu) >= 0x20800000u) {
        return bits - (64u << 23u);
    }
    return (bits & 0x80000000u) | u32(abs(s) * bitcast<f32>(0x6A000000u)); // * 2^85
}

// The IEEE f32 sum of `a` and `b` (bits): one float add when no operand or
// result can be subnormal, otherwise the same add scaled by 2^64.
fn add_bits(a: u32, b: u32) -> u32 {
    let ma = a & 0x7FFFFFFFu;
    let mb = b & 0x7FFFFFFFu;
    if (min(ma, mb) >= 0x0D800000u || max(ma, mb) >= 0x21800000u) {
        return bitcast<u32>(bitcast<f32>(a) + bitcast<f32>(b));
    }
    return scale_down(scale_up(a) + scale_up(b));
}

fn is_nan_bits(x: u32) -> bool {
    return (x & 0x7FFFFFFFu) > 0x7F800000u;
}

// Non-NaN f32 bits -> u32 in value order, -0 equal to +0.
fn order_key(x: u32) -> u32 {
    if ((x & 0x80000000u) != 0u) {
        return 0x80000000u - (x & 0x7FFFFFFFu);
    }
    return 0x80000000u + x;
}

// `a > b` on f32 bits, false when either is NaN, as a float compare.
fn greater(a: u32, b: u32) -> bool {
    return !is_nan_bits(a) && !is_nan_bits(b) && order_key(a) > order_key(b);
}

// f32 bits rounded to the tensor's format and widened back, with no float
// operation on the value.
fn r16(x: u32) -> u32 {
    return bitcast<u32>(decode(encode(bitcast<f32>(x))));
}

// x / 16 as f32 bits, round to nearest even (exact for the scores here).
fn div16_bits(x: u32) -> u32 {
    let e = (x >> 23u) & 0xFFu;
    if (e > 4u) {
        if (e == 0xFFu) {
            return x;
        }
        return x - (4u << 23u);
    }
    // Subnormal result: the significand moves right by 5 - max(e, 1).
    var m = x & 0x7FFFFFu;
    var s = 4u;
    if (e != 0u) {
        m |= 0x800000u;
        s = 5u - e;
    }
    let q = m >> s;
    let rem = m & ((1u << s) - 1u);
    let half = 1u << (s - 1u);
    let up = select(0u, 1u, rem > half || (rem == half && (q & 1u) != 0u));
    return (x & 0x80000000u) | (q + up);
}
// ---- end of exact f32 add ----

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
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }
    // 3px border, as the CPU (its output there is 0, the buffer's initial value).
    if (x < 3 || x >= i32(params.width) - 3 || y < 3 || y >= i32(params.height) - 3) {
        return;
    }

    // The 16-pixel Bresenham circle, in the CPU's order.
    var circle = array<vec2<i32>, 16>(
        vec2<i32>(0, -3), vec2<i32>(1, -3), vec2<i32>(2, -2), vec2<i32>(3, -1),
        vec2<i32>(3, 0), vec2<i32>(3, 1), vec2<i32>(2, 2), vec2<i32>(1, 3),
        vec2<i32>(0, 3), vec2<i32>(-1, 3), vec2<i32>(-2, 2), vec2<i32>(-3, 1),
        vec2<i32>(-3, 0), vec2<i32>(-3, -1), vec2<i32>(-2, -2), vec2<i32>(-1, -3),
    );

    // Everything below works on f32 bits (see `add_bits`), so bf16
    // subnormals and sums or differences below 2^-126 are exact.
    let p = bitcast<u32>(get_val(x, y));
    let t = params.threshold_bits;
    let high = r16(add_bits(p, t));
    let low = r16(add_bits(p, t ^ 0x80000000u));

    var vals: array<u32, 16>;
    var b_mask = 0u;
    var d_mask = 0u;
    for (var i = 0u; i < 16u; i++) {
        let v = bitcast<u32>(get_val(x + circle[i].x, y + circle[i].y));
        vals[i] = v;
        // Two independent tests, as `has_9_contiguous_generic`.
        if (greater(v, high)) { b_mask |= 1u << i; }
        if (greater(low, v)) { d_mask |= 1u << i; }
    }

    let b_ext = b_mask | (b_mask << 16u);
    let d_ext = d_mask | (d_mask << 16u);

    var is_corner = false;
    for (var i = 0u; i < 16u; i++) {
        if (((b_ext >> i) & 0x1FFu) == 0x1FFu) { is_corner = true; break; }
        if (((d_ext >> i) & 0x1FFu) == 0x1FFu) { is_corner = true; break; }
    }
    if (!is_corner) {
        return;
    }

    // Mean absolute difference as the CPU spells it, `v > p ? v - p : p - v`,
    // summed in order and rounded at each step.
    var score = 0u;
    for (var i = 0u; i < 16u; i++) {
        let v = vals[i];
        var diff: u32;
        if (greater(v, p)) {
            diff = add_bits(v, p ^ 0x80000000u);
        } else {
            diff = add_bits(p, v ^ 0x80000000u);
        }
        score = r16(add_bits(score, r16(diff)));
    }
    let bits = encode(bitcast<f32>(div16_bits(score)));

    let idx = u32(y) * params.width + u32(x);
    atomicOr(&output_data[idx >> 1u], bits << ((idx & 1u) << 4u));
}
