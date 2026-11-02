// RGB / BGR to gray on an f16 or bf16 tensor, any device.
//
// Bit-identical to the CPU backend's `cvt_color` for the same type, which
// computes `wr * r + wg * g + wb * b` on f16 / bf16 values: each product and
// each partial sum is rounded to the type, left to right. The weights arrive
// already rounded to the type (`T::from_f32(0.299)` etc.) and widened. A
// product of two 16-bit values is exact in f32, and an f32 sum of two of them
// rounded once more is the correctly rounded sum (f32 has at least 2p + 2
// bits for p = 11 and 8). Rounding goes through integer bit manipulation, so
// the compiler cannot fuse a product into the following add, and the f32
// products and sums are computed on bits (`mul_bits`, `add_bits`) so that a
// device which flushes f32 subnormals cannot change them: bf16 subnormals
// widen to f32 subnormals.
//
// Replaces `color_cvt_f16.wgsl`, which needed SHADER_F16 and rounded
// differently from the CPU; there was no bf16 shader.

struct Params {
    len: u32,    // pixels per plane
    code: u32,   // 0: RgbToGray, 1: BgrToGray
    format: u32, // 0: f16, 1: bf16
    _pad0: u32,
    // The weights as the tensor type, widened to f32 (exact), read as bits:
    // positive normal numbers below 1, which `mul_bits` relies on.
    wr: u32,
    wg: u32,
    wb: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two values per word, CHW
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
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

// ---- Exact f32 multiply and add, subnormals included ----
//
// A bf16 value below 2^-126 widens to an f32 subnormal, and a product or sum
// of small bf16 values can land below 2^-126; the device flushes f32
// subnormals in float arithmetic (the RTX 5070 Ti on Vulkan does), which made
// bf16 results there 0. `mul_bits` and `add_bits` are the IEEE f32 product
// and sum on bits, subnormal or not; `color_cvt_f32.wgsl` has the same
// functions and the argument for them.

// The IEEE f32 product of `w` and `x` (bits), rounded to nearest even, with
// subnormal results. `w` must be a positive normal number below 1, so the
// product never overflows.
fn mul_bits(w: u32, x: u32) -> u32 {
    let sign = (w ^ x) & 0x80000000u;
    let ex = (x >> 23u) & 0xFFu;
    let fx = x & 0x7FFFFFu;
    if (ex == 0xFFu) {
        if (fx != 0u) {
            // NaN: propagated with the quiet bit set, as x86 and AArch64 do.
            return x | 0x400000u;
        }
        return sign | 0x7F800000u;
    }
    // x = mx * 2^(bx - 150), w = mw * 2^(ew - 150).
    var mx = fx | 0x800000u;
    var bx = ex;
    if (ex == 0u) {
        if (fx == 0u) {
            return sign;
        }
        mx = fx;
        bx = 1u;
    }
    let mw = (w & 0x7FFFFFu) | 0x800000u;
    let ew = (w >> 23u) & 0xFFu;

    // P = mw * mx < 2^48 as hi:lo, from 16-bit halves.
    let a0 = mw & 0xFFFFu;
    let a1 = mw >> 16u;
    let b0 = mx & 0xFFFFu;
    let b1 = mx >> 16u;
    let ll = a0 * b0;
    let mid = a1 * b0 + a0 * b1; // below 2^25
    let lo = ll + (mid << 16u);
    let hi = a1 * b1 + (mid >> 16u) + select(0u, 1u, lo < ll);

    // The product is P * 2^k with its leading bit at `top`; mw >= 2^23 puts
    // `top` at 23 or above.
    var top: i32;
    if (hi != 0u) {
        top = 32 + i32(firstLeadingBit(hi));
    } else {
        top = i32(firstLeadingBit(lo));
    }
    let k = i32(bx) + i32(ew) - 300;
    let eb = top + k + 127; // biased exponent if the result is normal
    // Bits of P below the result's last place: down to 24 significant bits
    // when normal, down to a multiple of 2^-149 when subnormal (more than
    // top - 23 then, so at least 1).
    var shift = top - 23;
    if (eb < 1) {
        shift = -(k + 149);
    }
    if (shift > 49) {
        // Below half the smallest subnormal.
        return sign;
    }
    let s = u32(shift);
    var q: u32;
    var round_bit = false;
    var sticky = false;
    if (s == 0u) {
        q = lo; // P < 2^24, so hi is 0
    } else if (s < 32u) {
        q = (lo >> s) | (hi << (32u - s));
        round_bit = ((lo >> (s - 1u)) & 1u) != 0u;
        sticky = (lo & ((1u << (s - 1u)) - 1u)) != 0u;
    } else if (s == 32u) {
        q = hi;
        round_bit = (lo >> 31u) != 0u;
        sticky = (lo & 0x7FFFFFFFu) != 0u;
    } else {
        q = hi >> (s - 32u);
        round_bit = ((hi >> (s - 33u)) & 1u) != 0u;
        sticky = lo != 0u || (hi & ((1u << (s - 33u)) - 1u)) != 0u;
    }
    let up = select(0u, 1u, round_bit && (sticky || (q & 1u) != 0u));
    if (eb >= 1) {
        // q carries the implicit bit, so this is eb << 23 plus the fraction;
        // a carry out of the significand bumps the exponent.
        return sign | ((u32(eb - 1) << 23u) + q + up);
    }
    // Subnormal; a carry into bit 23 makes it the smallest normal.
    return sign | (q + up);
}

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

// f32 bits rounded to the tensor's format and widened back, with no float
// operation on the value.
fn r16(x: u32) -> u32 {
    return bitcast<u32>(decode(encode(bitcast<f32>(x))));
}
// ---- end of exact f32 multiply and add ----

fn load(idx: u32) -> f32 {
    return decode((input_data[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu);
}

// Bits of output pixel `i`.
fn gray_bits(i: u32) -> u32 {
    let n = params.len;
    let c0 = bitcast<u32>(load(i));
    let g = bitcast<u32>(load(n + i));
    let c2 = bitcast<u32>(load(2u * n + i));
    // Planar CHW: planes 0 and 2 are R and B for RgbToGray, B and R for BgrToGray.
    let r = select(c2, c0, params.code == 0u);
    let b = select(c0, c2, params.code == 0u);
    let rg = r16(add_bits(r16(mul_bits(params.wr, r)), r16(mul_bits(params.wg, g))));
    return encode(bitcast<f32>(add_bits(rg, r16(mul_bits(params.wb, b)))));
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // One invocation per output word (two pixels), 1-D range dispatched as a
    // 2-D grid (see gpu_kernels::dispatch_1d).
    let word = global_id.x + global_id.y * num_wg.x * 64u;
    let first = word * 2u;
    if (first >= params.len) {
        return;
    }
    let lo = gray_bits(first);
    // An odd pixel count leaves the last word's high half outside the image.
    var hi = 0u;
    if (first + 1u < params.len) {
        hi = gray_bits(first + 1u);
    }
    output_data[word] = lo | (hi << 16u);
}
