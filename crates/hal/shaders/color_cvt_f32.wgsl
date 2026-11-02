// RGB / BGR to gray on an f32 tensor, bit-identical to the CPU backend.
//
// The CPU computes `wr * r + wg * g + wb * b` in f32: each product rounded,
// then the sums left to right, each rounded, with no fused multiply-add and
// with IEEE subnormals. A plain float expression cannot reproduce that on a
// GPU: the compiler may contract `a * b + c` into an FMA (one rounding instead
// of two), and the device flushes subnormal operands and results to zero (the
// RTX 5070 Ti on Vulkan does). So:
//
// * Each product is computed with integer operations (`mul_bits`): the exact
//   48-bit product of the significands, rounded to nearest even at the
//   precision of the result, subnormal or normal. No float multiply exists for
//   the compiler to fuse into the following add.
// * Each sum is one float add (`add_bits`) when neither operand nor the result
//   can be subnormal, and otherwise the same add on both operands scaled by
//   2^64, where they and the result are normal and the add is exact or rounds
//   exactly as the CPU's does (see `add_bits`).
//
// Signed zeros, infinities and overflow match the CPU bit for bit. A NaN input
// gives a NaN, but which payload a NaN sum carries is the host FPU's business
// (x86 and AArch64 differ), so only the class is part of the contract.
//
// This shader used to evaluate the BGR sum in the opposite order from the CPU
// (`wb * b + wg * g + wr * r`) with plain float arithmetic.

struct Params {
    len: u32,  // pixels per plane
    code: u32, // 0: RgbToGray, 1: BgrToGray
    _pad0: u32,
    _pad1: u32,
    // f32 bits of the CPU's weights, 0.299f32, 0.587f32 and 0.114f32: positive
    // normal numbers below 1, which `mul_bits` relies on.
    wr: u32,
    wg: u32,
    wb: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

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
    // Zero or subnormal: mag * 2^-149 * 2^64 = mag * 2^-85 (0x15000000).
    let v = f32(mag) * bitcast<f32>(0x15000000u);
    return bitcast<f32>(bitcast<u32>(v) | (x & 0x80000000u));
}

// s * 2^-64 as f32 bits, for an `s` that is a multiple of 2^-85.
fn scale_down(s: f32) -> u32 {
    let bits = bitcast<u32>(s);
    // At or above 2^-62 (0x20800000) the result is normal.
    if ((bits & 0x7FFFFFFFu) >= 0x20800000u) {
        return bits - (64u << 23u);
    }
    // Otherwise a subnormal (or zero) with s * 2^85 (0x6A000000) as its
    // integer significand, below 2^23.
    return (bits & 0x80000000u) | u32(abs(s) * bitcast<f32>(0x6A000000u));
}

// The IEEE f32 sum of `a` and `b` (bits), subnormals included.
//
// One float add is exact or correctly rounded except when the device flushes
// a subnormal. That cannot change the result when both operands are at least
// 2^-100 (0x0D800000): both are normal, and their sum is a multiple of 2^-123,
// so it is zero or normal. Nor when one operand is at least 2^-60
// (0x21800000) and the other below 2^-100: the small one is less than half an
// ulp of the large one and cannot change its rounding (it is not a tie).
// Otherwise both are below 2^-60: scaled by 2^64 they are normal, no larger
// than 2^4, and multiples of 2^-85. Their sum is then either at least 2^-62,
// where it rounds to the same 24 bits the unscaled sum does, or below it,
// where it fits in 23 bits above 2^-85 and is exact, as the CPU's subnormal
// sum is.
fn add_bits(a: u32, b: u32) -> u32 {
    let ma = a & 0x7FFFFFFFu;
    let mb = b & 0x7FFFFFFFu;
    if (min(ma, mb) >= 0x0D800000u || max(ma, mb) >= 0x21800000u) {
        return bitcast<u32>(bitcast<f32>(a) + bitcast<f32>(b));
    }
    return scale_down(scale_up(a) + scale_up(b));
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // 1-D range dispatched as a 2-D grid so no axis exceeds 65535
    // workgroups (see gpu_kernels::dispatch_1d); surplus tail invocations
    // fall out at the bounds check below.
    let i = global_id.x + global_id.y * num_wg.x * 64u;
    if (i >= params.len) {
        return;
    }
    // Planar CHW: planes 0 and 2 are R and B for RgbToGray, B and R for
    // BgrToGray. Read as raw bits: a load is not arithmetic.
    let n = params.len;
    let c0 = bitcast<u32>(input_data[i]);
    let g = bitcast<u32>(input_data[n + i]);
    let c2 = bitcast<u32>(input_data[2u * n + i]);
    let r = select(c2, c0, params.code == 0u);
    let b = select(c0, c2, params.code == 0u);
    // ((wr * r + wg * g) + wb * b), in the CPU's order.
    let rg = add_bits(mul_bits(params.wr, r), mul_bits(params.wg, g));
    output_data[i] = bitcast<f32>(add_bits(rg, mul_bits(params.wb, b)));
}
