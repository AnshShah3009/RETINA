// RGB / BGR to gray on an f16 or bf16 tensor, any device.
//
// Bit-identical to the CPU backend's `cvt_color` for the same type, which
// computes `wr * r + wg * g + wb * b` on f16 / bf16 values: each product and
// each partial sum is rounded to the type, left to right. The weights arrive
// already rounded to the type (`T::from_f32(0.299)` etc.) and widened. A
// product of two 16-bit values is exact in f32, and an f32 sum of two of them
// rounded once more is the correctly rounded sum (f32 has at least 2p + 2
// bits for p = 11 and 8). Rounding goes through integer bit manipulation, so
// the compiler cannot fuse a product into the following add.
//
// Replaces `color_cvt_f16.wgsl`, which needed SHADER_F16 and rounded
// differently from the CPU; there was no bf16 shader.

struct Params {
    len: u32,    // pixels per plane
    code: u32,   // 0: RgbToGray, 1: BgrToGray
    format: u32, // 0: f16, 1: bf16
    _pad0: u32,
    wr: f32,
    wg: f32,
    wb: f32,
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

// `v` rounded to the tensor's format, as the CPU rounds after every operation.
fn round16(v: f32) -> f32 {
    return decode(encode(v));
}

fn load(idx: u32) -> f32 {
    return decode((input_data[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu);
}

// Bits of output pixel `i`.
fn gray_bits(i: u32) -> u32 {
    let n = params.len;
    let c0 = load(i);
    let g = load(n + i);
    let c2 = load(2u * n + i);
    // Planar CHW: planes 0 and 2 are R and B for RgbToGray, B and R for BgrToGray.
    let r = select(c2, c0, params.code == 0u);
    let b = select(c0, c2, params.code == 0u);
    let rg = round16(round16(params.wr * r) + round16(params.wg * g));
    return encode(rg + round16(params.wb * b));
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
