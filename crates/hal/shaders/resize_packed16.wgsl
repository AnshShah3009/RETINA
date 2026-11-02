// Resize of a planar CHW f16 or bf16 tensor, any device.
//
// Element (ch, y, x) of a `TensorShape::new(c, h, w)` tensor is at
// `ch * h * w + y * w + x`; each plane is an independent image, and the plane
// is `global_id.z` (the host dispatches `channels` workgroups in z).
//
// `main_bilinear` is bit-identical to the CPU backend's `resize` for the same
// type. The host passes the CPU's taps (`cpu::bilinear::bilinear_taps`): per
// output column, then per output row, the two source indices and the weights
// `1 - d` and `d`, already rounded to the type and widened (exact). The
// shader evaluates `(v00 w0x + v10 w1x) w0y + (v01 w0x + v11 w1x) w1y` left to
// right, rounding each product and each sum to the type as the CPU does. A
// product of two 16-bit values is exact in f32, and an f32 sum of two of them
// rounded once more is the correctly rounded sum (f32 has at least 2p + 2
// bits for p = 11 and 8). Rounding goes through integer bit manipulation, so
// the compiler cannot fuse a product into the following add.
//
// `main_lanczos` is `lanczos4.wgsl`'s arithmetic, in f32, on the exactly
// widened inputs; only the result is rounded to the type (there is no CPU
// Lanczos to follow).
//
// Two invocations share each output word, so each ORs its own half into the
// zero-initialised output.
//
// Replaces `resize_f16.wgsl` (computed the sampling positions in f16 and
// needed SHADER_F16) and `resize_bf16.wgsl` (read one bf16 per u32 word and
// truncated), neither of which had a caller.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
    format: u32, // 0: f16, 1: bf16
    _pad0: u32,
    _pad1: u32,
}

// One axis sample: source indices `i0`, `i1` and weights `w0 = 1 - d`, `w1 = d`.
struct Tap {
    i0: u32,
    i1: u32,
    w0: f32,
    w1: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two values per word, CHW
@group(0) @binding(1) var<storage, read_write> output_data: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
// `dst_w` column taps followed by `dst_h` row taps. Bilinear only.
@group(0) @binding(3) var<storage, read> taps: array<Tap>;

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

fn store(idx: u32, bits: u32) {
    atomicOr(&output_data[idx >> 1u], bits << ((idx & 1u) << 4u));
}

@compute @workgroup_size(16, 16)
fn main_bilinear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let ch = global_id.z;
    // The host never dispatches an empty source; it returns zeros instead.
    if (x >= params.dst_w || y >= params.dst_h || ch >= params.channels) {
        return;
    }

    let tx = taps[x];
    let ty = taps[params.dst_w + y];
    let plane = ch * params.src_h * params.src_w;
    let r0 = plane + ty.i0 * params.src_w;
    let r1 = plane + ty.i1 * params.src_w;

    let v0 = round16(round16(load(r0 + tx.i0) * tx.w0) + round16(load(r0 + tx.i1) * tx.w1));
    let v1 = round16(round16(load(r1 + tx.i0) * tx.w0) + round16(load(r1 + tx.i1) * tx.w1));
    let v = round16(v0 * ty.w0) + round16(v1 * ty.w1);

    store((ch * params.dst_h + y) * params.dst_w + x, encode(v));
}

fn lanczos_weight(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.0001) {
        return 1.0;
    }
    if (ax >= 4.0) {
        return 0.0;
    }

    let pi = 3.14159265359;
    let pi_x = pi * ax;
    let pi_x_4 = pi * ax / 4.0;

    return sin(pi_x) * sin(pi_x_4) / (pi_x * pi_x / 4.0);
}

@compute @workgroup_size(16, 16)
fn main_lanczos(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_dst = global_id.x;
    let y_dst = global_id.y;
    let ch = global_id.z;
    // The host never dispatches an empty source; the clamps below would wrap.
    if (x_dst >= params.dst_w || y_dst >= params.dst_h || ch >= params.channels
        || params.src_w == 0u || params.src_h == 0u) {
        return;
    }

    let src_width_f = f32(params.src_w) - 1.0;
    let src_height_f = f32(params.src_h) - 1.0;
    let dst_width_f = max(f32(params.dst_w) - 1.0, 1.0);
    let dst_height_f = max(f32(params.dst_h) - 1.0, 1.0);

    let src_x_f = f32(x_dst) * src_width_f / dst_width_f;
    let src_y_f = f32(y_dst) * src_height_f / dst_height_f;

    let kernel_radius = 4;

    let x0 = i32(floor(src_x_f)) - kernel_radius + 1;
    let y0 = i32(floor(src_y_f)) - kernel_radius + 1;

    let dx0 = f32(x0) - src_x_f;
    let dy0 = f32(y0) - src_y_f;

    let src_plane = ch * params.src_h * params.src_w;
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var j = 0; j < 8; j = j + 1) {
        let dy = dy0 + f32(j);
        let wy = lanczos_weight(dy);

        for (var i = 0; i < 8; i = i + 1) {
            let dx = dx0 + f32(i);
            let wx = lanczos_weight(dx);

            let weight = wx * wy;

            let src_x_clamped = clamp(x0 + i32(i), 0, i32(params.src_w) - 1);
            let src_y_clamped = clamp(y0 + i32(j), 0, i32(params.src_h) - 1);

            let idx = src_plane + u32(src_y_clamped) * params.src_w + u32(src_x_clamped);
            sum = sum + load(idx) * weight;
            weight_sum = weight_sum + weight;
        }
    }

    store((ch * params.dst_h + y_dst) * params.dst_w + x_dst, encode(sum / weight_sum));
}
