// 3x3 non-maximum suppression of a FAST score map held as f16 or bf16, any
// device. The CPU backend's test, on the scores' bit patterns (see `key16`);
// a kept score is written back as its original bits.
//
// Replaces `fast_nms_f16.wgsl` (needed SHADER_F16) and `fast_nms_bf16.wgsl`
// (two invocations did a non-atomic read-modify-write of the same word);
// neither had a caller, as `fast_detect` returned NotSupported for 16-bit
// suppression.

struct Params {
    width: u32,
    height: u32,
    threshold: f32, // unused here; the FAST params buffer is shared
    format: u32,    // 0: f16, 1: bf16
}

@group(0) @binding(0) var<storage, read> score_map: array<u32>; // two values per word
@group(0) @binding(1) var<storage, read_write> suppressed_map: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn bits_at(idx: u32) -> u32 {
    return (score_map[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu;
}

// Scores are compared on their 16-bit patterns rather than widened: a bf16
// score below 2^-126 widens to an f32 subnormal, which the device flushes in
// float comparisons, so two distinct tiny scores compared equal.
fn is_nan16(b: u32) -> bool {
    if (params.format == 1u) {
        return (b & 0x7FFFu) > 0x7F80u;
    }
    return (b & 0x7FFFu) > 0x7C00u;
}

// Non-NaN 16-bit float bits -> u32 in value order, -0 equal to +0.
fn key16(b: u32) -> u32 {
    if ((b & 0x8000u) != 0u) {
        return 0x8000u - (b & 0x7FFFu);
    }
    return 0x8000u + b;
}

// Bits of the score at (x, y); outside the image a score of 0.
fn get_score(x: i32, y: i32) -> u32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0u;
    }
    return bits_at(u32(y) * params.width + u32(x));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }

    let idx = u32(y) * params.width + u32(x);
    let s_bits = bits_at(idx);
    // A zero score stays 0 (the buffer's initial value). A NaN one is kept,
    // as on the CPU, where every comparison with it is false.
    if ((s_bits & 0x7FFFu) == 0u) {
        return;
    }
    let s_nan = is_nan16(s_bits);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let n_bits = get_score(x + dx, y + dy);
            if (s_nan || is_nan16(n_bits)) { continue; }
            let n = key16(n_bits);
            let c = key16(s_bits);
            if (n > c || (n == c && (dy > 0 || (dy == 0 && dx > 0)))) {
                return;
            }
        }
    }
    atomicOr(&suppressed_map[idx >> 1u], s_bits << ((idx & 1u) << 4u));
}
