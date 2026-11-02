// 3x3 non-maximum suppression of a FAST score map held as f16 or bf16, any
// device. `fast_nms_f32.wgsl`'s test on the exactly widened scores; a kept
// score is written back as its original bits.
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

// f16 bits (low 16) -> f32, exact. As in the other `*_packed16.wgsl`.
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

fn decode(bits: u32) -> f32 {
    if (params.format == 1u) {
        return bitcast<f32>(bits << 16u);
    }
    return f16_to_f32(bits);
}

fn bits_at(idx: u32) -> u32 {
    return (score_map[idx >> 1u] >> ((idx & 1u) << 4u)) & 0xFFFFu;
}

fn get_score(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0.0;
    }
    return decode(bits_at(u32(y) * params.width + u32(x)));
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
    let s = decode(s_bits);
    if (!(s > 0.0)) {
        return; // suppressed: the buffer's initial 0
    }
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let neighbor_s = get_score(x + dx, y + dy);
            if (neighbor_s > s || (neighbor_s == s && (dy > 0 || (dy == 0 && dx > 0)))) {
                return;
            }
        }
    }
    atomicOr(&suppressed_map[idx >> 1u], s_bits << ((idx & 1u) << 4u));
}
