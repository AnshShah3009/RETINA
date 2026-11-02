// Threshold on an f32 tensor, bit-identical to the CPU backend.
//
// Values are only compared and selected, never computed, so the shader works
// on bit patterns: comparing floats would let the device flush subnormal
// inputs and thresholds to zero (the RTX 5070 Ti on Vulkan does, so
// `0 > -1e-45` was false), and WGSL leaves `min` with a NaN operand to the
// implementation. `order_key` maps non-NaN bits to integers in value order,
// as `threshold_bf16.wgsl` does for bf16.
//
// The CPU semantics reproduced here, OpenCV's, with `v > t` false when
// either is NaN (`v` the input, `t` the threshold, `m` the max value):
//   Binary     v > t ? m : +0        BinaryInv  v > t ? +0 : m
//   Trunc      v > t ? t : v
//   ToZero     v > t ? v : +0        ToZeroInv  v > t ? +0 : v
// Every result is one of v, t, m or +0, and it is written back with its
// original bits, NaN payloads included.

// Same layout as the host's `ThresholdParams<f32>`. The threshold and max
// value are read as raw u32 bits: no float operation ever touches them.
struct Params {
    width: u32,
    height: u32,
    thresh_bits: u32,
    max_bits: u32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan(bits: u32) -> bool {
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

// Non-NaN f32 bits -> u32 in the same order as the values, -0 and +0 equal.
// Negative values count down from 0x80000000, positive ones up; the extremes
// (-inf, +inf) land on 0x00800000 and 0xFF800000, so nothing wraps.
fn order_key(bits: u32) -> u32 {
    if ((bits & 0x80000000u) != 0u) {
        return 0x80000000u - (bits & 0x7FFFFFFFu);
    }
    return 0x80000000u + bits;
}

fn apply_thresh(v: u32) -> u32 {
    let t = params.thresh_bits;
    let m = params.max_bits;
    let above = !is_nan(v) && !is_nan(t) && order_key(v) > order_key(t);
    switch params.typ {
        case 0u: { return select(0u, m, above); }      // Binary
        case 1u: { return select(m, 0u, above); }      // BinaryInv
        case 2u: { return select(v, t, above); }       // Trunc
        case 3u: { return select(0u, v, above); }      // ToZero
        case 4u: { return select(v, 0u, above); }      // ToZeroInv
        default: { return 0u; }
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // 1-D range dispatched as a 2-D grid so no axis exceeds 65535
    // workgroups (see gpu_kernels::dispatch_1d); surplus tail invocations
    // fall out at the bounds check below.
    let idx = global_id.x + global_id.y * num_wg.x * 64u;
    if (idx >= params.len) {
        return;
    }
    // Loaded and stored as raw bits: a load or store is not arithmetic, so
    // subnormals and NaN payloads pass through unchanged.
    output_data[idx] = bitcast<f32>(apply_thresh(bitcast<u32>(input_data[idx])));
}
