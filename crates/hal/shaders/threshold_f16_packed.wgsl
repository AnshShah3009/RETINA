// Threshold on an f16 tensor without `SHADER_F16`.
//
// The fallback for `threshold_f16.wgsl` on a device that lacks the feature.
// The tensor is read as u32 words holding two f16 values each. Every value is
// widened to f32 with `unpack2x16float`, which is exact, and only compared
// there. The result is always one of the input value, the threshold, the max
// value or +0, so it is written back as the original 16 bits rather than
// re-rounded with `pack2x16float` (which is indeterminate for infinities).
// The output is bit-identical to the native f16 shader.

// Same layout as `threshold_f32.wgsl`, plus the f16 bits of the threshold
// and max value in `thresh_bits` / `max_bits` (low 16 bits).
struct Params {
    width: u32,
    height: u32,
    thresh: f32,
    max_value: f32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32, // f16 elements, not words
    thresh_bits: u32,
    max_bits: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two f16 per word
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// `bits` is one f16 in the low half of a u32; returns the result's bits.
fn apply_thresh(bits: u32) -> u32 {
    let val = unpack2x16float(bits).x;
    let above = val > params.thresh;
    let t = params.thresh_bits & 0xFFFFu;
    let m = params.max_bits & 0xFFFFu;
    switch params.typ {
        case 0u: { return select(0u, m, above); }      // Binary
        case 1u: { return select(m, 0u, above); }      // BinaryInv
        case 2u: { return select(bits, t, above); }    // Trunc
        case 3u: { return select(0u, bits, above); }   // ToZero
        case 4u: { return select(bits, 0u, above); }   // ToZeroInv
        default: { return 0u; }
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // One invocation per word, 1-D range dispatched as a 2-D grid (see
    // gpu_kernels::dispatch_1d).
    let word = global_id.x + global_id.y * num_wg.x * 64u;
    let first = word * 2u;
    if (first >= params.len) {
        return;
    }
    let packed = input_data[word];
    let lo = apply_thresh(packed & 0xFFFFu);
    // An odd length leaves the last word's high half outside the tensor.
    var hi = 0u;
    if (first + 1u < params.len) {
        hi = apply_thresh(packed >> 16u);
    }
    output_data[word] = lo | (hi << 16u);
}
