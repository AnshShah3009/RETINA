// Threshold on a bf16 tensor, any device.
//
// `threshold_f16_packed.wgsl` with the bf16 widening. The tensor is read as
// u32 words holding two bf16 values each. Values are only compared, on their
// bits (see `order_key`), and the result is always one of the input value, the
// threshold, the max value or +0, so it is written back as the original 16
// bits and never re-rounded.
//
// This shader used to compare the raw bf16 bit patterns as u32 (wrong for
// negative values) and read each element as a whole u32 word, and nothing
// called it: `threshold` returned NotSupported for bf16.

// Same layout as `threshold_f16_packed.wgsl`: the threshold and max value
// widened to f32 (unused here), plus their bf16 bits in `thresh_bits` /
// `max_bits` (low 16).
struct Params {
    width: u32,
    height: u32,
    thresh: f32,
    max_value: f32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32, // bf16 elements, not words
    thresh_bits: u32,
    max_bits: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // two bf16 per word
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn is_nan(bits: u32) -> bool {
    return (bits & 0x7FFFu) > 0x7F80u;
}

// Maps non-NaN bf16 bits to a u32 with the same order as the values, -0 and
// +0 equal. Comparing these rather than the widened f32 keeps the test exact
// for bf16 subnormals, which are f32 subnormals that a device may flush to 0.
fn order_key(bits: u32) -> u32 {
    if ((bits & 0x8000u) != 0u) {
        return 0x8000u - (bits & 0x7FFFu);
    }
    return 0x8000u + bits;
}

// `bits` is one bf16 in the low half of a u32; returns the result's bits.
fn apply_thresh(bits: u32) -> u32 {
    let t = params.thresh_bits & 0xFFFFu;
    // `val > thresh`, false when either is NaN, as on the CPU.
    let above = !is_nan(bits) && !is_nan(t) && order_key(bits) > order_key(t);
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
