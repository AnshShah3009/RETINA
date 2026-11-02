// Threshold on an f16 tensor, in native f16.
//
// Needs `wgpu::Features::SHADER_F16`. `gpu_kernels::threshold` selects this
// shader only when the device was created with it
// (`GpuContext::supports_f16`) and `threshold_f16_packed.wgsl` otherwise.
// Both give the same bits.
enable f16;

// Same layout as `threshold_f32.wgsl`. The host widens the f16 threshold and
// max value to f32, which is exact, so `f16(...)` below recovers them exactly.
// (This used to declare them `u32` and `bitcast<f16>` them, which is not
// valid: a bitcast cannot change the size of its operand.)
struct Params {
    width: u32,
    height: u32,
    thresh: f32,
    max_value: f32,
    typ: u32, // 0: Binary, 1: BinaryInv, 2: Trunc, 3: ToZero, 4: ToZeroInv
    len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f16>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

// The comparisons are the CPU backend's scalar path, written out: Trunc is
// `if v > t { t } else { v }` rather than `min`, which WGSL leaves
// unspecified for NaN and for signed zeros.
fn apply_thresh(val: f16) -> f16 {
    let t = f16(params.thresh);
    let m = f16(params.max_value);
    let above = val > t;
    switch params.typ {
        case 0u: { return select(0.0h, m, above); }     // Binary
        case 1u: { return select(m, 0.0h, above); }     // BinaryInv
        case 2u: { return select(val, t, above); }      // Trunc
        case 3u: { return select(0.0h, val, above); }   // ToZero
        case 4u: { return select(val, 0.0h, above); }   // ToZeroInv
        default: { return 0.0h; }
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

    output_data[idx] = apply_thresh(input_data[idx]);
}
