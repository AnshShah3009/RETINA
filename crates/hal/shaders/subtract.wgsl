// Image subtraction kernel: output = a - b
// Works on flat arrays of f32

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // 1-D range dispatched as a 2-D grid so no axis exceeds 65535
    // workgroups (see gpu_kernels::dispatch_1d); surplus tail invocations
    // fall out at the bounds check below.
    let idx = gid.x + gid.y * num_wg.x * 256u;
    if (idx >= arrayLength(&a) || idx >= arrayLength(&b)) {
        return;
    }
    output_data[idx] = a[idx] - b[idx];
}
