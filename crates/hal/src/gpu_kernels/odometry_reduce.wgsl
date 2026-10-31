// Parallel Reduction for Odometry linear system
// Sums 27-element vectors (21 JTJ upper triangle + 6 JTr) across workgroups.

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> num_elements: u32;

var<workgroup> shared_data: array<array<f32, 27>, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Workgroups are spread over a 2-D grid so neither axis exceeds 65535
    // (see gpu_kernels::dispatch_groups_1d). Surplus tail workgroups own no
    // input and have no output slot; the test is workgroup-uniform, so
    // returning before the barriers is allowed.
    let wg = group_id.x + group_id.y * num_wg.x;
    if (wg * 128u >= num_elements) {
        return;
    }
    let t_id = local_id.x;
    let i = wg * 128u + t_id;

    // Load into shared memory
    if (i < num_elements) {
        let in_base = i * 27u;
        for (var k = 0u; k < 27u; k++) {
            shared_data[t_id][k] = input_data[in_base + k];
        }
    } else {
        for (var k = 0u; k < 27u; k++) {
            shared_data[t_id][k] = 0.0;
        }
    }

    workgroupBarrier();

    // Tree reduction
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (t_id < s) {
            for (var k = 0u; k < 27u; k++) {
                shared_data[t_id][k] += shared_data[t_id + s][k];
            }
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (t_id == 0u) {
        let out_base = wg * 27u;
        for (var k = 0u; k < 27u; k++) {
            output_data[out_base + k] = shared_data[0][k];
        }
    }
}
