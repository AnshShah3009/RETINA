// Parallel Reduction for ICP linear system
// Sums 27-element vectors across blocks.

struct Params {
    num_elements: u32, // Total pixels in this pass
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<array<f32, 27>, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let t_id = local_id.x;
    let i = global_id.x;

    // Load into shared memory
    if (i < params.num_elements) {
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
        let out_base = group_id.x * 27u;
        for (var k = 0u; k < 27u; k++) {
            output_data[out_base + k] = shared_data[0][k];
        }
    }
}
