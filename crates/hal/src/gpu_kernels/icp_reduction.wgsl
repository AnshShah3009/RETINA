// ICP Reduction Shader
// Computes sum of squared distances and inlier count on GPU.

@group(0) @binding(0) var<storage, read> input_data: array<vec4<f32>>; // [src_idx, tgt_idx, dist_sq, valid]
@group(0) @binding(1) var<storage, read_write> output_data: array<vec2<f32>>; // [sum_dist_sq, count]
@group(0) @binding(2) var<uniform> num_elements: u32;

var<workgroup> shared_data: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // 1. Initial Load
    if (gid < num_elements) {
        let entry = input_data[gid];
        let valid = entry.w; // 1.0 if valid, 0.0 if not
        shared_data[tid] = vec2<f32>(entry.z * valid, valid);
    } else {
        shared_data[tid] = vec2<f32>(0.0, 0.0);
    }
    
    workgroupBarrier();
    
    // 2. Parallel Reduction in shared memory
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
    }
    
    // 3. Output results for this workgroup
    if (tid == 0u) {
        output_data[group_id.x] = shared_data[0];
    }
}
