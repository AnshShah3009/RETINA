// Parallel Reduction Shader
// Performs parallel reduction (sum, min, max) on GPU

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> num_elements: u32;
@group(0) @binding(3) var<uniform> reduction_type: u32; // 0=sum, 1=min, 2=max

var<workgroup> shared_data: array<f32, 256>;

fn reduce_op(a: f32, b: f32) -> f32 {
    switch(reduction_type) {
        case 0u: { return a + b; } // Sum
        case 1u: { return min(a, b); } // Min
        case 2u: { return max(a, b); } // Max
        default: { return a + b; }
    }
}

fn get_identity() -> f32 {
    switch(reduction_type) {
        case 0u: { return 0.0; } // Sum identity
        case 1u: { return 3.402823e+38; } // Min identity (f32 max)
        case 2u: { return -3.402823e+38; } // Max identity (f32 min)
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    if (gid < num_elements) {
        shared_data[tid] = input_data[gid];
    } else {
        shared_data[tid] = get_identity();
    }
    
    workgroupBarrier();
    
    // Tree-based reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = reduce_op(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
    }
    
    // Write result for this workgroup
    if (tid == 0u) {
        output_data[group_id.x] = shared_data[0];
    }
}
