// Radix Sort Kernels (Keys Only)
// Implements LSD Radix Sort components for global sorting of large arrays.

@group(0) @binding(0) var<storage, read> input_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(3) var<uniform> params: SortParams;

struct SortParams {
    num_elements: u32,
    shift: u32,
    num_workgroups: u32,
    padding: u32,
}

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wid = group_id.x;
    
    atomicStore(&local_hist[tid], 0u);
    workgroupBarrier();
    
    if (gid < params.num_elements) {
        let key = input_keys[gid];
        let radix = (key >> params.shift) & 0xFFu;
        atomicAdd(&local_hist[radix], 1u);
    }
    workgroupBarrier();
    
    let count = atomicLoad(&local_hist[tid]);
    let output_idx = tid * params.num_workgroups + wid;
    histograms[output_idx] = count;
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wid = group_id.x;
    
    atomicStore(&local_hist[tid], 0u);
    workgroupBarrier();
    
    var key = 0u;
    var radix = 0u;
    var valid = false;
    
    if (gid < params.num_elements) {
        key = input_keys[gid];
        radix = (key >> params.shift) & 0xFFu;
        valid = true;
    }
    
    var local_offset = 0u;
    if (valid) {
        local_offset = atomicAdd(&local_hist[radix], 1u);
    }
    workgroupBarrier();
    
    if (valid) {
        let global_base = histograms[radix * params.num_workgroups + wid];
        let final_addr = global_base + local_offset;
        output_keys[final_addr] = key;
    }
}
