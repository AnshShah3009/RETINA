// Radix Sort Kernels (Key-Value Packed)
// Implements LSD Radix Sort components for global sorting of large arrays.
// Optimized for hardware with low storage buffer limits (e.g. iGPUs).

struct KV {
    key: u32,
    val: u32,
}

@group(0) @binding(0) var<storage, read> input_kv: array<KV>;
@group(0) @binding(1) var<storage, read_write> output_kv: array<KV>;
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
        let key = input_kv[gid].key;
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
    
    var kv = KV(0u, 0u);
    var radix = 0u;
    var valid = false;
    
    if (gid < params.num_elements) {
        kv = input_kv[gid];
        radix = (kv.key >> params.shift) & 0xFFu;
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
        output_kv[final_addr] = kv;
    }
}

// ----------------------------------------------------------------------------
// Utilities for packing/unpacking
// ----------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read> in_keys: array<u32>;
@group(0) @binding(1) var<storage, read> in_vals: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_kv_pack: array<KV>;

@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.num_elements) { return; }
    out_kv_pack[gid] = KV(in_keys[gid], in_vals[gid]);
}

@group(0) @binding(0) var<storage, read> in_kv_unpack: array<KV>;
@group(0) @binding(1) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_vals: array<u32>;

@compute @workgroup_size(256)
fn unpack(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.num_elements) { return; }
    let kv = in_kv_unpack[gid];
    out_keys[gid] = kv.key;
    out_vals[gid] = kv.val;
}
