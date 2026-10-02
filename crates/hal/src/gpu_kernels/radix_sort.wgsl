// Radix Sort Kernels (Global & Local)
// Implements LSD Radix Sort components for global sorting of large arrays.
//
// Storage-buffer budget: this device caps
// `max_storage_buffers_per_shader_stage` at 4 (from
// `wgpu::Limits::downlevel_defaults()`). Both entry points share this module,
// so `input_values` / `output_values` are packed into one interleaved
// `io_values: array<u32>` of `struct SortValue { key: u32, value: u32 }` - an
// 8-byte stride, no padding. `scatter` drops from five storage bindings to four;
// `histogram` was already at three and stays there (it never touches the value
// half at all).
//
// This module is LATENT: the live sort paths use `radix_sort_keys.wgsl` and
// `radix_sort_kv.wgsl`, which are three-storage and fine. The only reference
// here is the `ShaderLibrary` registry in `gpu_kernels/mod.rs`, which has no
// consumers. A host building these bind groups must now supply the packed
// `io_values` buffer rather than separate key/value buffers.

// ----------------------------------------------------------------------------
// Common Bindings
// ----------------------------------------------------------------------------
struct SortValue {
    key: u32,
    value: u32,
};

@group(0) @binding(0) var<storage, read> input_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> histograms: array<u32>; // [num_workgroups * 256]
@group(0) @binding(3) var<uniform> params: SortParams;
@group(0) @binding(4) var<storage, read_write> io_values: array<SortValue>;

struct SortParams {
    num_elements: u32,
    shift: u32,      // Current bit shift (0, 8, 16, 24)
    num_workgroups: u32,
    padding: u32,
}

// ----------------------------------------------------------------------------
// Kernel 1: Histogram
// Counts occurrences of each radix (8-bit digit) per workgroup.
// ----------------------------------------------------------------------------
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
    
    // Initialize local histogram
    atomicStore(&local_hist[tid], 0u);
    workgroupBarrier();
    
    // Count keys
    if (gid < params.num_elements) {
        let key = input_keys[gid];
        let radix = (key >> params.shift) & 0xFFu;
        atomicAdd(&local_hist[radix], 1u);
    }
    workgroupBarrier();
    
    // Write to global histogram buffer
    let count = atomicLoad(&local_hist[tid]);
    let output_idx = tid * params.num_workgroups + wid;
    histograms[output_idx] = count;
}

// ----------------------------------------------------------------------------
// Kernel 2: Scatter
// Reorders keys based on scanned histograms.
// ----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wid = group_id.x;
    
    // Reset local hist for use as a counter
    atomicStore(&local_hist[tid], 0u);
    workgroupBarrier();
    
    var key = 0u;
    var val = 0u;
    var radix = 0u;
    var valid = false;
    
    if (gid < params.num_elements) {
        key = input_keys[gid];
        val = io_values[gid].value;
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
        io_values[final_addr].key = key;
        io_values[final_addr].value = val;
    }
}
