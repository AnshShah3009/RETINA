// Radix Sort Kernels (Global & Local)
// Implements LSD Radix Sort components for global sorting of large arrays.

// ----------------------------------------------------------------------------
// Common Bindings
// ----------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> input_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> histograms: array<u32>; // [num_workgroups * 256]
@group(0) @binding(3) var<uniform> params: SortParams;

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
    // Layout: [radix 0: wg0, wg1...][radix 1: wg0, wg1...] to match prefix_sum expectations?
    // Actually, for prefix_sum to work easily, we usually want a flat array.
    // But for Scatter, we need: global_offset = scanned_hist[radix * num_wgs + wid]
    // So let's write in column-major order: index = radix * num_workgroups + wid
    
    // Note: atomicLoad is not available in WGSL for non-atomic arrays? 
    // We loaded into 'atomic<u32>', so we use atomicLoad.
    let count = atomicLoad(&local_hist[tid]);
    
    // We export all 256 bins.
    // Since workgroup size is 256, each thread writes one bin for this workgroup.
    let output_idx = tid * params.num_workgroups + wid;
    histograms[output_idx] = count;
}

// ----------------------------------------------------------------------------
// Kernel 2: Scatter
// Reorders keys based on scanned histograms.
// ----------------------------------------------------------------------------
// We need the *scanned* histograms here.
// Re-using binding 2 'histograms' as read-only for scatter.

var<workgroup> local_prefix: array<u32, 256>; // Exclusive scan of local counts

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wid = group_id.x;
    
    // 1. Load data and compute local histogram (again) 
    // We re-compute local counts to know *local* rank/offset.
    
    // Init temp storage
    local_prefix[tid] = 0u;
    workgroupBarrier();
    
    var key = 0u;
    var radix = 0u;
    var valid = false;
    
    if (gid < params.num_elements) {
        key = input_keys[gid];
        radix = (key >> params.shift) & 0xFFu;
        // In a naive implementation, we'd use atomics for local offset.
        // But for determinism and speed, we can do a local scan or multipass.
        // Let's use a simpler approach: atomic in shared memory for local rank.
        // Re-use the atomic array from histogram? WGSL doesn't allow aliasing types easily.
        // We need a separate variable if types differ.
        // Let's use `local_hist` (atomic) defined above.
    }
    
    // Reset local hist for use as a counter
    atomicStore(&local_hist[tid], 0u);
    workgroupBarrier();
    
    // We need a local exclusive scan of counts *per radix*? No, that's hard in one pass.
    // Standard approach:
    // 1. Read global offset for my radix from scanned histogram buffer.
    //    global_base = histograms[radix * num_workgroups + wid]
    // 2. Determine local offset.
    
    // To determine local offset efficiently without full scan:
    // We can use the standard "scan local histogram" approach but that's complex for 256 bins.
    // Simpler: iterate bits or use atomic add for local rank.
    // Atomic add gives unique index.
    
    var local_offset = 0u;
    if (gid < params.num_elements) {
        local_offset = atomicAdd(&local_hist[radix], 1u);
        valid = true;
    }
    workgroupBarrier();
    
    // Now `local_hist[r]` contains the total count of radix `r` in this workgroup.
    // We need to convert `local_hist` to a local prefix sum to calculate start offsets?
    // No, we have the global base.
    // But we need to know where this workgroup starts writing for radix `r`.
    // The global scanned histogram tells us exactly that!
    // scanned_hist[radix * num_wgs + wid] IS the global start index for this WG's radix `r`.
    
    if (valid) {
        let global_base = histograms[radix * params.num_workgroups + wid];
        let final_addr = global_base + local_offset;
        output_keys[final_addr] = key;
    }
}

// ----------------------------------------------------------------------------
// Kernel 3: Local Sort (Workgroup Only)
// Sorts up to 256 elements. Kept for small arrays.
// ----------------------------------------------------------------------------
var<workgroup> local_data: array<u32, 256>;
var<workgroup> temp_data: array<u32, 256>;
var<workgroup> prefix_sum_arr: array<u32, 256>;

@compute @workgroup_size(256)
fn main_local(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // ... (Previous implementation)
    // For brevity in this file update, I'm omitting the full body if I can't overwrite partial.
    // But I must overwrite full file. I'll copy the previous logic here.
    
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Only supports 1 WG
    if (gid < params.num_elements) {
        local_data[tid] = input_keys[gid];
    } else {
        local_data[tid] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    
    for (var bit = 0u; bit < 32u; bit++) {
        let val = local_data[tid];
        let bit_val = (val >> bit) & 1u;
        
        prefix_sum_arr[tid] = 1u - bit_val;
        workgroupBarrier();
        
        for (var offset = 1u; offset < 256u; offset <<= 1u) {
            var sum = prefix_sum_arr[tid];
            if (tid >= offset) {
                sum += prefix_sum_arr[tid - offset];
            }
            workgroupBarrier();
            prefix_sum_arr[tid] = sum;
            workgroupBarrier();
        }
        
        let total_zeros = prefix_sum_arr[255];
        var dest_idx = 0u;
        if (bit_val == 0u) {
            dest_idx = prefix_sum_arr[tid] - 1u;
        } else {
            dest_idx = total_zeros + (tid - prefix_sum_arr[tid]);
        }
        
        temp_data[dest_idx] = val;
        workgroupBarrier();
        local_data[tid] = temp_data[tid];
        workgroupBarrier();
    }
    
    if (gid < params.num_elements) {
        output_keys[gid] = local_data[tid];
    }
}
