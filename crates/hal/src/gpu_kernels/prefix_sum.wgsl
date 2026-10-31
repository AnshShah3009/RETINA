// Global Prefix Sum (Scan) Shader
// Performs work-efficient parallel exclusive prefix sum on GPU.

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> num_elements: u32;

var<workgroup> temp: array<u32, 512 + 16>; // Extra space for bank conflict avoidance

// Kernel 1: Scan individual blocks of 512 elements and record block totals.
@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Workgroups are spread over a 2-D grid so neither axis exceeds 65535
    // (see gpu_kernels::dispatch_groups_1d). Surplus tail workgroups own no
    // block and have no block_sums slot; the test is workgroup-uniform, so
    // returning before the barriers is allowed.
    let wid = group_id.x + group_id.y * num_wg.x;
    if (wid * 512u >= num_elements) {
        return;
    }
    let thid = local_id.x;
    let n = 512u;

    // Load data into shared memory. Slot `ai` holds block element `ai` and
    // slot `bi` holds element `bi` - the tree below scans slots in order.
    // Loading `data[2 * gid]` / `data[2 * gid + 1]` here put element 2t in
    // slot t and 2t+1 in slot t+256, so the block was scanned in a shuffled
    // order and every output past index 0 was wrong.
    let ai = thid;
    let bi = thid + 256u;
    let src_a = wid * n + ai;
    let src_b = wid * n + bi;
    
    let bank_offset_a = ai >> 5u;
    let bank_offset_b = bi >> 5u;

    if (src_a < num_elements) {
        temp[ai + bank_offset_a] = data[src_a];
    } else {
        temp[ai + bank_offset_a] = 0u;
    }
    
    if (src_b < num_elements) {
        temp[bi + bank_offset_b] = data[src_b];
    } else {
        temp[bi + bank_offset_b] = 0u;
    }

    // Up-sweep
    var offset = 1u;
    for (var d = 256u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (thid < d) {
            let ai_up = offset * (2u * thid + 1u) - 1u;
            let bi_up = offset * (2u * thid + 2u) - 1u;
            let ao = ai_up + (ai_up >> 5u);
            let bo = bi_up + (bi_up >> 5u);
            temp[bo] += temp[ao];
        }
        offset *= 2u;
    }

    // Clear last element and save block sum
    if (thid == 0u) {
        let last_idx = n - 1u + ((n - 1u) >> 5u);
        block_sums[wid] = temp[last_idx];
        temp[last_idx] = 0u;
    }

    // Down-sweep
    for (var d = 1u; d <= 256u; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (thid < d) {
            let ai_down = offset * (2u * thid + 1u) - 1u;
            let bi_down = offset * (2u * thid + 2u) - 1u;
            let ao = ai_down + (ai_down >> 5u);
            let bo = bi_down + (bi_down >> 5u);
            
            let t = temp[ao];
            temp[ao] = temp[bo];
            temp[bo] += t;
        }
    }
    workgroupBarrier();

    // Write block-local prefix sum
    if (src_a < num_elements) {
        data[src_a] = temp[ai + bank_offset_a];
    }
    if (src_b < num_elements) {
        data[src_b] = temp[bi + bank_offset_b];
    }
}

// Kernel 2: Add scanned block offsets back to the per-block prefix sums.
@group(0) @binding(0) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> target_data: array<u32>;

@compute @workgroup_size(256)
fn add_offsets(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Same 2-D workgroup grid as scan_blocks.
    let wid = group_id.x + group_id.y * num_wg.x;
    if (wid * 512u >= num_elements) {
        return;
    }
    let gid = wid * 256u + local_id.x;
    let offset = block_offsets[wid];
    
    if (2u * gid < num_elements) {
        target_data[2u * gid] += offset;
    }
    if (2u * gid + 1u < num_elements) {
        target_data[2u * gid + 1u] += offset;
    }
}
