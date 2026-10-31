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
// Digit of each tile element for the stable scatter; NO_DIGIT past the end.
var<workgroup> tile_digits: array<u32, 256>;
const NO_DIGIT: u32 = 256u;

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Workgroups are spread over a 2-D grid so neither axis exceeds 65535
    // (see gpu_kernels::dispatch_groups_1d). Surplus tail workgroups have no
    // histogram column; the test is workgroup-uniform, so returning before
    // the barriers is allowed.
    let wid = group_id.x + group_id.y * num_wg.x;
    if (wid >= params.num_workgroups) {
        return;
    }
    let tid = local_id.x;
    let gid = wid * 256u + tid;
    
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
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // Workgroups are spread over a 2-D grid so neither axis exceeds 65535
    // (see gpu_kernels::dispatch_groups_1d). Surplus tail workgroups have no
    // histogram column; the test is workgroup-uniform, so returning before
    // the barriers is allowed.
    let wid = group_id.x + group_id.y * num_wg.x;
    if (wid >= params.num_workgroups) {
        return;
    }
    let tid = local_id.x;
    let gid = wid * 256u + tid;

    var key = 0u;
    var radix = NO_DIGIT;
    if (gid < params.num_elements) {
        key = input_keys[gid];
        radix = (key >> params.shift) & 0xFFu;
    }
    tile_digits[tid] = radix;
    workgroupBarrier();

    // Stable rank: the number of earlier elements of this tile with the same
    // digit. LSD radix sort is only correct when every pass is stable - keys
    // equal in the current digit must keep the order earlier passes gave
    // them - so the rank must follow input order (an atomicAdd counter does
    // not; its order is unspecified).
    if (radix != NO_DIGIT) {
        var local_offset = 0u;
        for (var j = 0u; j < tid; j++) {
            local_offset += select(0u, 1u, tile_digits[j] == radix);
        }
        let global_base = histograms[radix * params.num_workgroups + wid];
        let final_addr = global_base + local_offset;
        output_keys[final_addr] = key;
    }
}
