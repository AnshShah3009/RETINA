// Order-preserving stream compaction of an f32 score map into keypoints, on
// the GPU. Replaces reading the whole per-pixel count buffer back, scanning it
// on the CPU and uploading the offsets again (8 MB each way at 1080p).
//
// The map is cut into blocks of BLOCK = 1024 pixels, one workgroup each, four
// consecutive pixels per invocation:
//   1. `block_counts`: blocks[b] = number of pixels with score > 0 in block b.
//   2. `scan_blocks` (one workgroup): blocks[b] becomes the exclusive prefix
//      sum, and blocks[num_blocks] the total.
//   3. `scatter`: each keypoint is written at blocks[b] + its rank in the
//      block, so keypoints come out in pixel (row-major) order, exactly the
//      order and indices of the CPU scan this replaces.

const WG: u32 = 256u;
const PER_THREAD: u32 = 4u;
const BLOCK: u32 = 1024u;

struct Params {
    width: u32,
    height: u32,
    num_elements: u32,
    num_blocks: u32,
}

struct Keypoint {
    x: f32,
    y: f32,
    size: f32,
    angle: f32,
    response: f32,
    octave: i32,
    class_id: i32,
    padding: i32,
}

var<workgroup> scan_buf: array<u32, 256>;

// Inclusive Hillis-Steele scan of `v` over the workgroup. Must be reached in
// uniform control flow.
fn workgroup_inclusive_scan(li: u32, v: u32) -> u32 {
    scan_buf[li] = v;
    workgroupBarrier();
    for (var off = 1u; off < WG; off = off * 2u) {
        var add = 0u;
        if (li >= off) {
            add = scan_buf[li - off];
        }
        workgroupBarrier();
        scan_buf[li] = scan_buf[li] + add;
        workgroupBarrier();
    }
    return scan_buf[li];
}

fn is_point(score: f32) -> u32 {
    return select(0u, 1u, score > 0.0);
}

// ---- 1. per-block counts --------------------------------------------------

@group(0) @binding(0) var<storage, read> score_map: array<f32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn block_counts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // 2-D grid of blocks (gpu_kernels::dispatch_groups_1d). `block` is
    // uniform over the workgroup, so returning here keeps the barriers below
    // in uniform control flow.
    let block = wid.x + wid.y * nwg.x;
    if (block >= params.num_blocks) {
        return;
    }
    let first = block * BLOCK + li * PER_THREAD;
    var count = 0u;
    for (var k = 0u; k < PER_THREAD; k++) {
        let i = first + k;
        if (i < params.num_elements) {
            count += is_point(score_map[i]);
        }
    }
    let total = workgroup_inclusive_scan(li, count);
    if (li == WG - 1u) {
        block_sums[block] = total;
    }
}

// ---- 2. exclusive scan of the block counts ---------------------------------

@group(0) @binding(0) var<storage, read_write> block_offsets: array<u32>;
@group(0) @binding(1) var<uniform> scan_params: Params;

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) li: u32) {
    let n = scan_params.num_blocks;
    let per = (n + WG - 1u) / WG;
    let begin = min(li * per, n);
    let end = min(begin + per, n);
    var sum = 0u;
    for (var b = begin; b < end; b++) {
        sum += block_offsets[b];
    }
    let inclusive = workgroup_inclusive_scan(li, sum);
    var running = inclusive - sum;
    for (var b = begin; b < end; b++) {
        let c = block_offsets[b];
        block_offsets[b] = running;
        running += c;
    }
    if (li == WG - 1u) {
        block_offsets[n] = inclusive;
    }
}

// ---- 3. scatter ------------------------------------------------------------

@group(0) @binding(0) var<storage, read> scatter_scores: array<f32>;
@group(0) @binding(1) var<storage, read> scatter_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_keypoints: array<Keypoint>;
@group(0) @binding(3) var<uniform> scatter_params: Params;

@compute @workgroup_size(256)
fn scatter(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let block = wid.x + wid.y * nwg.x;
    if (block >= scatter_params.num_blocks) {
        return;
    }
    let n = scatter_params.num_elements;
    let first = block * BLOCK + li * PER_THREAD;
    var scores: array<f32, 4>;
    var count = 0u;
    for (var k = 0u; k < PER_THREAD; k++) {
        let i = first + k;
        var s = 0.0;
        if (i < n) {
            s = scatter_scores[i];
        }
        scores[k] = s;
        count += is_point(s);
    }
    let inclusive = workgroup_inclusive_scan(li, count);
    var dst = scatter_offsets[block] + inclusive - count;
    // The host never passes width 0 (it returns early for an empty map);
    // the max keeps the divisor nonzero for any uniform
    // (docs/wgsl_int_div_audit.md).
    let w = max(scatter_params.width, 1u);
    for (var k = 0u; k < PER_THREAD; k++) {
        let s = scores[k];
        if (is_point(s) == 1u) {
            let i = first + k;
            var kp: Keypoint;
            kp.x = f32(i % w);
            kp.y = f32(i / w);
            kp.size = 7.0; // Default FAST size
            kp.angle = -1.0;
            kp.response = s;
            kp.octave = 0;
            kp.class_id = 0;
            kp.padding = 0;
            out_keypoints[dst] = kp;
            dst += 1u;
        }
    }
}
