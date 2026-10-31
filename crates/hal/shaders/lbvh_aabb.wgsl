// LBVH build, phase 2: compute leaf AABBs and propagate them up the tree.
//
// Split out from the combined `lbvh_build.wgsl`. This phase genuinely needs four
// storage buffers plus a uniform, which is exactly at the device limit
// (`max_storage_buffers_per_shader_stage` = 4, from
// `wgpu::Limits::downlevel_defaults()`), so it cannot be merged back into a
// module whose entry points also reach four - not because layouts are derived
// per module, but because then those entry points would reach more than four.
// See that file for the details.

// Node layout (48 bytes = 12 words, same as `LbvhNode` in lbvh.rs and the
// kNN shader): parent, left, right, padding (i32), min_bound (vec4<f32>),
// max_bound (vec4<f32>). This phase views it as words and touches it ONLY
// through atomics. The bottom-up pass has one invocation read bounds that a
// different workgroup wrote moments earlier; plain storage loads give no
// cross-workgroup visibility guarantee (on an RTX 5070 Ti stale bounds showed
// up as non-deterministic wrong kNN results in ~1 run in 5), while atomic
// accesses to one location are coherent device-wide.
const NODE_WORDS: u32 = 12u;
const W_PARENT: u32 = 0u;
const W_LEFT: u32 = 1u;
const W_RIGHT: u32 = 2u;
const W_MIN: u32 = 4u;
const W_MAX: u32 = 8u;

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> node_counters: array<atomic<u32>>;

struct Params {
    num_elements: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(4) var<uniform> params: Params;

fn load_i32(node: u32, word: u32) -> i32 {
    return bitcast<i32>(atomicLoad(&nodes[node * NODE_WORDS + word]));
}

fn store_i32(node: u32, word: u32, v: i32) {
    atomicStore(&nodes[node * NODE_WORDS + word], bitcast<u32>(v));
}

fn load_vec3(node: u32, word: u32) -> vec3<f32> {
    let base = node * NODE_WORDS + word;
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&nodes[base])),
        bitcast<f32>(atomicLoad(&nodes[base + 1u])),
        bitcast<f32>(atomicLoad(&nodes[base + 2u])),
    );
}

fn store_vec3(node: u32, word: u32, v: vec3<f32>) {
    let base = node * NODE_WORDS + word;
    atomicStore(&nodes[base], bitcast<u32>(v.x));
    atomicStore(&nodes[base + 1u], bitcast<u32>(v.y));
    atomicStore(&nodes[base + 2u], bitcast<u32>(v.z));
    atomicStore(&nodes[base + 3u], 0u);
}

@compute @workgroup_size(256)
fn compute_aabbs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // 1-D range dispatched as a 2-D grid so no axis exceeds 65535
    // workgroups (see gpu_kernels::dispatch_1d).
    let leaf_idx = global_id.x + global_id.y * num_wg.x * 256u;
    if (leaf_idx >= params.num_elements) {
        return;
    }

    let leaf_offset = params.num_elements - 1u;
    var child = leaf_offset + leaf_idx;

    // 1. Initialize leaf AABB. The bounds travel up in registers; each node's
    // bounds are published before the counter increment that lets the
    // sibling's invocation continue past it.
    let p = points[sorted_indices[leaf_idx]].xyz;
    var bmin = p;
    var bmax = p;
    store_vec3(child, W_MIN, bmin);
    store_vec3(child, W_MAX, bmax);
    store_i32(child, W_LEFT, -1); // Mark as leaf
    store_i32(child, W_RIGHT, -1);

    // 2. Propagate up the tree. Depth of a binary tree over n leaves is at
    // most n - 1; the cap only stops a malformed (cyclic) tree from looping
    // the GPU forever.
    var curr = load_i32(child, W_PARENT);
    var depth = 0u;
    while (curr != -1 && depth < params.num_elements) {
        depth = depth + 1u;
        let node = u32(curr);
        let count = atomicAdd(&node_counters[node], 1u);
        if (count == 0u) {
            // First child to reach this node; the second one finishes it.
            return;
        }

        // Second child: merge the sibling's published bounds into ours.
        let l = load_i32(node, W_LEFT);
        let r = load_i32(node, W_RIGHT);
        let sibling = u32(select(l, r, u32(l) == child));
        bmin = min(bmin, load_vec3(sibling, W_MIN));
        bmax = max(bmax, load_vec3(sibling, W_MAX));
        store_vec3(node, W_MIN, bmin);
        store_vec3(node, W_MAX, bmax);

        child = node;
        curr = load_i32(node, W_PARENT);
    }
}
