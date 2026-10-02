// LBVH build, phase 2: compute leaf AABBs and propagate them up the tree.
//
// Split out from the combined `lbvh_build.wgsl`. This phase genuinely needs four
// storage buffers plus a uniform, which is exactly at the device limit
// (`max_storage_buffers_per_shader_stage` = 4, from
// `wgpu::Limits::downlevel_defaults()`), so it cannot be merged back into a
// module whose entry points also reach four - not because layouts are derived
// per module, but because then those entry points would reach more than four.
// See that file for the details.

struct LbvhNode {
    parent: i32,
    left: i32,
    right: i32,
    padding: i32,
    min_bound: vec4<f32>,
    max_bound: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<LbvhNode>;
@group(0) @binding(3) var<storage, read_write> node_counters: array<atomic<u32>>;

struct Params {
    num_elements: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_aabbs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf_idx = i32(global_id.x);
    if (leaf_idx >= i32(params.num_elements)) {
        return;
    }

    let leaf_offset = i32(params.num_elements) - 1;
    let node_idx = leaf_offset + leaf_idx;
    
    // 1. Initialize leaf AABB
    let p_idx = sorted_indices[leaf_idx];
    let p = points[p_idx].xyz;
    nodes[node_idx].min_bound = vec4<f32>(p, 0.0);
    nodes[node_idx].max_bound = vec4<f32>(p, 0.0);
    nodes[node_idx].left = -1; // Mark as leaf
    nodes[node_idx].right = -1;

    // 2. Propagate up the tree
    var curr = nodes[node_idx].parent;
    while (curr != -1) {
        let count = atomicAdd(&node_counters[curr], 1u);
        if (count == 0u) {
            // First child to reach this node, terminate thread
            return;
        }
        
        // Second child to reach this node, compute AABB and continue up
        let l = nodes[curr].left;
        let r = nodes[curr].right;
        
        nodes[curr].min_bound = min(nodes[l].min_bound, nodes[r].min_bound);
        nodes[curr].max_bound = max(nodes[l].max_bound, nodes[r].max_bound);
        
        curr = nodes[curr].parent;
    }
}
