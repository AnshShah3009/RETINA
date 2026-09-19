// Karras (2012) "Thinking Parallel: Multi-threaded Tree Construction"
// Radix tree construction from sorted Morton codes.

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
@group(0) @binding(2) var<storage, read> morton_codes: array<u32>;
@group(0) @binding(3) var<storage, read_write> nodes: array<LbvhNode>; // size 2*N - 1
@group(0) @binding(4) var<storage, read_write> node_counters: array<atomic<u32>>; // size N - 1

struct Params {
    num_elements: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(5) var<uniform> params: Params;

// Length of common prefix between two Morton codes.
fn delta(i: i32, j: i32) -> i32 {
    if (j < 0 || j >= i32(params.num_elements)) {
        return -1;
    }
    
    let a = morton_codes[i];
    let b = morton_codes[j];
    
    if (a == b) {
        return 32 + i32(countLeadingZeros(u32(i ^ j)));
    }
    
    return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(256)
fn init_nodes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = i32(global_id.x);
    if (i >= i32(params.num_elements) * 2 - 1) {
        return;
    }
    nodes[i].parent = -1;
    nodes[i].left = -1;
    nodes[i].right = -1;
}

@compute @workgroup_size(256)
fn build_radix_tree(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = i32(global_id.x);
    if (i >= i32(params.num_elements) - 1) {
        return;
    }

    // Determine direction of the range (+1 or -1)
    let d = select(-1, 1, delta(i, i + 1) - delta(i, i - 1) > 0);

    // Compute upper bound for the length of the range
    let delta_min = delta(i, i - d);
    var l_max = 2;
    while (delta(i, i + l_max * d) > delta_min) {
        l_max *= 2;
    }

    // Find the other end using binary search
    var l = 0;
    var t = l_max / 2;
    while (t > 0) {
        if (delta(i, i + (l + t) * d) > delta_min) {
            l += t;
        }
        t /= 2;
    }
    let j = i + l * d;

    // Find the split position using binary search
    let delta_node = delta(i, j);
    var split = 0;
    var step = l;
    loop {
        step = (step + 1) / 2;
        let new_split = split + step;
        if (new_split < l) {
            if (delta(i, i + new_split * d) > delta_node) {
                split = new_split;
            }
        }
        if (step == 1) { break; }
    }
    let m = i + split * d + min(0, d);

    // Internal nodes are 0..N-2
    // Leaf nodes are N-1..2N-2
    let leaf_offset = i32(params.num_elements) - 1;
    
    var node_left = 0;
    if (min(i, j) == m) {
        node_left = leaf_offset + m;
    } else {
        node_left = m;
    }
    
    var node_right = 0;
    if (max(i, j) == m + 1) {
        node_right = leaf_offset + (m + 1);
    } else {
        node_right = m + 1;
    }

    nodes[i].left = node_left;
    nodes[i].right = node_right;
    nodes[node_left].parent = i;
    nodes[node_right].parent = i;
}

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
