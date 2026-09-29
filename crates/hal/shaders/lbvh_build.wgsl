// Karras (2012) "Thinking Parallel: Multi-threaded Tree Construction"
// Radix tree construction from sorted Morton codes.
//
// Phase 1 of the LBVH build. The AABB phase, which also needs the point cloud, is
// in `lbvh_aabb.wgsl`.
//
// It is a separate module because WGSL derives one bind group layout per
// *module*, not per entry point, and the device reports four storage buffers per
// stage. Holding all three phases together declared five, so a binding only
// `compute_aabbs` touched still counted against `init_nodes` and
// `build_radix_tree` and no pipeline in the file could be created: "Too many
// bindings of type StorageBuffers, limit is 4, count was 5".
//
// Neither entry point here touches `points`, `sorted_indices` or
// `node_counters`, so this module declares only the two it needs.

struct LbvhNode {
    parent: i32,
    left: i32,
    right: i32,
    padding: i32,
    min_bound: vec4<f32>,
    max_bound: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> morton_codes: array<u32>;
@group(0) @binding(1) var<storage, read_write> nodes: array<LbvhNode>; // size 2*N - 1

struct Params {
    num_elements: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(2) var<uniform> params: Params;

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
