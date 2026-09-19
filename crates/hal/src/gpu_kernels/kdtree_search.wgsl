// kdtree_search.wgsl
// GPU KD-Tree search kernel (NN and Radius search)

struct KdNode {
    left: i32,
    right: i32,
    axis: u32,
    point_idx: u32,
    split_val: f32,
    _padding: f32,
};

@group(0) @binding(0) var<storage, read> nodes: array<KdNode>;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>; // [x,y,z,1]
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>; // Closest point index
@group(0) @binding(4) var<storage, read_write> dists: array<f32>;   // Squared distance

@compute @workgroup_size(256)
fn kdtree_nn_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_idx = global_id.x;
    if (query_idx >= arrayLength(&queries)) {
        return;
    }
    
    let query = queries[query_idx].xyz;
    var best_dist = 1e38;
    var best_idx = 0xffffffffu;
    
    // Stackless traversal or fixed-depth stack
    // For simplicity, we use a simple iterative approach with a small fixed stack
    var stack: array<i32, 32>;
    var stack_ptr = 0;
    stack[stack_ptr] = 0; // Root
    stack_ptr = stack_ptr + 1;
    
    while (stack_ptr > 0) {
        stack_ptr = stack_ptr - 1;
        let curr_idx = stack[stack_ptr];
        if (curr_idx < 0) { continue; }
        
        let node = nodes[curr_idx];
        let p = points[node.point_idx].xyz;
        let d2 = dot(query - p, query - p);
        
        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = node.point_idx;
        }
        
        // Explore children
        let diff = query[node.axis] - node.split_val;
        let near = select(node.right, node.left, diff <= 0.0);
        let far = select(node.left, node.right, diff <= 0.0);
        
        // Always push far child if it could contain a closer point
        if (diff * diff < best_dist) {
            if (far >= 0 && stack_ptr < 32) {
                stack[stack_ptr] = far;
                stack_ptr = stack_ptr + 1;
            }
        }
        
        // Always push near child
        if (near >= 0 && stack_ptr < 32) {
            stack[stack_ptr] = near;
            stack_ptr = stack_ptr + 1;
        }
    }
    
    results[query_idx] = best_idx;
    dists[query_idx] = best_dist;
}
