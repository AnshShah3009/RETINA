// Point Cloud kNN + Covariance Computation
// Performs k-nearest neighbor search on an LBVH-accelerated point cloud.
// Outputs covariance matrices for PCA.

struct LbvhNode {
    parent: i32,
    left: i32,
    right: i32,
    padding: i32,
    min_bound: vec4<f32>,
    max_bound: vec4<f32>,
}

struct Params {
    num_points: u32,
    k_neighbors: u32,
    window_size: u32, 
    padding: u32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> covs: array<vec4<f32>>; // 2 vec4s per point
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> nodes: array<LbvhNode>;

fn distance_sq_to_aabb(p: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> f32 {
    let dx = max(0.0, max(b_min.x - p.x, p.x - b_max.x));
    let dy = max(0.0, max(b_min.y - p.y, p.y - b_max.y));
    let dz = max(0.0, max(b_min.z - p.z, p.z - b_max.z));
    return dx * dx + dy * dy + dz * dz;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let center_idx = sorted_indices[idx];
    let center = points[center_idx].xyz;
    let k = params.k_neighbors;

    var neighbor_indices: array<u32, 32>; // Max k=32
    var dists: array<f32, 32>;
    
    // Initialize
    for (var i = 0u; i < k; i = i + 1u) {
        dists[i] = 1e20;
        neighbor_indices[i] = 0xFFFFFFFFu;
    }

    // Tree Traversal
    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u; // Root is internal node 0
    stack_ptr = stack_ptr + 1u;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];
        let node = nodes[node_idx];

        // Pruning
        let d2_aabb = distance_sq_to_aabb(center, node.min_bound.xyz, node.max_bound.xyz);
        if (d2_aabb >= dists[k - 1u]) {
            continue;
        }

        // Internal Node
        if (node.left != -1) {
            let node_l = u32(node.left);
            let node_r = u32(node.right);
            
            // Push children to stack
            // Heuristic: push farther child first
            let d2_l = distance_sq_to_aabb(center, nodes[node_l].min_bound.xyz, nodes[node_l].max_bound.xyz);
            let d2_r = distance_sq_to_aabb(center, nodes[node_r].min_bound.xyz, nodes[node_r].max_bound.xyz);
            
            if (d2_l > d2_r) {
                if (stack_ptr < 32u) { stack[stack_ptr] = node_l; stack_ptr = stack_ptr + 1u; }
                if (stack_ptr < 32u) { stack[stack_ptr] = node_r; stack_ptr = stack_ptr + 1u; }
            } else {
                if (stack_ptr < 32u) { stack[stack_ptr] = node_r; stack_ptr = stack_ptr + 1u; }
                if (stack_ptr < 32u) { stack[stack_ptr] = node_l; stack_ptr = stack_ptr + 1u; }
            }
        } else {
            // Leaf Node
            let point_idx = sorted_indices[node_idx - (params.num_points - 1u)];
            if (point_idx == center_idx) { continue; }
            
            let other = points[point_idx].xyz;
            let d2 = dot(center - other, center - other);
            
            if (d2 < dists[k - 1u]) {
                var m = k - 1u;
                while (m > 0u && d2 < dists[m - 1u]) {
                    dists[m] = dists[m - 1u];
                    neighbor_indices[m] = neighbor_indices[m - 1u];
                    m = m - 1u;
                }
                dists[m] = d2;
                neighbor_indices[m] = point_idx;
            }
        }
    }

    // Compute Covariance
    var count = 0u;
    var centroid = vec3<f32>(0.0);
    for (var i = 0u; i < k; i = i + 1u) {
        if (neighbor_indices[i] != 0xFFFFFFFFu) {
            centroid += points[neighbor_indices[i]].xyz;
            count += 1u;
        }
    }
    
    if (count < 3u) {
        covs[center_idx * 2u] = vec4<f32>(0.0);
        covs[center_idx * 2u + 1u] = vec4<f32>(0.0);
        return;
    }
    
    centroid /= f32(count);

    var cxx = 0.0; var cxy = 0.0; var cxz = 0.0;
    var cyy = 0.0; var cyz = 0.0; var czz = 0.0;

    for (var i = 0u; i < k; i = i + 1u) {
        if (neighbor_indices[i] != 0xFFFFFFFFu) {
            let d = points[neighbor_indices[i]].xyz - centroid;
            cxx += d.x * d.x;
            cxy += d.x * d.y;
            cxz += d.x * d.z;
            cyy += d.y * d.y;
            cyz += d.y * d.z;
            czz += d.z * d.z;
        }
    }

    let inv_n = 1.0 / f32(count);
    covs[center_idx * 2u] = vec4<f32>(cxx * inv_n, cxy * inv_n, cxz * inv_n, cyy * inv_n);
    covs[center_idx * 2u + 1u] = vec4<f32>(cyz * inv_n, czz * inv_n, 0.0, 0.0);
}
