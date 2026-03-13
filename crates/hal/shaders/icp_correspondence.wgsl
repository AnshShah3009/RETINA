struct Params {
    num_src: u32,
    num_tgt: u32,
    max_dist_sq: f32,
}

@group(0) @binding(0) var<storage, read> src_points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tgt_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> correspondences: array<vec4<f32>>; // [src_idx, tgt_idx, dist_sq, valid]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let src_idx = global_id.x;
    if (src_idx >= params.num_src) { return; }

    let p_s = src_points[src_idx].xyz;
    var min_dist_sq = 1e10;
    var best_tgt_idx = 0u;
    var found = false;

    // Brute force nearest neighbor
    for (var i = 0u; i < params.num_tgt; i++) {
        let p_t = tgt_points[i].xyz;
        let diff = p_s - p_t;
        let d2 = dot(diff, diff);
        if (d2 < min_dist_sq) {
            min_dist_sq = d2;
            best_tgt_idx = i;
            found = true;
        }
    }

    if (found && min_dist_sq <= params.max_dist_sq) {
        correspondences[src_idx] = vec4<f32>(f32(src_idx), f32(best_tgt_idx), min_dist_sq, 1.0);
    } else {
        correspondences[src_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
