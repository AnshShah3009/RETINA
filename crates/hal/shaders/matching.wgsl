struct Params {
    query_len: u32,
    train_len: u32,
    desc_size: u32, // bytes per descriptor
    ratio_threshold: f32,
}

@group(0) @binding(0) var<storage, read> query_data: array<u32>;
@group(0) @binding(1) var<storage, read> train_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> matches_out: array<vec4<f32>>; // [train_idx, distance, second_best, 0]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let q_idx = global_id.x;
    if (q_idx >= params.query_len) {
        return;
    }

    let u32_per_desc = params.desc_size / 4u;
    let q_base = q_idx * u32_per_desc;

    var best_dist = 1000000u;
    var second_best = 1000000u;
    var best_train_idx = 0u;

    for (var t_idx = 0u; t_idx < params.train_len; t_idx++) {
        let t_base = t_idx * u32_per_desc;
        var dist = 0u;
        
        for (var i = 0u; i < u32_per_desc; i++) {
            dist += countOneBits(query_data[q_base + i] ^ train_data[t_base + i]);
        }

        if (dist < best_dist) {
            second_best = best_dist;
            best_dist = dist;
            best_train_idx = t_idx;
        } else if (dist < second_best) {
            second_best = dist;
        }
    }

    // Output all candidates for host-side ratio test or filter here
    matches_out[q_idx] = vec4<f32>(f32(best_train_idx), f32(best_dist), f32(second_best), 0.0);
}
