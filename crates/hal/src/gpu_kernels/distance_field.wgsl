// distance_field.wgsl
// GPU Jump Flooding Algorithm (JFA) for distance field computation

@group(0) @binding(0) var<storage, read> input_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> seed_positions: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> output_dist: array<f32>;

struct Params {
    width: u32,
    height: u32,
    step_size: u32,
    _padding: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }
    
    let idx = u32(y) * params.width + u32(x);
    var best_seed = seed_positions[idx];
    var min_dist_sq = 1e38;
    
    if (best_seed.x >= 0) {
        let dx = f32(best_seed.x - x);
        let dy = f32(best_seed.y - y);
        min_dist_sq = dx * dx + dy * dy;
    }
    
    let s = i32(params.step_size);
    
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = x + dx * s;
            let ny = y + dy * s;
            
            if (nx >= 0 && nx < i32(params.width) && ny >= 0 && ny < i32(params.height)) {
                let n_idx = u32(ny) * params.width + u32(nx);
                let n_seed = seed_positions[n_idx];
                
                if (n_seed.x >= 0) {
                    let ddx = f32(n_seed.x - x);
                    let ddy = f32(n_seed.y - y);
                    let dist_sq = ddx * ddx + ddy * ddy;
                    
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_seed = n_seed;
                    }
                }
            }
        }
    }
    
    seed_positions[idx] = best_seed;
    output_dist[idx] = sqrt(min_dist_sq);
}
