struct Params {
    width: u32,
    height: u32,
    threshold: f32,
    edge_threshold: f32,
}

@group(0) @binding(0) var<storage, read> dog_prev: array<f32>;
@group(0) @binding(1) var<storage, read> dog_curr: array<f32>;
@group(0) @binding(2) var<storage, read> dog_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> extrema_mask: array<u32>; // packed bitmask
@group(0) @binding(4) var<uniform> params: Params;

fn get_val(data: array<f32>, x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    return data[u32(cl_y) * params.width + u32(cl_x)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x;
    let y = i32(global_id.y);
    
    if (x_u32 * 4u >= params.width || y >= i32(params.height)) {
        return;
    }

    var res_combined = 0u;
    for (var k = 0u; k < 4u; k++) {
        let x = i32(x_u32 * 4u + k);
        if (x >= i32(params.width)) { break; }
        if (x < 1 || x >= i32(params.width) - 1 || y < 1 || y >= i32(params.height) - 1) {
            continue;
        }

        let val = dog_curr[u32(y) * params.width + u32(x)];
        var is_extrema = false;

        if (abs(val) > params.threshold) {
            var is_max = true;
            var is_min = true;

            // Check 3x3x3 neighborhood
            for (var ds = -1; ds <= 1; ds++) {
                for (var dy = -1; dy <= 1; dy++) {
                    for (var dx = -1; dx <= 1; dx++) {
                        if (ds == 0 && dx == 0 && dy == 0) { continue; }
                        
                        var neighbor_val: f32;
                        if (ds == -1) { neighbor_val = dog_prev[u32(y + dy) * params.width + u32(x + dx)]; }
                        else if (ds == 0) { neighbor_val = dog_curr[u32(y + dy) * params.width + u32(x + dx)]; }
                        else { neighbor_val = dog_next[u32(y + dy) * params.width + u32(x + dx)]; }

                        if (neighbor_val >= val) { is_max = false; }
                        if (neighbor_val <= val) { is_min = false; }
                    }
                }
            }

            if (is_max || is_min) {
                // Edge response check
                let dxx = get_val(dog_curr, x+1, y) + get_val(dog_curr, x-1, y) - 2.0*val;
                let dyy = get_val(dog_curr, x, y+1) + get_val(dog_curr, x, y-1) - 2.0*val;
                let dxy = (get_val(dog_curr, x+1, y+1) - get_val(dog_curr, x+1, y-1) - 
                           get_val(dog_curr, x-1, y+1) + get_val(dog_curr, x-1, y-1)) / 4.0;
                
                let tr = dxx + dyy;
                let det = dxx * dyy - dxy * dxy;
                
                let edge_threshold = params.edge_threshold;
                if (det > 0.0 && (tr * tr) / det < (edge_threshold + 1.0) * (edge_threshold + 1.0) / edge_threshold) {
                    is_extrema = true;
                }
            }
        }
        
        if (is_extrema) {
            res_combined = res_combined | (1u << (k * 8u)); // Use byte as boolean for consistency
        }
    }

    extrema_mask[u32(y) * ((params.width + 3u) / 4u) + x_u32] = res_combined;
}
