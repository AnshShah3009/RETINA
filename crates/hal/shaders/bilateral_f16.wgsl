enable f16;
struct Params {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f16,
    sigma_space_sq_inv: f16,
}

@group(0) @binding(0) var<storage, read> input_data: array<f16>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_f16(x: i32, y: i32) -> f16 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    return input_data[idx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x;
    let y = i32(global_id.y);
    
    if (x_u32 >= params.width || y >= i32(params.height)) {
        return;
    }

    let x = i32(x_u32);
    let center_val = get_f16(x, y);
    var sum = 0.0h;
    var norm = 0.0h;

    for (var j = -params.radius; j <= params.radius; j++) {
        for (var i = -params.radius; i <= params.radius; i++) {
            let val = get_f16(x + i, y + j);
            
            let dist_sq = f16(i * i + j * j);
            let range_sq = (val - center_val) * (val - center_val);
            
            let weight = exp(dist_sq * params.sigma_space_sq_inv + range_sq * params.sigma_color_sq_inv);
            sum += val * weight;
            norm += weight;
        }
    }
    
    let final_val = sum / norm;
    output_data[u32(y) * params.width + x_u32] = final_val;
}
