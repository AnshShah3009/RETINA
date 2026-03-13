struct Params {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_u8(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cl_y) * params.width + u32(cl_x);
    let u32_idx = idx / 4u;
    let shift = (idx % 4u) * 8u;
    return f32((input_data[u32_idx] >> shift) & 0xFFu);
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

        let center_val = get_u8(x, y);
        var sum = 0.0;
        var norm = 0.0;

        for (var j = -params.radius; j <= params.radius; j++) {
            for (var i = -params.radius; i <= params.radius; i++) {
                let val = get_u8(x + i, y + j);
                
                let dist_sq = f32(i * i + j * j);
                let range_sq = (val - center_val) * (val - center_val);
                
                let weight = exp(dist_sq * params.sigma_space_sq_inv + range_sq * params.sigma_color_sq_inv);
                sum += val * weight;
                norm += weight;
            }
        }
        
        let final_val = u32(sum / norm);
        res_combined = res_combined | ((final_val & 0xFFu) << (k * 8u));
    }

    output_data[u32(y) * ((params.width + 3u) / 4u) + x_u32] = res_combined;
}
