struct Params {
    width: u32,
    height: u32,
    min_disparity: i32,
    num_disparities: u32,
    block_size: u32,
    method: u32, // 0: BlockMatching, 1: SGM
}

@group(0) @binding(0) var<storage, read> left_data: array<u32>;
@group(0) @binding(1) var<storage, read> right_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> disparity_map: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(data: ptr<storage, array<u32>, read>, x: i32, y: i32) -> f32 {
    let ix = clamp(x, 0, i32(params.width) - 1);
    let iy = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(iy) * params.width + u32(ix);
    let combined = (*data)[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    let half_block = i32(params.block_size / 2u);
    
    // Skip border
    if (x < half_block || x >= i32(params.width) - half_block || 
        y < half_block || y >= i32(params.height) - half_block) {
        disparity_map[u32(y) * params.width + u32(x)] = -1.0;
        return;
    }

    var best_disparity = f32(params.min_disparity);
    var min_cost = 1e10;

    for (var d = 0u; d < params.num_disparities; d = d + 1u) {
        let cur_d = params.min_disparity + i32(d);
        if (x - cur_d < half_block) { continue; }

        var cost = 0.0;
        for (var dy = -half_block; dy <= half_block; dy = dy + 1) {
            for (var dx = -half_block; dx <= half_block; dx = dx + 1) {
                let lv = get_pixel(&left_data, x + dx, y + dy);
                let rv = get_pixel(&right_data, x + dx - cur_d, y + dy);
                cost = cost + abs(lv - rv);
            }
        }

        if (cost < min_cost) {
            min_cost = cost;
            best_disparity = f32(cur_d);
        }
    }

    disparity_map[u32(y) * params.width + u32(x)] = best_disparity;
}
