struct Params {
    width: u32,
    height: u32,
    min_radius: f32,
    max_radius: f32,
    num_radii: u32,
    edge_threshold: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> accumulator: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

@compute @workgroup_size(16, 16)
fn vote(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    // 1. Sobel Gradients
    let p00 = get_pixel(x - 1, y - 1);
    let p01 = get_pixel(x,     y - 1);
    let p02 = get_pixel(x + 1, y - 1);
    let p10 = get_pixel(x - 1, y);
    let p12 = get_pixel(x + 1, y);
    let p20 = get_pixel(x - 1, y + 1);
    let p21 = get_pixel(x,     y + 1);
    let p22 = get_pixel(x + 1, y + 1);

    let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
    let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);

    let mag = sqrt(gx * gx + gy * gy);

    // 2. Thresholding (Edge detection)
    if (mag < params.edge_threshold) {
        return;
    }

    let ux = gx / mag;
    let uy = gy / mag;

    // 3. Voting along gradient line for each radius
    for (var i = 0u; i < params.num_radii; i = i + 1u) {
        let r = params.min_radius + f32(i);
        
        // Vote in both directions along gradient
        for (var sign = -1.0; sign <= 1.0; sign = sign + 2.0) {
            let cx = i32(round(f32(x) + sign * r * ux));
            let cy = i32(round(f32(y) + sign * r * uy));
            
            if (cx >= 0 && u32(cx) < params.width && cy >= 0 && u32(cy) < params.height) {
                let acc_idx = (i * params.height + u32(cy)) * params.width + u32(cx);
                atomicAdd(&accumulator[acc_idx], 1u);
            }
        }
    }
}
