struct Params {
    width: u32,
    height: u32,
    low_threshold: f32,
    high_threshold: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> mag_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> dir_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

// Pass 1: Gradients and Directions
@compute @workgroup_size(16, 16)
fn gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

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
    let idx = u32(y) * params.width + u32(x);
    mag_data[idx] = mag;

    // Direction quantization: 0, 45, 90, 135
    let abs_gx = abs(gx);
    let abs_gy = abs(gy);
    let tan_22_5 = 0.41421356;
    
    var dir = 0u;
    if (abs_gy <= abs_gx * tan_22_5) {
        dir = 0u;
    } else if (abs_gx <= abs_gy * tan_22_5) {
        dir = 2u;
    } else if (gx * gy > 0.0) {
        dir = 1u;
    } else {
        dir = 3u;
    }
    dir_data[idx] = dir;
}

// Pass 2: Non-Maximum Suppression
@group(0) @binding(0) var<storage, read> mag_in: array<f32>;
@group(0) @binding(1) var<storage, read> dir_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> nms_out: array<f32>;

@compute @workgroup_size(16, 16)
fn nms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
        nms_out[u32(y) * params.width + u32(x)] = 0.0;
        return;
    }

    let idx = u32(y) * params.width + u32(x);
    let m = mag_in[idx];
    let dir = dir_in[idx];

    var m1 = 0.0;
    var m2 = 0.0;

    if (dir == 0u) {
        m1 = mag_in[idx - 1u];
        m2 = mag_in[idx + 1u];
    } else if (dir == 1u) {
        m1 = mag_in[u32(y - 1) * params.width + u32(x + 1)];
        m2 = mag_in[u32(y + 1) * params.width + u32(x - 1)];
    } else if (dir == 2u) {
        m1 = mag_in[u32(y - 1) * params.width + u32(x)];
        m2 = mag_in[u32(y + 1) * params.width + u32(x)];
    } else {
        m1 = mag_in[u32(y - 1) * params.width + u32(x - 1)];
        m2 = mag_in[u32(y + 1) * params.width + u32(x + 1)];
    }

    if (m >= m1 && m >= m2) {
        nms_out[idx] = m;
    } else {
        nms_out[idx] = 0.0;
    }
}

// Pass 3: Hysteresis (Simplified: Double Thresholding)
// Connectivity is tricky on GPU without multiple passes or atomic flags.
// For now, we'll do double thresholding + 1-level connectivity check.
@group(0) @binding(0) var<storage, read> nms_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> final_out: array<u32>;

@compute @workgroup_size(16, 16)
fn hysteresis(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_x = global_id.x; // Process 4 pixels
    let y = i32(global_id.y);
    let w = i32(params.width);

    if (u32_x * 4u >= params.width || u32(y) >= params.height) {
        return;
    }

    var result_packed = 0u;

    for (var i = 0u; i < 4u; i++) {
        let x = i32(u32_x * 4u + i);
        if (u32(x) >= params.width) { break; }

        let idx = u32(y) * params.width + u32(x);
        let v = nms_in[idx];

        var pixel_val = 0u;
        if (v >= params.high_threshold) {
            pixel_val = 255u;
        } else if (v >= params.low_threshold) {
            // Check 8-neighbors for strong edges
            var has_strong_neighbor = false;
            for (var ny = -1; ny <= 1; ny++) {
                for (var nx = -1; nx <= 1; nx++) {
                    let nix = clamp(x + nx, 0, w - 1);
                    let niy = clamp(y + ny, 0, i32(params.height) - 1);
                    if (nms_in[u32(niy) * params.width + u32(nix)] >= params.high_threshold) {
                        has_strong_neighbor = true;
                        break;
                    }
                }
                if (has_strong_neighbor) { break; }
            }
            if (has_strong_neighbor) {
                pixel_val = 255u;
            }
        }

        result_packed = result_packed | (pixel_val << (i * 8u));
    }

    final_out[u32_x + u32(y) * ((params.width + 3u) / 4u)] = result_packed;
}
