struct Params {
    width: u32,
    height: u32,
    num_kps: u32,
}

@group(0) @binding(0) var<storage, read> image_data: array<f32>;
@group(0) @binding(1) var<storage, read> keypoints: array<vec4<f32>>; // [x, y, size, octave]
@group(0) @binding(2) var<storage, read_write> orientations: array<f32>; // Dominant angle for each kp
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.1415926535;

fn get_val(x: i32, y: i32) -> f32 {
    let cl_x = clamp(x, 0, i32(params.width) - 1);
    let cl_y = clamp(y, 0, i32(params.height) - 1);
    return image_data[u32(cl_y) * params.width + u32(cl_x)];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let kp_idx = global_id.x;
    if (kp_idx >= params.num_kps) { return; }

    let kp = keypoints[kp_idx];
    let kx = i32(kp.x);
    let ky = i32(kp.y);
    let sigma = kp.z * 1.5;
    let radius = i32(3.0 * sigma);

    var hist = array<f32, 36>();
    for (var i = 0u; i < 36u; i++) { hist[i] = 0.0; }

    let sigma_sq_inv = -0.5 / (sigma * sigma);

    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let x = kx + dx;
            let y = ky + dy;
            
            let g_x = get_val(x + 1, y) - get_val(x - 1, y);
            let g_y = get_val(x, y + 1) - get_val(x, y - 1);
            
            let mag = sqrt(g_x * g_x + g_y * g_y);
            var ori = atan2(g_y, g_x) * 180.0 / PI;
            if (ori < 0.0) { ori += 360.0; }
            
            let weight = exp(f32(dx * dx + dy * dy) * sigma_sq_inv);
            let bin = u32(ori / 10.0) % 36u;
            hist[bin] += mag * weight;
        }
    }

    // Find peak in histogram
    var max_mag = 0.0;
    var best_bin = 0u;
    for (var i = 0u; i < 36u; i++) {
        if (hist[i] > max_mag) {
            max_mag = hist[i];
            best_bin = i;
        }
    }

    // Assign dominant orientation (simple version without parabolic interpolation for now)
    orientations[kp_idx] = f32(best_bin) * 10.0;
}
