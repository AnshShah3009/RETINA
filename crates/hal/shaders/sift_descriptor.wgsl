struct Params {
    width: u32,
    height: u32,
    num_kps: u32,
}

@group(0) @binding(0) var<storage, read> image_data: array<f32>;
@group(0) @binding(1) var<storage, read> keypoints: array<vec4<f32>>; // [x, y, size, angle]
@group(0) @binding(2) var<storage, read_write> descriptors: array<f32>; // 128 floats per kp
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
    let cx = kp.x;
    let cy = kp.y;
    let size = kp.z;
    let angle_rad = kp.w * PI / 180.0;
    
    let cos_a = cos(angle_rad);
    let sin_a = sin(angle_rad);

    // 4x4 subregions, each with 8 bins -> 128 dimensions
    var hist = array<f32, 128>();
    for (var i = 0u; i < 128u; i++) { hist[i] = 0.0; }

    let bin_width = size * 3.0; // Typical SIFT param
    let radius = i32(bin_width * 2.0); 

    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            // Rotate coordinate system
            let rx = (f32(dx) * cos_a + f32(dy) * sin_a) / bin_width;
            let ry = (-f32(dx) * sin_a + f32(dy) * cos_a) / bin_width;
            
            // Region index (-2.0 to 2.0 -> 0 to 4)
            let r_bin_x = rx + 1.5;
            let r_bin_y = ry + 1.5;

            if (r_bin_x > -1.0 && r_bin_x < 4.0 && r_bin_y > -1.0 && r_bin_y < 4.0) {
                let x = i32(cx) + dx;
                let y = i32(cy) + dy;
                
                let g_x = get_val(x + 1, y) - get_val(x - 1, y);
                let g_y = get_val(x, y + 1) - get_val(x, y - 1);
                let mag = sqrt(g_x * g_x + g_y * g_y);
                var ori = atan2(g_y, g_x) - angle_rad;
                while (ori < 0.0) { ori += 2.0 * PI; }
                let o_bin = (ori * 8.0 / (2.0 * PI));

                // Trilinear interpolation into histograms (simplified for now)
                let ix = i32(floor(r_bin_x));
                let iy = i32(floor(r_bin_y));
                let io = i32(floor(o_bin));

                if (ix >= 0 && ix < 4 && iy >= 0 && iy < 4) {
                    let bin_idx = u32((iy * 4 + ix) * 8 + (io % 8));
                    hist[bin_idx] += mag;
                }
            }
        }
    }

    // Normalize and output
    var norm_sq = 0.0;
    for (var i = 0u; i < 128u; i++) { norm_sq += hist[i] * hist[i]; }
    let norm_inv = 1.0 / (sqrt(norm_sq) + 1e-7);

    for (var i = 0u; i < 128u; i++) {
        descriptors[kp_idx * 128u + i] = min(0.2, hist[i] * norm_inv);
    }
}
