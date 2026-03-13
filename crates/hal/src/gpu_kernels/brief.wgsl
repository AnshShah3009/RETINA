// Rotated BRIEF Descriptor Kernel
// Computes 256-bit binary descriptors for keypoints.

struct Keypoint {
    x: f32,
    y: f32,
    size: f32,
    angle: f32,
    response: f32,
    octave: i32,
    class_id: i32,
    padding: i32,
}

struct BRIEFPoint {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

struct Params {
    width: u32,
    height: u32,
    num_keypoints: u32,
    num_pairs: u32, // Should be 256
}

@group(0) @binding(0) var<storage, read> image: array<u32>;
@group(0) @binding(1) var<storage, read> keypoints: array<Keypoint>;
@group(0) @binding(2) var<storage, read> patterns: array<BRIEFPoint>;
@group(0) @binding(3) var<storage, read_write> descriptors: array<u32>; // [num_kp * 8] (8 * 32 bits = 256 bits)
@group(0) @binding(4) var<uniform> params: Params;

fn get_pixel_bilinear(x: f32, y: f32) -> f32 {
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(params.width - 1u, x0 + 1u);
    let y1 = min(params.height - 1u, y0 + 1u);
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let p00 = get_u8(y0 * params.width + x0);
    let p10 = get_u8(y0 * params.width + x1);
    let p01 = get_u8(y1 * params.width + x0);
    let p11 = get_u8(y1 * params.width + x1);
    
    return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

fn get_u8(idx: u32) -> f32 {
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return f32((image[u32_idx] >> shift) & 0xFFu);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let kp_idx = global_id.x;
    if (kp_idx >= params.num_keypoints) {
        return;
    }

    let kp = keypoints[kp_idx];
    let cos_a = cos(kp.angle);
    let sin_a = sin(kp.angle);
    
    // Each thread processes ONE keypoint and all its 256 pairs
    // Note: To optimize, we could use multiple threads per keypoint, 
    // but 256-bit packing is easy in one thread.
    
    for (var i = 0u; i < 8u; i++) { // 8 * 32 bits = 256 bits
        var desc_chunk = 0u;
        for (var j = 0u; j < 32u; j++) {
            let pair_idx = i * 32u + j;
            let pair = patterns[pair_idx];
            
            // Rotate pair points
            let x1_rot = pair.x1 * cos_a - pair.y1 * sin_a + kp.x;
            let y1_rot = pair.x1 * sin_a + pair.y1 * cos_a + kp.y;
            let x2_rot = pair.x2 * cos_a - pair.y2 * sin_a + kp.x;
            let y2_rot = pair.x2 * sin_a + pair.y2 * cos_a + kp.y;
            
            let v1 = get_pixel_bilinear(x1_rot, y1_rot);
            let v2 = get_pixel_bilinear(x2_rot, y2_rot);
            
            if (v1 < v2) {
                desc_chunk = desc_chunk | (1u << j);
            }
        }
        descriptors[kp_idx * 8u + i] = desc_chunk;
    }
}
