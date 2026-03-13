// ORB Intensity Centroid (Orientation) Kernel
// Calculates the dominant orientation of keypoints using intensity moments.

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

struct Params {
    width: u32,
    height: u32,
    num_keypoints: u32,
    radius: i32,
}

@group(0) @binding(0) var<storage, read> image: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read> keypoints: array<Keypoint>;
@group(0) @binding(2) var<storage, read_write> angles: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height)) {
        return 0.0;
    }
    let idx = u32(y) * params.width + u32(x);
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return f32((image[u32_idx] >> shift) & 0xFFu);
}

var<workgroup> local_m10: array<f32, 256>;
var<workgroup> local_m01: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let kp_idx = group_id.x;
    if (kp_idx >= params.num_keypoints) {
        return;
    }

    let tid = local_id.x;
    let kp = keypoints[kp_idx];
    let cx = i32(kp.x + 0.5);
    let cy = i32(kp.y + 0.5);
    let r = params.radius;
    
    var m10 = 0.0;
    var m01 = 0.0;
    
    // Each thread processes a portion of the patch
    for (var i = tid; i < u32((2 * r + 1) * (2 * r + 1)); i += 256u) {
        let dx = i32(i % u32(2 * r + 1)) - r;
        let dy = i32(i / u32(2 * r + 1)) - r;
        
        if (dx * dx + dy * dy <= r * r) {
            let val = get_pixel(cx + dx, cy + dy);
            m10 += f32(dx) * val;
            m01 += f32(dy) * val;
        }
    }
    
    local_m10[tid] = m10;
    local_m01[tid] = m01;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            local_m10[tid] += local_m10[tid + s];
            local_m01[tid] += local_m01[tid + s];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        angles[kp_idx] = atan2(local_m01[0], local_m10[0]);
    }
}
