// Feature collection kernels
// Collects keypoints from a score map into a list of coordinates.

struct Params {
    width: u32,
    height: u32,
    num_elements: u32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> score_map: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Pass 1: Count keypoints per thread or block
@compute @workgroup_size(256)
fn count_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.num_elements) { return; }
    
    // Check 4 pixels at once
    let val4 = score_map[gid];
    var count = 0u;
    if ((val4 & 0xFFu) > 0u) { count++; }
    if (((val4 >> 8u) & 0xFFu) > 0u) { count++; }
    if (((val4 >> 16u) & 0xFFu) > 0u) { count++; }
    if (((val4 >> 24u) & 0xFFu) > 0u) { count++; }
    
    counts[gid] = count;
}

// Pass 2: Collect points using scanned indices
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

@group(0) @binding(0) var<storage, read> score_map_2: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_keypoints: array<Keypoint>;
@group(0) @binding(3) var<uniform> params_2: Params;

@compute @workgroup_size(256)
fn collect_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params_2.num_elements) { return; }
    
    let val4 = score_map_2[gid];
    var base_idx = indices[gid];
    
    for (var i = 0u; i < 4u; i++) {
        let score = (val4 >> (i * 8u)) & 0xFFu;
        if (score > 0u) {
            let total_idx = gid * 4u + i;
            let real_x = total_idx % params_2.width;
            let real_y = total_idx / params_2.width;
            
            if (total_idx < params_2.width * params_2.height) {
                out_keypoints[base_idx].x = f32(real_x);
                out_keypoints[base_idx].y = f32(real_y);
                out_keypoints[base_idx].response = f32(score);
                out_keypoints[base_idx].size = 7.0; // Default FAST size
                out_keypoints[base_idx].angle = -1.0;
                base_idx++;
            }
        }
    }
}
