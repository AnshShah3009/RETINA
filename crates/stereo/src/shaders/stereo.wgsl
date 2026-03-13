// Stereo Block Matching Compute Shader
// Performs sum of absolute differences (SAD) over a local window

struct StereoParams {
    width: u32,
    height: u32,
    min_disparity: u32,
    max_disparity: u32,
    block_size: u32,
}

@group(0) @binding(0)
var left_image: texture_2d<f32>;

@group(0) @binding(1)
var right_image: texture_2d<f32>;

@group(0) @binding(2)
var disparity_output: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> params: StereoParams;

// Compute sum of absolute differences for a given disparity
fn compute_sad(x: u32, y: u32, disparity: u32) -> f32 {
    let half_block = params.block_size / 2u;
    var sad: f32 = 0.0;
    var count: u32 = 0u;
    
    for (var dy: u32 = 0u; dy < params.block_size; dy = dy + 1u) {
        for (var dx: u32 = 0u; dx < params.block_size; dx = dx + 1u) {
            let lx = i32(x) + i32(dx) - i32(half_block);
            let ly = i32(y) + i32(dy) - i32(half_block);
            let rx = i32(x) + i32(dx) - i32(half_block) - i32(disparity);
            let ry = i32(y) + i32(dy) - i32(half_block);
            
            // Check bounds
            if (lx >= 0 && ly >= 0 && rx >= 0 && 
                lx < i32(params.width) && ly < i32(params.height) &&
                rx < i32(params.width) && ry < i32(params.height)) {
                
                let left_val = textureLoad(left_image, vec2<i32>(lx, ly), 0).r;
                let right_val = textureLoad(right_image, vec2<i32>(rx, ry), 0).r;
                
                sad = sad + abs(left_val - right_val);
                count = count + 1u;
            }
        }
    }
    
    if (count > 0u) {
        return sad / f32(count);
    } else {
        return 999999.0; // Invalid
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Check bounds
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let half_block = params.block_size / 2u;
    
    // Skip border pixels
    if (x < half_block || x >= params.width - half_block ||
        y < half_block || y >= params.height - half_block) {
        textureStore(disparity_output, vec2<i32>(i32(x), i32(y)), vec4<f32>(-1.0, 0.0, 0.0, 1.0));
        return;
    }
    
    var best_disparity: u32 = params.min_disparity;
    var best_cost: f32 = 999999.0;
    var second_best_cost: f32 = 999999.0;
    
    // Search over all disparities
    for (var d: u32 = params.min_disparity; d <= params.max_disparity; d = d + 1u) {
        // Check if we can search this disparity (must have valid right image pixel)
        if (x >= half_block + d) {
            let cost = compute_sad(x, y, d);
            
            if (cost < best_cost) {
                second_best_cost = best_cost;
                best_cost = cost;
                best_disparity = d;
            } else if (cost < second_best_cost) {
                second_best_cost = cost;
            }
        }
    }
    
    // Uniqueness check (optional)
    let uniqueness_threshold: f32 = 0.95;
    var final_disparity: f32 = f32(best_disparity);
    
    if (second_best_cost < best_cost / uniqueness_threshold) {
        // Ambiguous match
        final_disparity = -1.0;
    }
    
    // Store result
    textureStore(disparity_output, vec2<i32>(i32(x), i32(y)), vec4<f32>(final_disparity, 0.0, 0.0, 1.0));
}
