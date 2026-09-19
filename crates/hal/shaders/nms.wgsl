// GPU Pixel-wise Non-Maximum Suppression (NMS)
// Suppresses pixels if they are not the maximum in a NxN window.

struct Params {
    width: u32,
    height: u32,
    threshold: f32,
    window_radius: i32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (x >= w || y >= h) {
        return;
    }

    let center_val = input_data[y * w + x];
    
    if (center_val < params.threshold) {
        output_data[y * w + x] = 0.0;
        return;
    }

    let r = params.window_radius;
    
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            
            let nx = x + dx;
            let ny = y + dy;
            
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                let neighbor_val = input_data[ny * w + nx];
                
                // If a neighbor is strictly greater, or equal with a tie-breaking rule
                if (neighbor_val > center_val) {
                    output_data[y * w + x] = 0.0;
                    return;
                }
                
                // Tie-breaking: smaller index wins if values are equal
                if (neighbor_val == center_val) {
                    let neighbor_idx = ny * w + nx;
                    let center_idx = y * w + x;
                    if (neighbor_idx < center_idx) {
                        output_data[y * w + x] = 0.0;
                        return;
                    }
                }
            }
        }
    }

    output_data[y * w + x] = center_val;
}
