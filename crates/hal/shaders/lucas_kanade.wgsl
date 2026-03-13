// GPU Lucas-Kanade Sparse Optical Flow
// Each thread tracks one feature point.

struct Params {
    num_points: u32,
    window_radius: i32,
    max_iters: u32,
    min_eigenvalue: f32,
}

struct Point {
    x: f32,
    y: f32,
}

@group(0) @binding(0) var<storage, read> prev_pyramid: array<f32>;
@group(0) @binding(1) var<storage, read> next_pyramid: array<f32>;
@group(0) @binding(2) var<storage, read> points_prev: array<Point>;
@group(0) @binding(3) var<storage, read_write> points_next: array<Point>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> pyramid_params: vec4<u32>; // w, h, level_offset, padding

fn get_pixel(x: f32, y: f32, data_offset: u32) -> f32 {
    let w = pyramid_params.x;
    let h = pyramid_params.y;
    
    if (x < 0.0 || x >= f32(w - 1u) || y < 0.0 || y >= f32(h - 1u)) {
        return 0.0;
    }
    
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let p00 = prev_pyramid[data_offset + y0 * w + x0];
    let p10 = prev_pyramid[data_offset + y0 * w + x1];
    let p01 = prev_pyramid[data_offset + y1 * w + x0];
    let p11 = prev_pyramid[data_offset + y1 * w + x1];
    
    return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

fn get_pixel_next(x: f32, y: f32, data_offset: u32) -> f32 {
    let w = pyramid_params.x;
    let h = pyramid_params.y;
    
    if (x < 0.0 || x >= f32(w - 1u) || y < 0.0 || y >= f32(h - 1u)) {
        return 0.0;
    }
    
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let p00 = next_pyramid[data_offset + y0 * w + x0];
    let p10 = next_pyramid[data_offset + y0 * w + x1];
    let p01 = next_pyramid[data_offset + y1 * w + x0];
    let p11 = next_pyramid[data_offset + y1 * w + x1];
    
    return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let p_prev = points_prev[idx];
    var p_next = points_next[idx]; // Starts with initial guess (usually p_prev)
    
    let r = params.window_radius;
    let offset = pyramid_params.z;
    
    // Spatial gradient computation (G matrix)
    var G00 = 0.0;
    var G01 = 0.0;
    var G11 = 0.0;
    
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let x = p_prev.x + f32(dx);
            let y = p_prev.y + f32(dy);
            
            let Ix = (get_pixel(x + 1.0, y, offset) - get_pixel(x - 1.0, y, offset)) * 0.5;
            let Iy = (get_pixel(x, y + 1.0, offset) - get_pixel(x, y - 1.0, offset)) * 0.5;
            
            G00 += Ix * Ix;
            G01 += Ix * Iy;
            G11 += Iy * Iy;
        }
    }
    
    let det = G00 * G11 - G01 * G01;
    if (det < params.min_eigenvalue) { return; }
    let inv_det = 1.0 / det;

    // Iterative refinement
    for (var iter = 0u; iter < params.max_iters; iter++) {
        var b0 = 0.0;
        var b1 = 0.0;
        
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                let x = p_prev.x + f32(dx);
                let y = p_prev.y + f32(dy);
                
                let nx = p_next.x + f32(dx);
                let ny = p_next.y + f32(dy);
                
                let Ix = (get_pixel(x + 1.0, y, offset) - get_pixel(x - 1.0, y, offset)) * 0.5;
                let Iy = (get_pixel(x, y + 1.0, offset) - get_pixel(x, y - 1.0, offset)) * 0.5;
                
                let It = get_pixel(x, y, offset) - get_pixel_next(nx, ny, offset);
                
                b0 += It * Ix;
                b1 += It * Iy;
            }
        }
        
        let du = (G11 * b0 - G01 * b1) * inv_det;
        let dv = (-G01 * b0 + G00 * b1) * inv_det;
        
        p_next.x += du;
        p_next.y += dv;
        
        if (du * du + dv * dv < 1e-4) { break; }
    }
    
    points_next[idx] = p_next;
}
