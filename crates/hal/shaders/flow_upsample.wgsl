// Flow Upsampling Shader for Farneback Optical Flow
// Upsamples flow field from coarser pyramid level

struct UpsampleParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read> input_flow: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_flow: array<f32>;
@group(0) @binding(2) var<uniform> params: UpsampleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.dst_w || y >= params.dst_h) {
        return;
    }
    
    let src_width_f = f32(params.src_w) - 1.0;
    let src_height_f = f32(params.src_h) - 1.0;
    let dst_width_f = f32(params.dst_w) - 1.0;
    let dst_height_f = f32(params.dst_h) - 1.0;
    
    // Map destination coordinates to source coordinates
    let src_x_f = f32(x) * src_width_f / dst_width_f;
    let src_y_f = f32(y) * src_height_f / dst_height_f;
    
    // Bilinear interpolation of flow values
    let x0 = u32(floor(src_x_f));
    let y0 = u32(floor(src_y_f));
    let x1 = min(params.src_w - 1u, x0 + 1u);
    let y1 = min(params.src_h - 1u, y0 + 1u);
    
    let fx = src_x_f - f32(x0);
    let fy = src_y_f - f32(y0);
    
    // Sample flow at four corners
    let idx00 = (y0 * params.src_w + x0) * 2u;
    let idx10 = (y0 * params.src_w + x1) * 2u;
    let idx01 = (y1 * params.src_w + x0) * 2u;
    let idx11 = (y1 * params.src_w + x1) * 2u;
    
    let u00 = input_flow[idx00 + 0u];
    let v00 = input_flow[idx00 + 1u];
    let u10 = input_flow[idx10 + 0u];
    let v10 = input_flow[idx10 + 1u];
    let u01 = input_flow[idx01 + 0u];
    let v01 = input_flow[idx01 + 1u];
    let u11 = input_flow[idx11 + 0u];
    let v11 = input_flow[idx11 + 1u];
    
    // Bilinear interpolation
    let u = mix(mix(u00, u10, fx), mix(u01, u11, fx), fy);
    let v = mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
    
    // Scale flow to match destination resolution
    let scale_x = dst_width_f / src_width_f;
    let scale_y = dst_height_f / src_height_f;
    
    let dst_idx = (y * params.dst_w + x) * 2u;
    output_flow[dst_idx + 0u] = u * scale_x;
    output_flow[dst_idx + 1u] = v * scale_y;
}
