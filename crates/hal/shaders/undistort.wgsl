// GPU Undistort Kernel
// Directly computes undistorted coordinates and samples the source image.

struct CameraParams {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    ifx: f32, // 1/fx
    ify: f32, // 1/fy
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    _pad: u32,
}

struct ImageParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    interpolation: u32, // 0: Nearest, 1: Linear, 2: Cubic
    border_mode: u32,   // 0: Constant, 1: Replicate, 2: Wrap, 3: Reflect, 4: Reflect101
    border_val: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>; // Packed u8
@group(0) @binding(2) var<uniform> cam: CameraParams;
@group(0) @binding(3) var<uniform> img: ImageParams;
@group(0) @binding(4) var<uniform> rect_mat: mat3x3<f32>; // Inv Rectification * Inv NewK

fn map_coord(coord: i32, len: u32, mode: u32) -> i32 {
    let n = i32(len);
    if (mode == 0u) { // Constant
        if (coord < 0 || coord >= n) { return -1; }
        return coord;
    } else if (mode == 1u) { // Replicate
        return clamp(coord, 0, n - 1);
    } else if (mode == 2u) { // Wrap
        var c = coord % n;
        if (c < 0) { c += n; }
        return c;
    } else if (mode == 3u) { // Reflect
        let period = 2 * n;
        var c = coord % period;
        if (c < 0) { c += period; }
        if (c >= n) { c = period - c - 1; }
        return c;
    } else if (mode == 4u) { // Reflect101
        if (n == 1) { return 0; }
        let period = 2 * n - 2;
        var c = coord % period;
        if (c < 0) { c += period; }
        if (c >= n) { c = period - c; }
        return c;
    }
    return -1;
}

fn get_u8(x: i32, y: i32) -> f32 {
    let ix = map_coord(x, img.src_w, img.border_mode);
    let iy = map_coord(y, img.src_h, img.border_mode);
    
    if (ix < 0 || iy < 0) {
        return img.border_val;
    }
    
    let idx = u32(iy) * img.src_w + u32(ix);
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return f32((input_data[u32_idx] >> shift) & 0xFFu);
}

fn sample_bilinear(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    
    let dx = x - f32(x0);
    let dy = y - f32(y0);
    
    let p00 = get_u8(x0, y0);
    let p10 = get_u8(x1, y0);
    let p01 = get_u8(x0, y1);
    let p11 = get_u8(x1, y1);
    
    return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x; // Processes 4 pixels
    let y_dst = global_id.y;
    
    if (x_u32 * 4u >= img.dst_w || y_dst >= img.dst_h) {
        return;
    }

    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x_dst = x_u32 * 4u + i;
        if (x_dst >= img.dst_w) { break; }

        // 1. Back-project to normalized coordinates
        let dst_pt = vec3<f32>(f32(x_dst), f32(y_dst), 1.0);
        let norm_pt = rect_mat * dst_pt;
        let x = norm_pt.x / norm_pt.z;
        let y = norm_pt.y / norm_pt.z;

        // 2. Apply distortion model
        let r2 = x*x + y*y;
        let r4 = r2*r2;
        let r6 = r4*r2;
        
        let radial = 1.0 + cam.k1*r2 + cam.k2*r4 + cam.k3*r6;
        let x_dist = x * radial + 2.0*cam.p1*x*y + cam.p2*(r2 + 2.0*x*x);
        let y_dist = y * radial + cam.p1*(r2 + 2.0*y*y) + 2.0*cam.p2*x*y;

        // 3. Project to source pixel coordinates
        let sx = x_dist * cam.fx + cam.cx;
        let sy = y_dist * cam.fy + cam.cy;

        // 4. Sample and pack
        var val_f = 0.0;
        if (img.interpolation == 0u) {
            val_f = get_u8(i32(round(sx)), i32(round(sy)));
        } else {
            val_f = sample_bilinear(sx, sy);
        }
        
        let val = u32(clamp(val_f + 0.5, 0.0, 255.0));
        res_combined = res_combined | (val << (i * 8u));
    }

    output_data[y_dst * ((img.dst_w + 3u) / 4u) + x_u32] = res_combined;
}
