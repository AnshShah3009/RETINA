// GPU Remap Kernel
// Performs image remapping with multiple interpolation and border modes.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    interpolation: u32, // 0: Nearest, 1: Linear, 2: Cubic
    border_mode: u32,   // 0: Constant, 1: Replicate, 2: Wrap, 3: Reflect, 4: Reflect101
    border_val: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // Packed u8
@group(0) @binding(1) var<storage, read> map_x: array<f32>;
@group(0) @binding(2) var<storage, read> map_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<u32>; // Packed u8
@group(0) @binding(4) var<uniform> params: Params;

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
    let ix = map_coord(x, params.src_w, params.border_mode);
    let iy = map_coord(y, params.src_h, params.border_mode);
    
    if (ix < 0 || iy < 0) {
        return params.border_val;
    }
    
    let idx = u32(iy) * params.src_w + u32(ix);
    let u32_idx = idx >> 2u;
    let shift = (idx & 3u) << 3u;
    return f32((input_data[u32_idx] >> shift) & 0xFFu);
}

fn sample_nearest(x: f32, y: f32) -> f32 {
    return get_u8(i32(round(x)), i32(round(y)));
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

fn bicubic_weight(x: f32) -> f32 {
    let abs_x = abs(x);
    let a = -0.5;
    if (abs_x <= 1.0) {
        return (a + 2.0) * abs_x * abs_x * abs_x - (a + 3.0) * abs_x * abs_x + 1.0;
    } else if (abs_x < 2.0) {
        return a * abs_x * abs_x * abs_x - 5.0 * a * abs_x * abs_x + 8.0 * a * abs_x - 4.0 * a;
    }
    return 0.0;
}

fn sample_bicubic(x: f32, y: f32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let dx = x - f32(xi);
    let dy = y - f32(yi);
    
    var val = 0.0;
    for (var m = -1; m <= 2; m++) {
        let wy = bicubic_weight(f32(m) - dy);
        for (var n = -1; n <= 2; n++) {
            let wx = bicubic_weight(f32(n) - dx);
            val += get_u8(xi + n, yi + m) * wx * wy;
        }
    }
    return val;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x; // Processes 4 pixels
    let y_dst = global_id.y;
    
    if (x_u32 * 4u >= params.dst_w || y_dst >= params.dst_h) {
        return;
    }

    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x_dst = x_u32 * 4u + i;
        if (x_dst >= params.dst_w) { break; }

        let idx = y_dst * params.dst_w + x_dst;
        let sx = map_x[idx];
        let sy = map_y[idx];
        
        var val_f = 0.0;
        if (params.interpolation == 0u) {
            val_f = sample_nearest(sx, sy);
        } else if (params.interpolation == 1u) {
            val_f = sample_bilinear(sx, sy);
        } else if (params.interpolation == 2u) {
            val_f = sample_bicubic(sx, sy);
        }
        
        let val = u32(clamp(val_f + 0.5, 0.0, 255.0));
        res_combined = res_combined | (val << (i * 8u));
    }

    output_data[y_dst * ((params.dst_w + 3u) / 4u) + x_u32] = res_combined;
}
