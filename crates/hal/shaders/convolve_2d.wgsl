struct Params {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32, // 0: Constant, 1: Replicate, 2: Reflect, 3: Wrap
    border_const: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> kernel_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_input_index(x: i32, y: i32) -> i32 {
    let w = i32(params.width);
    let h = i32(params.height);
    
    var ix = x;
    var iy = y;

    // Border handling
    if (params.border_mode == 0u) { // Constant
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) {
            return -1;
        }
    } else if (params.border_mode == 1u) { // Replicate
        ix = clamp(ix, 0, w - 1);
        iy = clamp(iy, 0, h - 1);
    } else if (params.border_mode == 2u) { // Reflect
        // Reflect 101: gfedcb|abcdefgh|gfedcba
        // Not perfectly implemented here, using simple reflection for now
        // Simple abs reflect:
        // if ix < 0 { ix = -ix - 1; } etc?
        // Let's implement standard Reflect101 logic used in OpenCV default
        // For standard Reflect:
        if (w == 1) { ix = 0; }
        else {
            let period = 2 * w;
            ix = ix % period;
            if (ix < 0) { ix += period; }
            if (ix >= w) { ix = period - ix - 1; }
        }
        
        if (h == 1) { iy = 0; }
        else {
            let period = 2 * h;
            iy = iy % period;
            if (iy < 0) { iy += period; }
            if (iy >= h) { iy = period - iy - 1; }
        }
    } else if (params.border_mode == 3u) { // Wrap
        ix = ix % w;
        if (ix < 0) { ix += w; }
        iy = iy % h;
        if (iy < 0) { iy += h; }
    }

    return iy * w + ix;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    let kw = i32(params.kernel_width);
    let kh = i32(params.kernel_height);
    let cx = kw / 2;
    let cy = kh / 2;

    var sum = 0.0;

    for (var ky = 0; ky < kh; ky++) {
        for (var kx = 0; kx < kw; kx++) {
            let ix = x + kx - cx;
            let iy = y + ky - cy;
            
            let input_idx = get_input_index(ix, iy);
            var val = params.border_const;
            
            if (input_idx >= 0) {
                val = input_data[input_idx];
            }
            
            let kernel_val = kernel_data[ky * kw + kx];
            sum += val * kernel_val;
        }
    }

    let out_idx = y * i32(params.width) + x;
    output_data[out_idx] = sum;
}
