// AKAZE Derivatives Kernel
// Computes Lx, Ly, Lxx, Lyy, Lxy and Ldet

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> Lx: array<f32>;
@group(0) @binding(2) var<storage, read_write> Ly: array<f32>;
@group(0) @binding(3) var<storage, read_write> Ldet: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn get_val(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    return input_data[iy * w + ix];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (x >= w || y >= h) {
        return;
    }

    let idx = y * w + x;

    // First derivatives (Scharr)
    let lx = (get_val(x + 1, y - 1) + 3.0 * get_val(x + 1, y) + get_val(x + 1, y + 1)) -
             (get_val(x - 1, y - 1) + 3.0 * get_val(x - 1, y) + get_val(x - 1, y + 1));
    
    let ly = (get_val(x - 1, y + 1) + 3.0 * get_val(x, y + 1) + get_val(x + 1, y + 1)) -
             (get_val(x - 1, y - 1) + 3.0 * get_val(x, y - 1) + get_val(x + 1, y - 1));

    Lx[idx] = lx / 32.0;
    Ly[idx] = ly / 32.0;

    // Second derivatives
    let lxx = get_val(x + 1, y) + get_val(x - 1, y) - 2.0 * get_val(x, y);
    let lyy = get_val(x, y + 1) + get_val(x, y - 1) - 2.0 * get_val(x, y);
    let lxy = (get_val(x + 1, y + 1) + get_val(x - 1, y - 1) - get_val(x - 1, y + 1) - get_val(x + 1, y - 1)) / 4.0;

    Ldet[idx] = lxx * lyy - lxy * lxy;
}
