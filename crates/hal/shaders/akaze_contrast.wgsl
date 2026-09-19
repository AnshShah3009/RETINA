// AKAZE Contrast K Computation
// Computes gradient magnitudes and a histogram for percentile calculation

struct Params {
    width: u32,
    height: u32,
    num_bins: u32,
    max_mag: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (x >= w - 1 || y >= h - 1 || x == 0 || y == 0) {
        return;
    }

    let idx = y * w + x;
    
    // Simple central difference for K estimation
    let lx = input_data[y * w + x + 1] - input_data[y * w + x - 1];
    let ly = input_data[(y + 1) * w + x] - input_data[(y - 1) * w + x];
    let mag = sqrt(lx * lx + ly * ly);

    let bin = u32(clamp(mag / params.max_mag * f32(params.num_bins), 0.0, f32(params.num_bins - 1u)));
    atomicAdd(&histogram[bin], 1u);
}
