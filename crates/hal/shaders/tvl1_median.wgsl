// TV-L1 optical flow: median filter of both flow planes, as OpenCV DualTVL1
// does at the start of every outer iteration (cv::medianBlur with
// `medianFiltering` = 3 or 5, replicated border). Reads `src`, writes `dst`;
// the host swaps the two buffers afterwards.
//
// `src` and `dst` are whole TV-L1 states (see tvl1_primal_dual.wgsl): the flow
// in planes 0 and 1, the four dual variables in planes 2..5. The dual
// variables are copied across unchanged so `dst` is a complete state.
//
// median_radius is 1 (3x3) or 2 (5x5); the host never dispatches this shader
// for any other value, and the window is capped at 5x5 here regardless.
//
// Planar layout: u1 at [i], u2 at [width * height + i], p11..p22 at
// [k * width * height + i] for k = 2..5.

struct Tvl1Params {
    width: u32,
    height: u32,
    l_t: f32,
    theta: f32,
    taut: f32,
    median_radius: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Tvl1Params;

fn median_at(x: u32, y: u32, base: u32) -> f32 {
    let r = i32(min(params.median_radius, 2u));
    let last_x = i32(params.width) - 1;
    let last_y = i32(params.height) - 1;
    var vals: array<f32, 25>;
    var count = 0u;
    for (var dy = -r; dy <= r; dy++) {
        let yy = u32(clamp(i32(y) + dy, 0, last_y));
        for (var dx = -r; dx <= r; dx++) {
            let xx = u32(clamp(i32(x) + dx, 0, last_x));
            // Insertion sort as the values arrive.
            let v = src[base + yy * params.width + xx];
            var j = count;
            while (j > 0u && vals[j - 1u] > v) {
                vals[j] = vals[j - 1u];
                j = j - 1u;
            }
            vals[j] = v;
            count = count + 1u;
        }
    }
    // count is (2r + 1)², odd, so the middle element is the median.
    return vals[count >> 1u];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let w = params.width;
    let h = params.height;

    if (x >= w || y >= h) {
        return;
    }

    let n = w * h;
    let i = y * w + x;
    dst[i] = median_at(x, y, 0u);
    dst[n + i] = median_at(x, y, n);
    for (var k = 2u; k < 6u; k++) {
        dst[k * n + i] = src[k * n + i];
    }
}
