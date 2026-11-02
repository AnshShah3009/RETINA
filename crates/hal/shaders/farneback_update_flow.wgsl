// Farneback: average the normal-equation terms over the window and solve.
// Port of OpenCV's FarnebackUpdateFlow_Blur / _GaussianBlur
// (modules/video/src/optflowgf.cpp).
//
// `mats` holds the five planes written by farneback_update_matrices.wgsl
// (AᵀA entries G11, G12, G22 and AᵀdB entries h1, h2). Each is averaged over a
// (2 half_window + 1)² window with replicated borders: a box (win_sigma <= 0,
// OpenCV's default) or a normalised Gaussian of standard deviation win_sigma
// (OPTFLOW_FARNEBACK_GAUSSIAN, which uses 0.3 * half_window). Then
//   flow = G⁻¹ h,   1 / det(G) regularised with OpenCV's + 1e-3,
// which assumes 0-255 intensities as OpenCV's 8-bit path does.
//
// The flow is the total displacement (the matrices already contain the
// current estimate), so this pass only writes it. It never reads `flow`, so
// overwriting it in place is race-free.
//
// Flow is planar: u at [i], v at [width * height + i].

struct FlowParams {
    width: u32,
    height: u32,
    half_window: u32,
    win_sigma: f32,
}

@group(0) @binding(0) var<storage, read> mats: array<f32>;
@group(0) @binding(1) var<storage, read_write> flow: array<f32>;
@group(0) @binding(2) var<uniform> params: FlowParams;

fn window_weight(d: i32) -> f32 {
    if (params.win_sigma <= 0.0) {
        return 1.0;
    }
    let df = f32(d);
    return exp(-(df * df) / (2.0 * params.win_sigma * params.win_sigma));
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
    let half = i32(min(params.half_window, 256u));
    let last_x = i32(w) - 1;
    let last_y = i32(h) - 1;

    var g11 = 0.0;
    var g12 = 0.0;
    var g22 = 0.0;
    var h1 = 0.0;
    var h2 = 0.0;
    var wsum = 0.0;

    for (var dy = -half; dy <= half; dy++) {
        let yy = u32(clamp(i32(y) + dy, 0, last_y));
        let wy = window_weight(dy);
        for (var dx = -half; dx <= half; dx++) {
            let xx = u32(clamp(i32(x) + dx, 0, last_x));
            let wgt = wy * window_weight(dx);
            let j = yy * w + xx;
            g11 = g11 + wgt * mats[j];
            g12 = g12 + wgt * mats[n + j];
            g22 = g22 + wgt * mats[2u * n + j];
            h1 = h1 + wgt * mats[3u * n + j];
            h2 = h2 + wgt * mats[4u * n + j];
            wsum = wsum + wgt;
        }
    }

    let s = 1.0 / wsum;
    g11 = g11 * s;
    g12 = g12 * s;
    g22 = g22 * s;
    h1 = h1 * s;
    h2 = h2 * s;

    let idet = 1.0 / (g11 * g22 - g12 * g12 + 1e-3);
    let i = y * w + x;
    flow[i] = (g22 * h1 - g12 * h2) * idet;
    flow[n + i] = (g11 * h2 - g12 * h1) * idet;
}
