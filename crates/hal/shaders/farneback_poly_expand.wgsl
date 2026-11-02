// Polynomial expansion for Farneback optical flow.
//
// Uniform layout matches `FarnebackParams` in `gpu_kernels/optical_flow.rs`
// (#[repr(C)], 4-byte fields, no padding):
//   0 width, 4 height, 8 stride, 12 flow_stride, 16 window_size,
//   20 poly_n, 24 poly_sigma, 28 num_iters.
// Reading poly_n from byte 12 took `flow_stride` (16 on an 8-pixel level)
// and reading sigma from byte 16 took a zero, so the Gaussian was NaN.
//
// Each pixel is the Gaussian-weighted least-squares fit of
// I ≈ f + d x + e y + a x² + b xy + c y² on the (2 poly_n + 1)² window, as in
// OpenCV's FarnebackPolyExp: `poly_n` is the half-size of the neighbourhood
// (OpenCV loops x over -n..=n), and samples past the image edge replicate the
// border pixel, so every pixel solves the same Gram matrix. The old shader used
// a poly_n-wide window (half-size poly_n / 2) and dropped the samples outside
// the image, which disagreed with OpenCV everywhere, most near the border.
// A constant image has f equal to that constant and a = b = c = 0.
// Everything that does not depend on the image (the weights and the
// elimination of the Gram matrix) comes from `farneback_poly_setup.wgsl`.
//
// Output: six coefficients per pixel, [f, d, e, a, b, c] at 6 * idx. This is a
// scratch buffer private to `gpu_kernels::optical_flow::farneback`.

struct PolyExpandParams {
    src_w: u32,
    src_h: u32,
    stride: u32,
    flow_stride: u32,
    window_size: u32,
    poly_n: u32,
    sigma: f32,
    num_iters: u32,
}

// Image-independent data from `farneback_poly_setup.wgsl` (layout there): the
// 1-D Gaussian weights and a record of the elimination of the shared Gram
// matrix. A pixel accumulates only its right-hand side b and replays on it the
// row swaps and updates the full solve made, then back-substitutes, which
// gives the same bits as solving the whole system per pixel did.
const PIV: u32 = 0u;
const FAC: u32 = 6u;
const USE: u32 = 42u;
const UPPER: u32 = 78u;
const GAUSS: u32 = 114u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PolyExpandParams;
@group(0) @binding(3) var<storage, read> setup: array<f32>;

fn solve6(b_in: array<f32, 6>) -> array<f32, 6> {
    var b = b_in;
    for (var col = 0; col < 6; col++) {
        let max_row = i32(bitcast<u32>(setup[PIV + u32(col)]));
        if (max_row >= 6) {
            continue;
        }
        if (max_row != col) {
            let tb = b[col];
            b[col] = b[max_row];
            b[max_row] = tb;
        }
        for (var row = col + 1; row < 6; row++) {
            let k = u32(col * 6 + row);
            if (setup[USE + k] == 0.0) {
                continue;
            }
            let factor = setup[FAC + k];
            b[row] = b[row] - factor * b[col];
        }
    }
    var x: array<f32, 6>;
    for (var i = 5; i >= 0; i--) {
        let aii = setup[UPPER + u32(i * 6 + i)];
        if (abs(aii) < 1e-12) {
            x[i] = 0.0;
            continue;
        }
        var sum = b[i];
        for (var j = i + 1; j < 6; j++) {
            sum = sum - setup[UPPER + u32(i * 6 + j)] * x[j];
        }
        let xi = sum / aii;
        x[i] = select(0.0, xi, xi == xi);
    }
    return x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.src_w || y >= params.src_h) {
        return;
    }

    let idx = y * params.stride + x;
    let half = i32(min(params.poly_n, 64u));
    let last_x = i32(params.src_w) - 1;
    let last_y = i32(params.src_h) - 1;

    var atb: array<f32, 6>;

    for (var dy = -half; dy <= half; dy++) {
        let sy = clamp(i32(y) + dy, 0, last_y);
        let row = u32(sy) * params.stride;
        let wy = setup[GAUSS + u32(dy + half)];
        for (var dx = -half; dx <= half; dx++) {
            let sx = clamp(i32(x) + dx, 0, last_x);
            let val = input[row + u32(sx)];
            let xf = f32(dx);
            let yf = f32(dy);
            let w = setup[GAUSS + u32(dx + half)] * wy;
            let basis = array<f32, 6>(1.0, xf, yf, xf * xf, xf * yf, yf * yf);
            for (var i = 0; i < 6; i++) {
                atb[i] = atb[i] + basis[i] * val * w;
            }
        }
    }

    let sol = solve6(atb);
    // [f, d, e, a, b, c] = [1, x, y, x², xy, y²]
    output[idx * 6u + 0u] = sol[0];
    output[idx * 6u + 1u] = sol[1];
    output[idx * 6u + 2u] = sol[2];
    output[idx * 6u + 3u] = sol[3];
    output[idx * 6u + 4u] = sol[4];
    output[idx * 6u + 5u] = sol[5];
}
