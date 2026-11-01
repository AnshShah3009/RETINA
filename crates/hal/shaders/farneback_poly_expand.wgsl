// Polynomial expansion for Farneback optical flow.
//
// Uniform layout matches `FarnebackParams` in `gpu_kernels/optical_flow.rs`
// (#[repr(C)], 4-byte fields, no padding):
//   0 width, 4 height, 8 stride, 12 flow_stride, 16 window_size,
//   20 poly_n, 24 poly_sigma, 28 num_iters.
// Reading poly_n from byte 12 took `flow_stride` (16 on an 8-pixel level)
// and reading sigma from byte 16 took a zero, so the Gaussian was NaN.
//
// Each pixel is the weighted least-squares fit of
// I ≈ f + d x + e y + a x² + b xy + c y² on the poly_n window.
// A constant image has f equal to that constant and a = b = c = 0.

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

fn gaussian_weight(x: f32, sigma: f32) -> f32 {
    let two_sigma_sq = 2.0 * sigma * sigma;
    if (two_sigma_sq <= 0.0) {
        return 1.0;
    }
    let w = exp(-(x * x) / two_sigma_sq);
    if (w == w) {
        return w;
    }
    return 1.0;
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PolyExpandParams;

fn solve6(a_in: array<f32, 36>, b_in: array<f32, 6>) -> array<f32, 6> {
    var a = a_in;
    var b = b_in;
    for (var col = 0; col < 6; col++) {
        var max_row = col;
        var max_val = abs(a[col * 6 + col]);
        for (var row = col + 1; row < 6; row++) {
            let v = abs(a[row * 6 + col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }
        if (max_val < 1e-12) {
            for (var row = 0; row < 6; row++) {
                a[row * 6 + col] = 0.0;
            }
            continue;
        }
        if (max_row != col) {
            for (var j = 0; j < 6; j++) {
                let tmp = a[col * 6 + j];
                a[col * 6 + j] = a[max_row * 6 + j];
                a[max_row * 6 + j] = tmp;
            }
            let tb = b[col];
            b[col] = b[max_row];
            b[max_row] = tb;
        }
        let pivot = a[col * 6 + col];
        for (var row = col + 1; row < 6; row++) {
            let factor = a[row * 6 + col] / pivot;
            if (abs(factor) < 1e-15) {
                continue;
            }
            for (var j = col; j < 6; j++) {
                a[row * 6 + j] = a[row * 6 + j] - factor * a[col * 6 + j];
            }
            b[row] = b[row] - factor * b[col];
        }
    }
    var x: array<f32, 6>;
    for (var i = 5; i >= 0; i--) {
        if (abs(a[i * 6 + i]) < 1e-12) {
            x[i] = 0.0;
            continue;
        }
        var sum = b[i];
        for (var j = i + 1; j < 6; j++) {
            sum = sum - a[i * 6 + j] * x[j];
        }
        let xi = sum / a[i * 6 + i];
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
    let n = max(params.poly_n, 1u);
    let half = i32(n) / 2;

    var ata: array<f32, 36>;
    var atb: array<f32, 6>;

    for (var dy = -half; dy <= half; dy++) {
        for (var dx = -half; dx <= half; dx++) {
            let sx = i32(x) + dx;
            let sy = i32(y) + dy;
            if (sx < 0 || sy < 0 || sx >= i32(params.src_w) || sy >= i32(params.src_h)) {
                continue;
            }
            let val = input[u32(sy) * params.stride + u32(sx)];
            let xf = f32(dx);
            let yf = f32(dy);
            let w = gaussian_weight(xf, params.sigma) * gaussian_weight(yf, params.sigma);
            let basis = array<f32, 6>(1.0, xf, yf, xf * xf, xf * yf, yf * yf);
            for (var i = 0; i < 6; i++) {
                for (var j = 0; j < 6; j++) {
                    ata[i * 6 + j] = ata[i * 6 + j] + basis[i] * basis[j] * w;
                }
                atb[i] = atb[i] + basis[i] * val * w;
            }
        }
    }

    let sol = solve6(ata, atb);
    // [f, d, e, a, b, c] = [1, x, y, x², xy, y²]
    output[idx * 6u + 0u] = sol[0];
    output[idx * 6u + 1u] = sol[1];
    output[idx * 6u + 2u] = sol[2];
    output[idx * 6u + 3u] = sol[3];
    output[idx * 6u + 4u] = sol[4];
    output[idx * 6u + 5u] = sol[5];
}
