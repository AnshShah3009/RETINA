// Image-independent half of Farneback's polynomial expansion
// (`farneback_poly_expand.wgsl`), run once per `farneback` call.
//
// Every pixel's fit solves A s = b with the same 6x6 Gram matrix A (samples
// past the edge replicate the border, so the window never shrinks) and the
// same Gaussian weights; only b depends on the image. The expansion used to
// rebuild A, two exp() per neighbour and the whole elimination of A at every
// pixel. This pass does that work once, with the same expressions in the same
// order, and records what the per-pixel solve needs:
//
//   [PIV + col]        pivot row chosen for `col`, as u32 bits; 6 when the
//                      column was zeroed (max |a| < 1e-12) and skipped
//   [FAC + col*6+row]  elimination factor a[row][col] / pivot (row > col)
//   [USE + col*6+row]  1 when that factor was applied (|factor| >= 1e-15)
//   [UPPER + i*6+j]    A after elimination, for back substitution
//   [GAUSS + k]        1-D Gaussian weight of offset k - poly_n
//
// so `farneback_poly_expand.wgsl` replays exactly the operations the full
// solve applied to b, and its output is bit-identical to solving per pixel.
//
// Uniform layout matches `FarnebackParams` (see the poly-expand shader).

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

const PIV: u32 = 0u;
const FAC: u32 = 6u;
const USE: u32 = 42u;
const UPPER: u32 = 78u;
const GAUSS: u32 = 114u;

@group(0) @binding(0) var<storage, read_write> setup: array<f32>;
@group(0) @binding(1) var<uniform> params: PolyExpandParams;

var<workgroup> gram: array<f32, 36>;

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

fn basis_at(k: u32, xf: f32, yf: f32) -> f32 {
    let basis = array<f32, 6>(1.0, xf, yf, xf * xf, xf * yf, yf * yf);
    return basis[k];
}

// One workgroup. Invocations 0..36 each accumulate one Gram entry over the
// window in the same neighbour order as the per-pixel loop did; invocation 0
// then eliminates.
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    let half = i32(min(params.poly_n, 64u));
    let side = u32(2 * half + 1);

    for (var k = li; k < side; k += 64u) {
        setup[GAUSS + k] = gaussian_weight(f32(i32(k) - half), params.sigma);
    }

    if (li < 36u) {
        let i = li / 6u;
        let j = li % 6u;
        var acc = 0.0;
        for (var dy = -half; dy <= half; dy++) {
            for (var dx = -half; dx <= half; dx++) {
                let xf = f32(dx);
                let yf = f32(dy);
                let w = gaussian_weight(xf, params.sigma) * gaussian_weight(yf, params.sigma);
                acc = acc + basis_at(i, xf, yf) * basis_at(j, xf, yf) * w;
            }
        }
        gram[li] = acc;
    }
    workgroupBarrier();
    if (li != 0u) {
        return;
    }

    var a: array<f32, 36>;
    for (var k = 0; k < 36; k++) {
        a[k] = gram[k];
        setup[FAC + u32(k)] = 0.0;
        setup[USE + u32(k)] = 0.0;
    }
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
            setup[PIV + u32(col)] = bitcast<f32>(6u);
            continue;
        }
        setup[PIV + u32(col)] = bitcast<f32>(u32(max_row));
        if (max_row != col) {
            for (var j = 0; j < 6; j++) {
                let tmp = a[col * 6 + j];
                a[col * 6 + j] = a[max_row * 6 + j];
                a[max_row * 6 + j] = tmp;
            }
        }
        let pivot = a[col * 6 + col];
        for (var row = col + 1; row < 6; row++) {
            let factor = a[row * 6 + col] / pivot;
            setup[FAC + u32(col * 6 + row)] = factor;
            if (abs(factor) < 1e-15) {
                continue;
            }
            setup[USE + u32(col * 6 + row)] = 1.0;
            for (var j = col; j < 6; j++) {
                a[row * 6 + j] = a[row * 6 + j] - factor * a[col * 6 + j];
            }
        }
    }
    for (var k = 0; k < 36; k++) {
        setup[UPPER + u32(k)] = a[k];
    }
}
