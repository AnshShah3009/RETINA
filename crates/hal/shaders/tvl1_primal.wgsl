// TV-L1 optical flow: one primal step. Port of OpenCV DualTVL1's estimateV
// (thresholding of the linearised L1 data term), divergence and estimateU,
// fused: every quantity is read at this pixel except the dual variables,
// which this pass only reads, so `flow` is updated in place without a race.
//
//   rho = rho_c + I1wx u1 + I1wy u2
//   v   = u + l_t ∇I1w            if rho < -l_t |∇I1w|²
//         u - l_t ∇I1w            if rho >  l_t |∇I1w|²
//         u - rho ∇I1w / |∇I1w|²  otherwise
//   u   = v + theta div(p)
// with l_t = lambda * theta. div is OpenCV's backward-difference divergence
// (the adjoint of the forward gradient in tvl1_dual.wgsl).
//
// Planar layouts (n = width * height): warp = I1wx, I1wy, grad, rho_c;
// dual = p11, p12, p21, p22 (p1 = (p11, p12) is the dual of u1);
// flow = u1, u2.

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

@group(0) @binding(0) var<storage, read> warp: array<f32>;
@group(0) @binding(1) var<storage, read> dual: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f32>;
@group(0) @binding(3) var<uniform> params: Tvl1Params;

// div(px, py) at (x, y): px(x) - px(x - 1) + py(y) - py(y - 1), where the
// terms past the top or left edge are zero. `base_x` / `base_y` are the plane
// offsets of the two components.
fn divergence(x: u32, y: u32, base_x: u32, base_y: u32) -> f32 {
    let w = params.width;
    let i = y * w + x;
    var d = dual[base_x + i] + dual[base_y + i];
    if (x > 0u) {
        d = d - dual[base_x + i - 1u];
    }
    if (y > 0u) {
        d = d - dual[base_y + i - w];
    }
    return d;
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
    let gx = warp[i];
    let gy = warp[n + i];
    let grad = warp[2u * n + i];
    let rho_c = warp[3u * n + i];
    let u1 = flow[i];
    let u2 = flow[n + i];
    let l_t = params.l_t;

    let rho = rho_c + gx * u1 + gy * u2;
    var d1 = 0.0;
    var d2 = 0.0;
    if (rho < -l_t * grad) {
        d1 = l_t * gx;
        d2 = l_t * gy;
    } else if (rho > l_t * grad) {
        d1 = -l_t * gx;
        d2 = -l_t * gy;
    } else if (grad > 1.1920929e-7) {
        let fi = -rho / grad;
        d1 = fi * gx;
        d2 = fi * gy;
    }

    let div1 = divergence(x, y, 0u, n);
    let div2 = divergence(x, y, 2u * n, 3u * n);
    flow[i] = u1 + d1 + params.theta * div1;
    flow[n + i] = u2 + d2 + params.theta * div2;
}
