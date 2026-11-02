// TV-L1 optical flow: warp I1 by the current flow and linearise the data term.
// Port of OpenCV DualTVL1 (modules/optflow tvl1flow.cpp): buildFlowMap +
// remap(INTER_CUBIC) of I1 and its centred gradient + calcGradRho.
//
// For each pixel x with flow u0 = (u1, u2), at x + u0:
//   I1w, I1wx, I1wy = bicubic samples (Keys, A = -0.75, as OpenCV) of I1 and of
//                     its centred gradient (0.5 (I(x+1) - I(x-1)), replicated
//                     border, as OpenCV's centeredGradient)
//   grad  = I1wx² + I1wy²
//   rho_c = I1w - I1wx u1 - I1wy u2 - I0
// so rho(u) = rho_c + I1wx u1 + I1wy u2 is the linearised residual.
// Taps outside the image read zero, as OpenCV's remap (BORDER_CONSTANT) does.
// A replicated border instead looks like a plausible match where content
// leaves the frame, and pulled the flow there off by several pixels up to ~20
// pixels in from the edge; with zeros those pixels see a large residual and
// no gradient, so the TV term fills them in from the interior.
//
// Layouts are planar. flow: u1 at [i], u2 at [n + i]. warp: I1wx, I1wy, grad,
// rho_c at plane 0, 1, 2, 3 (n = width * height).
//
// The uniform is `Tvl1Params` in gpu_kernels/optical_flow.rs, shared by every
// TV-L1 shader.

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

@group(0) @binding(0) var<storage, read> i0: array<f32>;
@group(0) @binding(1) var<storage, read> i1: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read_write> warp: array<f32>;
@group(0) @binding(4) var<uniform> params: Tvl1Params;

// I1 with a replicated border, for the centred gradient.
fn pix(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return i1[u32(cy) * params.width + u32(cx)];
}

// OpenCV interpolateCubic.
fn cubic(t: f32) -> vec4<f32> {
    let a = -0.75;
    let c0 = ((a * (t + 1.0) - 5.0 * a) * (t + 1.0) + 8.0 * a) * (t + 1.0) - 4.0 * a;
    let c1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    let c2 = ((a + 2.0) * (1.0 - t) - (a + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0;
    return vec4<f32>(c0, c1, c2, 1.0 - c0 - c1 - c2);
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
    let u1 = flow[i];
    let u2 = flow[n + i];

    // Clamp before the integer conversion. At -2 or w + 1 the only tap with a
    // nonzero weight is the one at that (outside) coordinate, so every sample
    // further out reads zero as well.
    let sx = clamp(f32(x) + u1, -2.0, f32(w) + 1.0);
    let sy = clamp(f32(y) + u2, -2.0, f32(h) + 1.0);
    let fx0 = floor(sx);
    let fy0 = floor(sy);
    let cx = cubic(sx - fx0);
    let cy = cubic(sy - fy0);
    let x0 = i32(fx0) - 1;
    let y0 = i32(fy0) - 1;

    var iw = 0.0;
    var iwx = 0.0;
    var iwy = 0.0;
    for (var r = 0; r < 4; r++) {
        let ty = y0 + r;
        if (ty < 0 || ty >= i32(h)) {
            continue;
        }
        var row = 0.0;
        var row_x = 0.0;
        var row_y = 0.0;
        for (var c = 0; c < 4; c++) {
            let tx = x0 + c;
            if (tx < 0 || tx >= i32(w)) {
                continue;
            }
            let wc = cx[c];
            row = row + wc * pix(tx, ty);
            row_x = row_x + wc * 0.5 * (pix(tx + 1, ty) - pix(tx - 1, ty));
            row_y = row_y + wc * 0.5 * (pix(tx, ty + 1) - pix(tx, ty - 1));
        }
        iw = iw + cy[r] * row;
        iwx = iwx + cy[r] * row_x;
        iwy = iwy + cy[r] * row_y;
    }

    warp[i] = iwx;
    warp[n + i] = iwy;
    warp[2u * n + i] = iwx * iwx + iwy * iwy;
    warp[3u * n + i] = iw - iwx * u1 - iwy * u2 - i0[i];
}
