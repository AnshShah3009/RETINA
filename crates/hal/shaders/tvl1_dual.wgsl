// TV-L1 optical flow: one dual step. Port of OpenCV DualTVL1's forwardGradient
// + estimateDualVariables:
//   p = (p + taut ∇u) / (1 + taut |∇u|),   taut = tau / theta
// for each flow component. ∇ is the forward difference, zero across the last
// column (x component) and the last row (y component). Only this pixel's dual
// variables are written and `flow` is only read, so the update is in place.
//
// Planar layouts (n = width * height): flow = u1, u2;
// dual = p11, p12, p21, p22.

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

@group(0) @binding(0) var<storage, read> flow: array<f32>;
@group(0) @binding(1) var<storage, read_write> dual: array<f32>;
@group(0) @binding(2) var<uniform> params: Tvl1Params;

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

    var u1x = 0.0;
    var u2x = 0.0;
    if (x + 1u < w) {
        u1x = flow[i + 1u] - u1;
        u2x = flow[n + i + 1u] - u2;
    }
    var u1y = 0.0;
    var u2y = 0.0;
    if (y + 1u < h) {
        u1y = flow[i + w] - u1;
        u2y = flow[n + i + w] - u2;
    }

    let taut = params.taut;
    let ng1 = 1.0 + taut * sqrt(u1x * u1x + u1y * u1y);
    let ng2 = 1.0 + taut * sqrt(u2x * u2x + u2y * u2y);

    dual[i] = (dual[i] + taut * u1x) / ng1;
    dual[n + i] = (dual[n + i] + taut * u1y) / ng1;
    dual[2u * n + i] = (dual[2u * n + i] + taut * u2x) / ng2;
    dual[3u * n + i] = (dual[3u * n + i] + taut * u2y) / ng2;
}
