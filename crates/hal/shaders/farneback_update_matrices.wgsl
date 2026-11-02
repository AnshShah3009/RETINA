// Farneback: per-pixel normal-equation terms for the current flow.
// Port of OpenCV's FarnebackUpdateMatrices (modules/video/src/optflowgf.cpp).
//
// poly1 / poly2 hold six coefficients per pixel, [f, d, e, a, b, c] for
// I ≈ f + d x + e y + a x² + b xy + c y² (see farneback_poly_expand.wgsl).
// In the paper's notation A = [[a, b/2], [b/2, c]] and B = [d, e].
//
// For pixel p with current flow w, poly2 is sampled bilinearly at p + w and
//   A  = (A1(p) + A2(p + w)) / 2
//   dB = (B1(p) - B2(p + w)) / 2 + A w
// so that A w_new = dB is the total displacement, not a residual. When p + w
// leaves the image (bilinear footprint not inside), A = A1 and B2 = 0, as in
// OpenCV. Pixels within 5 of the border are down-weighted by OpenCV's table.
//
// Output, planar (plane k at k * width * height):
//   M0 = A11² + A12², M1 = (A11 + A22) A12, M2 = A22² + A12²,
//   M3 = A11 dB1 + A12 dB2, M4 = A12 dB1 + A22 dB2
// i.e. AᵀA and AᵀdB, which farneback_update_flow.wgsl averages over the
// window and solves.
//
// Flow is planar: u at [i], v at [width * height + i].

struct FlowParams {
    width: u32,
    height: u32,
    half_window: u32,
    win_sigma: f32,
}

@group(0) @binding(0) var<storage, read> poly1: array<f32>;
@group(0) @binding(1) var<storage, read> poly2: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read_write> mats: array<f32>;
@group(0) @binding(4) var<uniform> params: FlowParams;

// OpenCV's `border[]` for BORDER = 5: distance 0 and 1 from the edge weigh
// 0.14, distance 2..4 weigh 0.4472, the rest 1.
fn border_weight(d: u32) -> f32 {
    if (d < 2u) {
        return 0.14;
    }
    if (d < 5u) {
        return 0.4472;
    }
    return 1.0;
}

fn p2(i: u32, k: u32) -> f32 {
    return poly2[i * 6u + k];
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
    let dx = flow[i];
    let dy = flow[n + i];

    let d1 = poly1[i * 6u + 1u];
    let e1 = poly1[i * 6u + 2u];
    let a1 = poly1[i * 6u + 3u];
    let b1 = poly1[i * 6u + 4u];
    let c1 = poly1[i * 6u + 5u];

    let fx = f32(x) + dx;
    let fy = f32(y) + dy;
    let fx0 = floor(fx);
    let fy0 = floor(fy);

    var r2 = 0.0;
    var r3 = 0.0;
    var r4 = a1;
    var r5 = c1;
    var r6 = b1 * 0.5;

    // OpenCV: (unsigned)x1 < (unsigned)(width - 1), same for y. Compared in
    // float so a huge or non-finite flow cannot wrap an integer conversion.
    if (fx0 >= 0.0 && fy0 >= 0.0 && fx0 < f32(w) - 1.0 && fy0 < f32(h) - 1.0) {
        let x1 = u32(fx0);
        let y1 = u32(fy0);
        let ax = fx - fx0;
        let ay = fy - fy0;
        let w00 = (1.0 - ax) * (1.0 - ay);
        let w01 = ax * (1.0 - ay);
        let w10 = (1.0 - ax) * ay;
        let w11 = ax * ay;
        let j00 = y1 * w + x1;
        let j01 = j00 + 1u;
        let j10 = j00 + w;
        let j11 = j10 + 1u;

        let d2 = w00 * p2(j00, 1u) + w01 * p2(j01, 1u) + w10 * p2(j10, 1u) + w11 * p2(j11, 1u);
        let e2 = w00 * p2(j00, 2u) + w01 * p2(j01, 2u) + w10 * p2(j10, 2u) + w11 * p2(j11, 2u);
        let a2 = w00 * p2(j00, 3u) + w01 * p2(j01, 3u) + w10 * p2(j10, 3u) + w11 * p2(j11, 3u);
        let b2 = w00 * p2(j00, 4u) + w01 * p2(j01, 4u) + w10 * p2(j10, 4u) + w11 * p2(j11, 4u);
        let c2 = w00 * p2(j00, 5u) + w01 * p2(j01, 5u) + w10 * p2(j10, 5u) + w11 * p2(j11, 5u);

        r2 = d2;
        r3 = e2;
        r4 = (a1 + a2) * 0.5;
        r5 = (c1 + c2) * 0.5;
        r6 = (b1 + b2) * 0.25;
    }

    r2 = (d1 - r2) * 0.5;
    r3 = (e1 - r3) * 0.5;

    r2 = r2 + r4 * dx + r6 * dy;
    r3 = r3 + r6 * dx + r5 * dy;

    let scale = border_weight(x) * border_weight(w - 1u - x) * border_weight(y)
        * border_weight(h - 1u - y);
    r2 = r2 * scale;
    r3 = r3 * scale;
    r4 = r4 * scale;
    r5 = r5 * scale;
    r6 = r6 * scale;

    mats[i] = r4 * r4 + r6 * r6;
    mats[n + i] = (r4 + r5) * r6;
    mats[2u * n + i] = r5 * r5 + r6 * r6;
    mats[3u * n + i] = r4 * r2 + r6 * r3;
    mats[4u * n + i] = r6 * r2 + r5 * r3;
}
