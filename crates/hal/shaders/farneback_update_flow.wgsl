// Flow update for Farneback. Uniform layout is `FarnebackParams`
// (see farneback_poly_expand.wgsl). `window_size` is the averaging window
// the CPU Farneback uses around each pixel; the Gaussian uses `poly_sigma`.

struct FlowUpdateParams {
    src_w: u32,
    src_h: u32,
    stride: u32,
    flow_stride: u32,
    window_size: u32,
    poly_n: u32,
    sigma: f32,
    num_iters: u32,
}

@group(0) @binding(0) var<storage, read> poly1: array<f32>;
@group(0) @binding(1) var<storage, read> poly2: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f32>;
@group(0) @binding(3) var<uniform> params: FlowUpdateParams;

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.src_w || y >= params.src_h) {
        return;
    }

    let flow_idx = y * params.flow_stride + x * 2u;
    var u = flow[flow_idx + 0u];
    var v = flow[flow_idx + 1u];

    let half = i32(max(params.window_size, 1u)) / 2;
    var g11_acc = 0.0;
    var g12_acc = 0.0;
    var g22_acc = 0.0;
    var b1_acc = 0.0;
    var b2_acc = 0.0;
    var w_acc = 0.0;

    for (var wy = -half; wy <= half; wy++) {
        for (var wx = -half; wx <= half; wx++) {
            let xx = i32(x) + wx;
            let yy = i32(y) + wy;
            if (xx < 0 || yy < 0 || xx >= i32(params.src_w) || yy >= i32(params.src_h)) {
                continue;
            }
            let wgt = gaussian_weight(f32(wx), params.sigma) * gaussian_weight(f32(wy), params.sigma);
            let pidx = u32(yy) * params.stride + u32(xx);
            let fidx = u32(yy) * params.flow_stride + u32(xx) * 2u;
            let uu = flow[fidx + 0u];
            let vv = flow[fidx + 1u];

            let d1 = poly1[pidx * 6u + 1u];
            let e1 = poly1[pidx * 6u + 2u];
            let a1 = poly1[pidx * 6u + 3u];
            let b1 = poly1[pidx * 6u + 4u];
            let c1 = poly1[pidx * 6u + 5u];

            let px = f32(xx) + uu;
            let py = f32(yy) + vv;
            let x0 = i32(floor(px));
            let y0 = i32(floor(py));
            let fx = px - f32(x0);
            let fy = py - f32(y0);
            let sx0 = clamp(x0, 0, i32(params.src_w) - 1);
            let sy0 = clamp(y0, 0, i32(params.src_h) - 1);
            let sx1 = clamp(x0 + 1, 0, i32(params.src_w) - 1);
            let sy1 = clamp(y0 + 1, 0, i32(params.src_h) - 1);
            let i00 = u32(sy0) * params.stride + u32(sx0);
            let i10 = u32(sy0) * params.stride + u32(sx1);
            let i01 = u32(sy1) * params.stride + u32(sx0);
            let i11 = u32(sy1) * params.stride + u32(sx1);

            let a2 = mix(mix(poly2[i00 * 6u + 3u], poly2[i10 * 6u + 3u], fx),
                         mix(poly2[i01 * 6u + 3u], poly2[i11 * 6u + 3u], fx), fy);
            let b2 = mix(mix(poly2[i00 * 6u + 4u], poly2[i10 * 6u + 4u], fx),
                         mix(poly2[i01 * 6u + 4u], poly2[i11 * 6u + 4u], fx), fy);
            let c2 = mix(mix(poly2[i00 * 6u + 5u], poly2[i10 * 6u + 5u], fx),
                         mix(poly2[i01 * 6u + 5u], poly2[i11 * 6u + 5u], fx), fy);
            let d2 = mix(mix(poly2[i00 * 6u + 1u], poly2[i10 * 6u + 1u], fx),
                         mix(poly2[i01 * 6u + 1u], poly2[i11 * 6u + 1u], fx), fy);
            let e2 = mix(mix(poly2[i00 * 6u + 2u], poly2[i10 * 6u + 2u], fx),
                         mix(poly2[i01 * 6u + 2u], poly2[i11 * 6u + 2u], fx), fy);

            let g11 = a1 + a2;
            let g12 = b1 + b2;
            let g22 = c1 + c2;
            let rhs1 = -d1 + (a2 * uu + b2 * vv + d2);
            let rhs2 = -e1 + (b2 * uu + c2 * vv + e2);

            g11_acc = g11_acc + wgt * g11;
            g12_acc = g12_acc + wgt * g12;
            g22_acc = g22_acc + wgt * g22;
            b1_acc = b1_acc + wgt * rhs1;
            b2_acc = b2_acc + wgt * rhs2;
            w_acc = w_acc + wgt;
        }
    }

    if (w_acc > 0.0) {
        let det = g11_acc * g22_acc - g12_acc * g12_acc;
        if (abs(det) > 0.0001) {
            u = u + (g22_acc * b1_acc - g12_acc * b2_acc) / det;
            v = v + (g11_acc * b2_acc - g12_acc * b1_acc) / det;
        }
    }

    flow[flow_idx + 0u] = u;
    flow[flow_idx + 1u] = v;
}
