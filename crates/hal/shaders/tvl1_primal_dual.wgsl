// TV-L1 optical flow: one inner iteration, the primal step followed by the
// dual step. Port of OpenCV DualTVL1's estimateV + divergence + estimateU
// (primal) and forwardGradient + estimateDualVariables (dual).
//
// Primal, with l_t = lambda * theta:
//   rho = rho_c + I1wx u1 + I1wy u2
//   v   = u + l_t ∇I1w            if rho < -l_t |∇I1w|²
//         u - l_t ∇I1w            if rho >  l_t |∇I1w|²
//         u - rho ∇I1w / |∇I1w|²  otherwise
//   u'  = v + theta div(p)
// div is OpenCV's backward-difference divergence (the adjoint of the forward
// gradient below).
//
// Dual, with taut = tau / theta, for each flow component:
//   p' = (p + taut ∇u') / (1 + taut |∇u'|)
// ∇ is the forward difference, zero across the last column (x component) and
// the last row (y component).
//
// The two steps used to be two dispatches, primal in place on the flow and
// then dual in place on the dual variables. Fusing them halves the dispatches
// (and the barriers between them) of the solver's hot loop. The dual step at
// a pixel needs u' at its right and lower neighbours, which another
// invocation owns, so each invocation evaluates the primal step there too, by
// the same function on the same inputs: every u' it uses is bit-identical to
// the one the owner writes. Because neighbours read the old state while it is
// being replaced, the state is ping-ponged: read `src`, write `dst`. Within a
// 16x16 workgroup the u' values are shared through workgroup memory, so only
// the halo column and row are evaluated twice.
//
// State layout (n = width * height), planar: u1, u2 at planes 0, 1 (so the
// warp, median and upsample shaders see the first 2n as a flow), then p11,
// p12, p21, p22 at planes 2..5 (p1 = (p11, p12) is the dual of u1).
// warp = I1wx, I1wy, grad, rho_c (see tvl1_warp.wgsl).

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
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Tvl1Params;

// div(px, py) at (x, y): px(x) - px(x - 1) + py(y) - py(y - 1), where the
// terms past the top or left edge are zero. `base_x` / `base_y` are the plane
// offsets of the two components.
fn divergence(x: u32, y: u32, base_x: u32, base_y: u32) -> f32 {
    let w = params.width;
    let i = y * w + x;
    var d = src[base_x + i] + src[base_y + i];
    if (x > 0u) {
        d = d - src[base_x + i - 1u];
    }
    if (y > 0u) {
        d = d - src[base_y + i - w];
    }
    return d;
}

// The primal step at (x, y): the updated (u1, u2).
fn primal(x: u32, y: u32) -> vec2<f32> {
    let w = params.width;
    let n = w * params.height;
    let i = y * w + x;
    let gx = warp[i];
    let gy = warp[n + i];
    let grad = warp[2u * n + i];
    let rho_c = warp[3u * n + i];
    let u1 = src[i];
    let u2 = src[n + i];
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

    let div1 = divergence(x, y, 2u * n, 3u * n);
    let div2 = divergence(x, y, 4u * n, 5u * n);
    return vec2<f32>(u1 + d1 + params.theta * div1, u2 + d2 + params.theta * div2);
}

// u' of the workgroup's 16x16 pixels plus the column to their right and the
// row below them (the corner is never needed): [ty * TILE_W + tx].
const TILE: u32 = 16u;
const TILE_W: u32 = 17u;
var<workgroup> tile: array<vec2<f32>, 289>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let x = global_id.x;
    let y = global_id.y;
    let w = params.width;
    let h = params.height;
    let inside = x < w && y < h;

    // Each u' is evaluated once per workgroup and shared, instead of three
    // times per pixel (here and as the right / lower neighbour of two
    // others). The first 32 invocations also evaluate the halo: 0..16 the
    // column at x0 + 16, 16..32 the row at y0 + 16.
    if (inside) {
        tile[lid.y * TILE_W + lid.x] = primal(x, y);
    }
    let x0 = wid.x * TILE;
    let y0 = wid.y * TILE;
    if (li < TILE) {
        let hx = x0 + TILE;
        let hy = y0 + li;
        if (hx < w && hy < h) {
            tile[li * TILE_W + TILE] = primal(hx, hy);
        }
    } else if (li < 2u * TILE) {
        let hx = x0 + li - TILE;
        let hy = y0 + TILE;
        if (hx < w && hy < h) {
            tile[TILE * TILE_W + li - TILE] = primal(hx, hy);
        }
    }
    workgroupBarrier();

    if (!inside) {
        return;
    }

    let n = w * h;
    let i = y * w + x;
    let t = lid.y * TILE_W + lid.x;
    let u = tile[t];
    let u1 = u.x;
    let u2 = u.y;

    var u1x = 0.0;
    var u2x = 0.0;
    if (x + 1u < w) {
        let r = tile[t + 1u];
        u1x = r.x - u1;
        u2x = r.y - u2;
    }
    var u1y = 0.0;
    var u2y = 0.0;
    if (y + 1u < h) {
        let d = tile[t + TILE_W];
        u1y = d.x - u1;
        u2y = d.y - u2;
    }

    let taut = params.taut;
    let ng1 = 1.0 + taut * sqrt(u1x * u1x + u1y * u1y);
    let ng2 = 1.0 + taut * sqrt(u2x * u2x + u2y * u2y);

    dst[i] = u1;
    dst[n + i] = u2;
    dst[2u * n + i] = (src[2u * n + i] + taut * u1x) / ng1;
    dst[3u * n + i] = (src[3u * n + i] + taut * u1y) / ng1;
    dst[4u * n + i] = (src[4u * n + i] + taut * u2x) / ng2;
    dst[5u * n + i] = (src[5u * n + i] + taut * u2y) / ng2;
}
