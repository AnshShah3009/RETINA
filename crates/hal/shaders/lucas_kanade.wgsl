// GPU pyramidal Lucas-Kanade sparse optical flow (Bouguet / OpenCV
// calcOpticalFlowPyrLK). One thread tracks one feature at one pyramid level.
//
// At level L the feature sits at p / 2^L. The template window of `prev` is
// sampled bilinearly around that point (the feature, never the running
// estimate); `next` is sampled bilinearly at p / 2^L + g + v, where g is the
// guess carried from the coarser level and v the in-level residual flow. On
// exit the guess for the next finer level is 2 (g + v) (or g + v at level 0).

struct Params {
    num_points: u32,
    window_radius: i32,
    max_iters: u32,
    // Relative conditioning floor: lambda_min(G) <= min_eigenvalue * trace(G)
    // means the window is untextured or aperture-limited and is not tracked.
    min_eigenvalue: f32,
}

struct Point {
    x: f32,
    y: f32,
}

@group(0) @binding(0) var<storage, read> prev_img: array<f32>;
@group(0) @binding(1) var<storage, read> next_img: array<f32>;
// Feature positions in level-0 coordinates.
@group(0) @binding(2) var<storage, read> points: array<Point>;
// Flow guess g at this level on entry; guess for the next level on exit.
@group(0) @binding(3) var<storage, read_write> guess: array<Point>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> level_params: vec4<u32>; // w, h, level, unused

// Bilinear sample with replicated border.
fn sample_prev(x: f32, y: f32) -> f32 {
    let w = level_params.x;
    let h = level_params.y;
    let cx = clamp(x, 0.0, f32(w - 1u));
    let cy = clamp(y, 0.0, f32(h - 1u));
    let x0 = min(u32(floor(cx)), w - 1u);
    let y0 = min(u32(floor(cy)), h - 1u);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let ax = cx - f32(x0);
    let ay = cy - f32(y0);
    let top = mix(prev_img[y0 * w + x0], prev_img[y0 * w + x1], ax);
    let bot = mix(prev_img[y1 * w + x0], prev_img[y1 * w + x1], ax);
    return mix(top, bot, ay);
}

fn sample_next(x: f32, y: f32) -> f32 {
    let w = level_params.x;
    let h = level_params.y;
    let cx = clamp(x, 0.0, f32(w - 1u));
    let cy = clamp(y, 0.0, f32(h - 1u));
    let x0 = min(u32(floor(cx)), w - 1u);
    let y0 = min(u32(floor(cy)), h - 1u);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let ax = cx - f32(x0);
    let ay = cy - f32(y0);
    let top = mix(next_img[y0 * w + x0], next_img[y0 * w + x1], ax);
    let bot = mix(next_img[y1 * w + x0], next_img[y1 * w + x1], ax);
    return mix(top, bot, ay);
}

// Central-difference gradient of prev at an integer pixel (replicated border),
// matching the CPU path.
fn grad_at(x: i32, y: i32) -> vec2<f32> {
    let w = i32(level_params.x);
    let h = i32(level_params.y);
    let xm = u32(clamp(x - 1, 0, w - 1));
    let xp = u32(clamp(x + 1, 0, w - 1));
    let ym = u32(clamp(y - 1, 0, h - 1));
    let yp = u32(clamp(y + 1, 0, h - 1));
    let xc = u32(clamp(x, 0, w - 1));
    let yc = u32(clamp(y, 0, h - 1));
    let uw = u32(w);
    return vec2<f32>(
        (prev_img[yc * uw + xp] - prev_img[yc * uw + xm]) * 0.5,
        (prev_img[yp * uw + xc] - prev_img[ym * uw + xc]) * 0.5,
    );
}

// Bilinear interpolation of the central-difference gradient image.
fn sample_grad(x: f32, y: f32) -> vec2<f32> {
    let w = level_params.x;
    let h = level_params.y;
    let cx = clamp(x, 0.0, f32(w - 1u));
    let cy = clamp(y, 0.0, f32(h - 1u));
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let ax = cx - f32(x0);
    let ay = cy - f32(y0);
    let top = mix(grad_at(x0, y0), grad_at(x0 + 1, y0), ax);
    let bot = mix(grad_at(x0, y0 + 1), grad_at(x0 + 1, y0 + 1), ax);
    return mix(top, bot, ay);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let w = f32(level_params.x);
    let h = f32(level_params.y);
    let level = level_params.z;
    let s = 1.0 / f32(1u << level);
    let p = vec2<f32>(points[idx].x * s, points[idx].y * s);
    let g = vec2<f32>(guess[idx].x, guess[idx].y);
    let r = params.window_radius;
    let rf = f32(r);

    var v = vec2<f32>(0.0, 0.0);

    let fits = p.x - rf >= 0.0 && p.x + rf <= w - 1.0 && p.y - rf >= 0.0 && p.y + rf <= h - 1.0;
    if (fits) {
        var G00 = 0.0;
        var G01 = 0.0;
        var G11 = 0.0;
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                let gr = sample_grad(p.x + f32(dx), p.y + f32(dy));
                G00 += gr.x * gr.x;
                G01 += gr.x * gr.y;
                G11 += gr.y * gr.y;
            }
        }
        let det = G00 * G11 - G01 * G01;
        let tr = G00 + G11;
        let min_eig = 0.5 * (tr - sqrt((G00 - G11) * (G00 - G11) + 4.0 * G01 * G01));
        if (det > 0.0 && min_eig > params.min_eigenvalue * tr) {
            let inv_det = 1.0 / det;
            var prev_delta = vec2<f32>(0.0, 0.0);
            for (var iter = 0u; iter < params.max_iters; iter++) {
                let c = p + g + v;
                if (!(c.x >= 0.0 && c.x <= w - 1.0 && c.y >= 0.0 && c.y <= h - 1.0)) {
                    break;
                }
                var b0 = 0.0;
                var b1 = 0.0;
                for (var dy = -r; dy <= r; dy++) {
                    for (var dx = -r; dx <= r; dx++) {
                        let tx = p.x + f32(dx);
                        let ty = p.y + f32(dy);
                        let gr = sample_grad(tx, ty);
                        let it = sample_next(c.x + f32(dx), c.y + f32(dy)) - sample_prev(tx, ty);
                        b0 += it * gr.x;
                        b1 += it * gr.y;
                    }
                }
                // delta = -G^{-1} b
                let delta = vec2<f32>(
                    (G01 * b1 - G11 * b0) * inv_det,
                    (G01 * b0 - G00 * b1) * inv_det,
                );
                v += delta;
                if (dot(delta, delta) <= 1e-4) { break; }
                // OpenCV's oscillation guard.
                if (iter > 0u && abs(delta.x + prev_delta.x) < 0.01 && abs(delta.y + prev_delta.y) < 0.01) {
                    v -= 0.5 * delta;
                    break;
                }
                prev_delta = delta;
            }
        }
    }

    var out = g + v;
    if (level > 0u) {
        out = out * 2.0;
    }
    guess[idx] = Point(out.x, out.y);
}
