//! GPU dense optical flow (`gpu_kernels::optical_flow::{farneback,
//! tvl1_optical_flow}`): layout, accuracy against known motion, and parity
//! with OpenCV.
//!
//! Both return a `(2, h, w)` tensor. It used to be labelled CHW while the
//! Farneback shaders stored (u, v) interleaved per pixel, so indexing it by
//! its shape read garbage. The kernels were also wrong beyond the layout:
//! Farneback's flow update had the sign of the data term flipped, mixed the
//! paper's A and b scales, updated the flow in place while neighbours read it,
//! and weighted the window by `poly_sigma` instead of `window_size`; it
//! diverged (mean endpoint error ~1e7 px on a 1.5 px shift). TV-L1 never
//! applied its TV term (the divergence it computed was never read), read its
//! dual step from a buffer a third the size it indexed, warped already-warped
//! images, had no pyramid and used theta / tau for lambda * theta; its error
//! (6-12 px) was worse than reporting zero flow.
//!
//! The frames are a smooth sinusoidal texture, 8-bit values, and the same
//! texture moved by a known affine motion, so the true flow is known at every
//! pixel. Endpoint error (EPE) is measured away from the border, where the
//! content of `next` has no counterpart in `prev`. The bounds are about twice
//! the error measured on an RTX 5070 Ti (Vulkan); the values are in each
//! test's comment. Skips when no adapter is available.

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::optical_flow::{FlowBlurMode, Tvl1Config, farneback, tvl1_optical_flow};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{GpuTensor, Result};
use pollster::block_on;

fn require_gpu() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU dense optical flow test (no adapter): {e}");
            None
        }
    }
}

/// Sum of 14 plane waves (wavelengths 7..32 px) around 128.
fn texture(x: f64, y: f64) -> f64 {
    let mut s = 0x1234_5678u32;
    let mut r = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f64 / (1u32 << 24) as f64
    };
    let mut v = 128.0;
    for _ in 0..14 {
        let wl = 7.0 + 25.0 * r();
        let th = std::f64::consts::TAU * r();
        let ph = std::f64::consts::TAU * r();
        let a = 9.0 + 6.0 * r();
        v += a * (std::f64::consts::TAU * (th.cos() / wl * x + th.sin() / wl * y) + ph).sin();
    }
    v.clamp(0.0, 255.0)
}

/// A pair moved by `p -> m (p - c) + c + t` about the centre `c`, and its flow.
struct Pair {
    w: usize,
    h: usize,
    prev: Vec<f32>,
    next: Vec<f32>,
    /// Ground truth, planar: u then v.
    flow: Vec<f32>,
}

fn pair(w: usize, h: usize, m: [[f64; 2]; 2], t: [f64; 2]) -> Pair {
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv = [
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
    ];
    let n = w * h;
    let mut prev = vec![0f32; n];
    let mut next = vec![0f32; n];
    let mut flow = vec![0f32; 2 * n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (px, py) = (x as f64 - cx, y as f64 - cy);
            prev[i] = texture(x as f64, y as f64).round() as f32;
            // next(q) = prev(p) where q = m p + t (about the centre).
            let (qx, qy) = (px - t[0], py - t[1]);
            let sx = inv[0][0] * qx + inv[0][1] * qy + cx;
            let sy = inv[1][0] * qx + inv[1][1] * qy + cy;
            next[i] = texture(sx, sy).round() as f32;
            flow[i] = (m[0][0] * px + m[0][1] * py + t[0] - px) as f32;
            flow[n + i] = (m[1][0] * px + m[1][1] * py + t[1] - py) as f32;
        }
    }
    Pair {
        w,
        h,
        prev,
        next,
        flow,
    }
}

fn translation(w: usize, h: usize, dx: f64, dy: f64) -> Pair {
    pair(w, h, [[1.0, 0.0], [0.0, 1.0]], [dx, dy])
}

/// Rotation by 3 degrees and zoom by 1.04 about the centre.
fn rotation_zoom(w: usize, h: usize) -> Pair {
    let (s, th) = (1.04f64, 3f64.to_radians());
    pair(
        w,
        h,
        [[s * th.cos(), -s * th.sin()], [s * th.sin(), s * th.cos()]],
        [0.0, 0.0],
    )
}

fn upload(ctx: &GpuContext, data: &[f32], w: usize, h: usize) -> GpuTensor<f32> {
    let t: CpuTensor<f32> = Tensor::from_vec(data.to_vec(), TensorShape::new(1, h, w)).unwrap();
    t.to_gpu_ctx(ctx).unwrap()
}

type FlowFn<'a> = &'a dyn Fn(&GpuTensor<f32>, &GpuTensor<f32>) -> Result<GpuTensor<f32>>;

/// Runs `method` on the pair and returns the flow on the CPU, checking the
/// shape it reports.
fn run(ctx: &GpuContext, p: &Pair, method: FlowFn<'_>) -> CpuTensor<f32> {
    let prev = upload(ctx, &p.prev, p.w, p.h);
    let next = upload(ctx, &p.next, p.w, p.h);
    let flow = method(&prev, &next).unwrap();
    assert_eq!(flow.shape, TensorShape::new(2, p.h, p.w));
    let flow = flow.to_cpu_ctx(ctx).unwrap();
    assert_eq!(flow.as_slice().unwrap().len(), 2 * p.w * p.h);
    flow
}

/// Mean and max endpoint error over the pixels at least `margin` from the
/// border, reading the flow through `Tensor::index` (plane, row, column).
fn epe(flow: &CpuTensor<f32>, p: &Pair, margin: usize) -> (f64, f64) {
    let n = p.w * p.h;
    let (mut sum, mut max, mut count) = (0.0f64, 0.0f64, 0usize);
    for y in margin..p.h - margin {
        for x in margin..p.w - margin {
            let u = flow.index(0, y, x).unwrap();
            let v = flow.index(1, y, x).unwrap();
            let i = y * p.w + x;
            let e = ((u - p.flow[i]) as f64).hypot((v - p.flow[n + i]) as f64);
            assert!(e.is_finite(), "non-finite flow at ({x}, {y}): ({u}, {v})");
            sum += e;
            max = max.max(e);
            count += 1;
        }
    }
    (sum / count as f64, max)
}

fn fb(
    ctx: &'static GpuContext,
    levels: u32,
    mode: FlowBlurMode,
) -> impl Fn(&GpuTensor<f32>, &GpuTensor<f32>) -> Result<GpuTensor<f32>> {
    move |a, b| farneback(ctx, a, b, levels, 0.5, 15, 3, 5, 1.2, mode)
}

fn tvl1(
    ctx: &'static GpuContext,
) -> impl Fn(&GpuTensor<f32>, &GpuTensor<f32>) -> Result<GpuTensor<f32>> {
    move |a, b| tvl1_optical_flow(ctx, a, b, Tvl1Config::default())
}

/// A pure translation: every pixel of plane 0 is dx and every pixel of plane 1
/// is dy, away from the border. The frame is not square, so a transposed or
/// interleaved layout cannot pass by accident.
fn assert_translation_layout(name: &str, flow: &CpuTensor<f32>, p: &Pair, dx: f32, dy: f32) {
    let margin = 16;
    for y in margin..p.h - margin {
        for x in margin..p.w - margin {
            let u = flow.index(0, y, x).unwrap();
            let v = flow.index(1, y, x).unwrap();
            assert!(
                (u - dx).abs() < 0.1 && (v - dy).abs() < 0.1,
                "{name}: flow at ({x}, {y}) is ({u}, {v}), want ({dx}, {dy}); plane 0 must be u, plane 1 v"
            );
        }
    }
}

#[test]
fn farneback_returns_planar_u_then_v() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let p = translation(160, 112, 1.5, -0.75);
    for mode in [FlowBlurMode::Box, FlowBlurMode::Gaussian] {
        let flow = run(ctx, &p, &fb(ctx, 3, mode));
        assert_translation_layout(&format!("farneback {mode:?}"), &flow, &p, 1.5, -0.75);
    }
}

#[test]
fn tvl1_returns_planar_u_then_v() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let p = translation(160, 112, 1.5, -0.75);
    let flow = run(ctx, &p, &tvl1(ctx));
    assert_translation_layout("tvl1", &flow, &p, 1.5, -0.75);
}

/// (case, mean EPE bound, max EPE bound) per method.
fn accuracy_cases() -> Vec<(&'static str, Pair)> {
    vec![
        (
            "translation (1.5, -0.75)",
            translation(160, 112, 1.5, -0.75),
        ),
        ("rotation 3 deg + zoom 1.04", rotation_zoom(160, 112)),
        ("translation (4, 3)", translation(160, 112, 4.0, 3.0)),
    ]
}

fn check_accuracy(ctx: &GpuContext, method: &str, f: FlowFn<'_>, bounds: &[(f64, f64)]) {
    for ((case, p), (mean_bound, max_bound)) in accuracy_cases().iter().zip(bounds) {
        let flow = run(ctx, p, f);
        let (mean, max) = epe(&flow, p, 16);
        println!("{method} {case}: EPE mean {mean:.4} max {max:.4}");
        assert!(
            mean < *mean_bound && max < *max_bound,
            "{method} {case}: EPE mean {mean:.4} (bound {mean_bound}), max {max:.4} (bound {max_bound})"
        );
    }
}

#[test]
fn farneback_box_matches_known_motion() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    check_accuracy(
        ctx,
        "farneback box",
        &fb(ctx, 3, FlowBlurMode::Box),
        // Measured: 0.0077 / 0.026, 0.128 / 0.362, 0.0007 / 0.0034.
        &[(0.015, 0.05), (0.2, 0.6), (0.002, 0.01)],
    );
}

#[test]
fn farneback_gaussian_matches_known_motion() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    check_accuracy(
        ctx,
        "farneback gaussian",
        &fb(ctx, 3, FlowBlurMode::Gaussian),
        // Measured: 0.0189 / 0.084, 0.098 / 0.375, 0.0011 / 0.0070.
        &[(0.04, 0.15), (0.2, 0.6), (0.003, 0.015)],
    );
}

#[test]
fn tvl1_matches_known_motion() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    check_accuracy(
        ctx,
        "tvl1",
        &tvl1(ctx),
        // Measured: 0.043 / 0.062, 0.045 / 0.178, 0.0043 / 0.020. OpenCV 5.0
        // DualTVL1 gives 0.043 on the 128x128 version of the first pair: the
        // bias is the bicubic warp of a 7 px wavelength, not the port.
        &[(0.08, 0.12), (0.09, 0.35), (0.01, 0.04)],
    );
}

/// A shift of (9, -7) px is out of reach of a single level (the polynomial
/// fit and the window see a few pixels), so it is only found when each level
/// hands the next a flow scaled by the size ratio. A missing or wrong scale
/// between levels leaves a large error.
///
/// TV-L1 runs with 10 scales: with OpenCV's 5 scales of 0.8 the coarsest
/// scale still sees ~4.7 px of motion on a texture with 7 px wavelengths, and
/// OpenCV 5.0 DualTVL1 with its defaults misses this shift too (mean EPE
/// 10.5 px on the 256x256 version of this pair; 0.005 with 10 scales).
#[test]
fn pyramids_carry_large_motion_between_levels() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let p = translation(256, 192, 9.0, -7.0);
    let single = run(ctx, &p, &fb(ctx, 1, FlowBlurMode::Box));
    let (single_mean, _) = epe(&single, &p, 24);
    let pyramid = run(ctx, &p, &fb(ctx, 4, FlowBlurMode::Box));
    let (fb_mean, fb_max) = epe(&pyramid, &p, 24);
    let config = Tvl1Config {
        num_scales: 10,
        ..Default::default()
    };
    let tv = run(ctx, &p, &|a, b| tvl1_optical_flow(ctx, a, b, config));
    let (tv_mean, tv_max) = epe(&tv, &p, 24);
    println!(
        "(9, -7): farneback 1 level {single_mean:.4}; 4 levels mean {fb_mean:.4} max {fb_max:.4}; \
         tvl1 (10 scales) mean {tv_mean:.4} max {tv_max:.4}"
    );
    // Measured: one level 9.0; four levels 0.0016 / 0.0052; TV-L1 0.0044 /
    // 0.021.
    assert!(single_mean > 5.0, "one level should not reach 11 px");
    assert!(
        fb_mean < 0.005 && fb_max < 0.015,
        "farneback 4 levels: {fb_mean} / {fb_max}"
    );
    assert!(
        tv_mean < 0.01 && tv_max < 0.05,
        "tvl1 10 scales: {tv_mean} / {tv_max}"
    );
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/optical_flow")
        .join(name)
}

/// The OpenCV parity pair (`generate_fixtures.py --dense-flow-parity`): a
/// 96x80 texture rotated by 3 degrees and zoomed by 1.04.
fn parity_frames(ctx: &GpuContext) -> (GpuTensor<f32>, GpuTensor<f32>, usize, usize) {
    let frame =
        |name: &str| -> ndarray::Array2<f32> { ndarray_npy::read_npy(fixture(name)).unwrap() };
    let prev = frame("dense_parity_frame1.npy");
    let next = frame("dense_parity_frame2.npy");
    let (h, w) = prev.dim();
    (
        upload(ctx, prev.as_slice().unwrap(), w, h),
        upload(ctx, next.as_slice().unwrap(), w, h),
        w,
        h,
    )
}

/// (interior mean, interior max, whole-frame max) of |got - want|, interior =
/// at least 8 px from the border. `want` is OpenCV's planar [2, H, W].
fn parity(got: &CpuTensor<f32>, want_file: &str, w: usize, h: usize) -> (f64, f64, f64) {
    let want: ndarray::Array3<f32> = ndarray_npy::read_npy(fixture(want_file)).unwrap();
    assert_eq!(want.dim(), (2, h, w));
    let (mut sum, mut max, mut count, mut all_max) = (0.0f64, 0.0f64, 0, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let du = got.index(0, y, x).unwrap() - want[[0, y, x]];
            let dv = got.index(1, y, x).unwrap() - want[[1, y, x]];
            let d = (du as f64).hypot(dv as f64);
            all_max = all_max.max(d);
            if (8..h - 8).contains(&y) && (8..w - 8).contains(&x) {
                sum += d;
                max = max.max(d);
                count += 1;
            }
        }
    }
    (sum / count as f64, max, all_max)
}

/// OpenCV 5.0 `calcOpticalFlowFarneback(prev, next, None, 0.5, 2, 15, 3, 5,
/// 1.2, flags)`. OpenCV's `levels = 2` adds two levels to the frame, which is
/// `num_levels = 3` here (the 32-pixel floor stops both at two on 96x80).
#[test]
fn farneback_matches_opencv() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let (prev, next, w, h) = parity_frames(ctx);
    for (mode, tag, bounds) in [
        // Measured: 0.00041 / 0.0015 / 0.0090.
        (FlowBlurMode::Box, "box", (0.001, 0.004, 0.02)),
        // Measured: 0.00051 / 0.0032 / 0.069.
        (FlowBlurMode::Gaussian, "gauss", (0.0012, 0.008, 0.15)),
    ] {
        let got = farneback(ctx, &prev, &next, 3, 0.5, 15, 3, 5, 1.2, mode)
            .unwrap()
            .to_cpu_ctx(ctx)
            .unwrap();
        let (mean, max, all_max) = parity(&got, &format!("dense_parity_farneback_{tag}.npy"), w, h);
        println!(
            "farneback {tag} vs OpenCV: interior mean {mean:.5} max {max:.5}; whole frame max {all_max:.5}"
        );
        assert!(
            mean < bounds.0 && max < bounds.1 && all_max < bounds.2,
            "farneback {tag} vs OpenCV: interior mean {mean:.5} max {max:.5}, whole frame max {all_max:.5}"
        );
    }
}

/// OpenCV 5.0 contrib `DualTVL1OpticalFlow_create(epsilon=0)` with its other
/// defaults (epsilon 0 runs every iteration, as the GPU port does).
///
/// The interior agrees to a few thousandths of a pixel. The outermost two or
/// three pixels differ by up to ~1 px: there the flow points out of the frame,
/// the data term is void, and what remains is set by small differences such
/// as OpenCV mapping its 0.8x pyramid with exactly 1.25 instead of the size
/// ratio (96 / 77) and quantising remap coordinates to 1/32 px.
#[test]
fn tvl1_matches_opencv() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let (prev, next, w, h) = parity_frames(ctx);
    let got = tvl1_optical_flow(ctx, &prev, &next, Tvl1Config::default())
        .unwrap()
        .to_cpu_ctx(ctx)
        .unwrap();
    let (mean, max, all_max) = parity(&got, "dense_parity_tvl1.npy", w, h);
    println!("tvl1 vs OpenCV: interior mean {mean:.5} max {max:.5}; whole frame max {all_max:.5}");
    // Measured: 0.0021 / 0.119 / 1.14.
    assert!(
        mean < 0.005 && max < 0.25 && all_max < 2.5,
        "tvl1 vs OpenCV: interior mean {mean:.5} max {max:.5}, whole frame max {all_max:.5}"
    );
}

#[test]
fn dense_flow_rejects_bad_input() {
    let Some(ctx) = require_gpu() else {
        return;
    };
    let a = upload(ctx, &vec![0.0; 64 * 48], 64, 48);
    let b = upload(ctx, &vec![0.0; 48 * 64], 48, 64);
    let mode = FlowBlurMode::Box;
    assert!(farneback(ctx, &a, &b, 3, 0.5, 15, 3, 5, 1.2, mode).is_err());
    assert!(tvl1_optical_flow(ctx, &a, &b, Tvl1Config::default()).is_err());
    assert!(farneback(ctx, &a, &a, 0, 0.5, 15, 3, 5, 1.2, mode).is_err());
    assert!(farneback(ctx, &a, &a, 3, 1.0, 15, 3, 5, 1.2, mode).is_err());
    assert!(farneback(ctx, &a, &a, 3, 0.5, 15, 3, 5, 0.0, mode).is_err());
    for config in [
        Tvl1Config {
            scale_step: 1.0,
            ..Default::default()
        },
        Tvl1Config {
            median_filtering: 4,
            ..Default::default()
        },
        Tvl1Config {
            theta: 0.0,
            ..Default::default()
        },
    ] {
        assert!(tvl1_optical_flow(ctx, &a, &a, config).is_err());
    }
    let rgb = CpuTensor::<f32>::from_vec(vec![0.0f32; 3 * 64 * 48], TensorShape::new(3, 48, 64))
        .unwrap()
        .to_gpu_ctx(ctx)
        .unwrap();
    assert!(farneback(ctx, &rgb, &rgb, 3, 0.5, 15, 3, 5, 1.2, mode).is_err());
    // Identical flat frames: zero flow (up to rounding in the polynomial
    // fit), no NaN from the regularised solve or the thresholding.
    let flat = upload(ctx, &vec![100.0; 64 * 48], 64, 48);
    for flow in [
        farneback(ctx, &flat, &flat, 3, 0.5, 15, 3, 5, 1.2, mode).unwrap(),
        tvl1_optical_flow(ctx, &flat, &flat, Tvl1Config::default()).unwrap(),
    ] {
        let flow = flow.to_cpu_ctx(ctx).unwrap();
        assert!(flow.as_slice().unwrap().iter().all(|v| v.abs() < 1e-6));
    }
}
