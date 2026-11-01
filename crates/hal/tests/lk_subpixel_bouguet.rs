//! `optical_flow_lk` must be pyramidal Lucas-Kanade in Bouguet's formulation,
//! the algorithm behind OpenCV's `calcOpticalFlowPyrLK`:
//!
//! * the template window of the *previous* frame is sampled bilinearly around the
//!   feature's own position at each level (`p / 2^L`), never around the running
//!   estimate;
//! * the *next* frame is sampled bilinearly at `p/2^L + g + v`, where `g` is the
//!   guess carried from the coarser level and `v` the in-level residual flow;
//! * each solve updates `v`, and the guess is propagated as `g = 2 (g + v)`.
//!
//! Regression for GAPS g-h7: the temporal term was sampled at an *integer* offset
//! `round(u) - anchor` from an anchor window while the solve's result was added to
//! the fractional estimate, double-counting the sub-pixel part. On a smooth image
//! shifted by 2.3 px (window 21) the recovered `dx` oscillated with the iteration
//! budget: 2.155, 2.459, 2.763, 2.362, 1.864, 1.877 at 1, 2, 3, 5, 10, 20
//! iterations.

use cv_core::storage::CpuStorage;
use cv_core::{Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

type T32 = Tensor<f32, CpuStorage<f32>>;

/// A smooth, band-limited texture (sum of oblique sinusoids, wavelengths
/// 29-53 px) evaluated analytically, so a sub-pixel shift is exact rather than
/// an interpolation of the previous frame.
fn texture(x: f32, y: f32) -> f32 {
    use std::f32::consts::TAU;
    128.0
        + 45.0 * (TAU * (x / 37.0 + y / 53.0)).sin()
        + 40.0 * (TAU * (x / 43.0 - y / 29.0)).cos()
        + 30.0 * (TAU * (x / 31.0 + y / 47.0) + 1.0).sin()
}

/// `prev(x, y) = f(x, y)`, `next(x, y) = f(x - dx, y - dy)`: content moves by
/// `(+dx, +dy)` from `prev` to `next`.
fn frames(w: usize, h: usize, dx: f32, dy: f32) -> (T32, T32) {
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            a[y * w + x] = texture(x as f32, y as f32);
            b[y * w + x] = texture(x as f32 - dx, y as f32 - dy);
        }
    }
    let shape = TensorShape::new(1, h, w);
    (
        Tensor::from_vec(a, shape).unwrap(),
        Tensor::from_vec(b, shape).unwrap(),
    )
}

fn pyramid(cpu: &CpuBackend, base: T32, levels: usize) -> Vec<T32> {
    let mut p = vec![base];
    for _ in 1..levels {
        let next = cpu.pyramid_down(p.last().unwrap()).unwrap();
        p.push(next);
    }
    p
}

fn flow_of(
    cpu: &CpuBackend,
    prev: &[T32],
    next: &[T32],
    pt: [f32; 2],
    win: usize,
    iters: u32,
) -> [f32; 2] {
    let out = cpu.optical_flow_lk(prev, next, &[pt], win, iters).unwrap();
    [out[0][0] - pt[0], out[0][1] - pt[1]]
}

/// Identical frames must not move a sub-pixel point. Before the fix the anchor
/// was `round(p)`, so the template sat up to half a pixel away from the feature.
#[test]
fn identical_frames_keep_subpixel_points() {
    let cpu = CpuBackend::new().unwrap();
    let (a, _) = frames(128, 128, 0.0, 0.0);
    for levels in [1usize, 3] {
        let p = pyramid(&cpu, a.clone(), levels);
        for pt in [[60.5f32, 60.5], [60.75, 60.25], [63.3, 58.9]] {
            let f = flow_of(&cpu, &p, &p, pt, 21, 20);
            assert!(
                f[0].abs() < 1e-3 && f[1].abs() < 1e-3,
                "{levels} level(s): identical frames moved {pt:?} by {f:?}"
            );
        }
    }
}

/// A smooth texture shifted by (2.3, -1.7) px must be recovered within 0.02 px,
/// and the error must be non-increasing in the iteration budget.
#[test]
fn subpixel_shift_converges_monotonically() {
    let cpu = CpuBackend::new().unwrap();
    let truth = [2.3f32, -1.7];
    let (a, b) = frames(128, 128, truth[0], truth[1]);
    let pt = [64.0f32, 64.0];
    let mut errs = Vec::new();
    for iters in [1u32, 2, 3, 5, 10, 20] {
        let f = flow_of(
            &cpu,
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            pt,
            21,
            iters,
        );
        let e = ((f[0] - truth[0]).powi(2) + (f[1] - truth[1]).powi(2)).sqrt();
        println!(
            "iters {iters:>2}: flow ({:.4}, {:.4}) err {e:.4}",
            f[0], f[1]
        );
        errs.push((iters, f, e));
    }
    for w in errs.windows(2) {
        assert!(
            w[1].2 <= w[0].2 + 1e-3,
            "error grew with budget: {} iters -> {:.4}, {} iters -> {:.4}",
            w[0].0,
            w[0].2,
            w[1].0,
            w[1].2
        );
    }
    for (iters, f, e) in &errs {
        if *iters >= 5 {
            assert!(
                *e < 0.02,
                "{iters} iters: recovered {f:?}, {e:.4} px from {truth:?}"
            );
        }
    }
    // Stable: once converged the budget must not change the answer.
    let (_, f10, _) = errs[4];
    let (_, f20, _) = errs[5];
    assert!((f10[0] - f20[0]).abs() < 1e-3 && (f10[1] - f20[1]).abs() < 1e-3);
}

/// A ~10 px shift is outside a single level's basin with window 11, so this
/// exercises guess propagation `g = 2 (g + v)` across a 3-level pyramid and the
/// feature-centred template at the finer levels.
#[test]
fn pyramid_recovers_large_subpixel_shift() {
    let cpu = CpuBackend::new().unwrap();
    let truth = [9.6f32, -4.3];
    let (a, b) = frames(192, 160, truth[0], truth[1]);
    let pa = pyramid(&cpu, a, 3);
    let pb = pyramid(&cpu, b, 3);
    for pt in [[96.0f32, 80.0], [90.4, 77.7]] {
        let f = flow_of(&cpu, &pa, &pb, pt, 11, 30);
        let e = ((f[0] - truth[0]).powi(2) + (f[1] - truth[1]).powi(2)).sqrt();
        println!("pyramid {pt:?}: flow ({:.4}, {:.4}) err {e:.4}", f[0], f[1]);
        assert!(
            e < 0.02,
            "pyramid track of {pt:?} recovered {f:?}, truth {truth:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// GPU kernel (`shaders/lucas_kanade.wgsl`): same contract.
// ---------------------------------------------------------------------------

fn require_gpu() -> Option<&'static cv_hal::gpu::GpuContext> {
    use cv_hal::gpu::GpuContext;
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    match pollster::block_on(GpuContext::init_global()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            println!("Skipping GPU case (no adapter): {e}");
            None
        }
    }
}

fn gpu_flows(
    gpu: &cv_hal::gpu::GpuContext,
    prev: &[T32],
    next: &[T32],
    pts: &[[f32; 2]],
    win: usize,
    iters: u32,
) -> Vec<[f32; 2]> {
    use cv_hal::tensor_ext::TensorToGpu;
    let up = |p: &[T32]| -> Vec<_> { p.iter().map(|t| t.to_gpu_ctx(gpu).unwrap()).collect() };
    let out = gpu
        .optical_flow_lk(&up(prev), &up(next), pts, win, iters)
        .unwrap();
    out.iter()
        .zip(pts)
        .map(|(o, p)| [o[0] - p[0], o[1] - p[1]])
        .collect()
}

#[test]
fn gpu_matches_bouguet_contract() {
    let Some(gpu) = require_gpu() else { return };
    let cpu = CpuBackend::new().unwrap();

    // Identical frames, sub-pixel points, 3 levels.
    let (a, _) = frames(128, 128, 0.0, 0.0);
    let pa = pyramid(&cpu, a, 3);
    let pts = [[60.5f32, 60.5], [60.75, 60.25], [63.3, 58.9]];
    for (pt, f) in pts.iter().zip(gpu_flows(gpu, &pa, &pa, &pts, 21, 20)) {
        assert!(
            f[0].abs() < 1e-3 && f[1].abs() < 1e-3,
            "GPU: identical frames moved {pt:?} by {f:?}"
        );
    }

    // Single level, (2.3, -1.7).
    let truth = [2.3f32, -1.7];
    let (a, b) = frames(128, 128, truth[0], truth[1]);
    let f = gpu_flows(gpu, &[a], &[b], &[[64.0, 64.0]], 21, 20)[0];
    let e = ((f[0] - truth[0]).powi(2) + (f[1] - truth[1]).powi(2)).sqrt();
    println!(
        "GPU single level: flow ({:.4}, {:.4}) err {e:.4}",
        f[0], f[1]
    );
    assert!(
        e < 0.02,
        "GPU single level recovered {f:?}, truth {truth:?}"
    );

    // Three levels, (9.6, -4.3).
    let truth = [9.6f32, -4.3];
    let (a, b) = frames(192, 160, truth[0], truth[1]);
    let pa = pyramid(&cpu, a, 3);
    let pb = pyramid(&cpu, b, 3);
    let pts = [[96.0f32, 80.0], [90.4, 77.7]];
    for (pt, f) in pts.iter().zip(gpu_flows(gpu, &pa, &pb, &pts, 11, 30)) {
        let e = ((f[0] - truth[0]).powi(2) + (f[1] - truth[1]).powi(2)).sqrt();
        println!(
            "GPU pyramid {pt:?}: flow ({:.4}, {:.4}) err {e:.4}",
            f[0], f[1]
        );
        assert!(
            e < 0.02,
            "GPU pyramid track of {pt:?} recovered {f:?}, truth {truth:?}"
        );
    }
}
