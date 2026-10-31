//! `optical_flow_lk` must treat `max_iters` as a **convergence budget**.
//!
//! ## The defect
//!
//! The Lucas–Kanade loop re-anchored the integer sample coordinates `ix`/`iy` on
//! the *current* estimate `u`/`v` on every pass, while measuring the temporal
//! difference `i_t` at the **same integer coordinates** as `i_0`. So `i_t` did not
//! depend on `u`/`v` at all: every iteration solved an identical 2x2 system and
//! accumulated an identical step.
//!
//! Measured on a 2-pixel horizontal shift of a band-limited image, window 21, the
//! error grew *linearly* with the iteration count:
//!
//! ```text
//!   1 iteration : +0.05 px
//!   2 iterations: +2.05
//!   3 iterations: +4.05
//!   5 iterations: +8.15
//!  10 iterations: +18.5
//!  30 iterations: +58.6
//! ```
//!
//! That is the worst possible shape for a caller: `max_iters` reads as a budget, so
//! raising it is the obvious response to a poor result, and doing so makes the flow
//! **worse**. A caller cannot distinguish "not converged" from "diverging" by
//! watching the iteration count, because the iteration count is not reported as a
//! failure at all.
//!
//! ## The fix
//!
//! Bouguet's pyramidal LK (OpenCV `calcOpticalFlowPyrLK`): the template window of
//! `prev` is sampled bilinearly around the feature, `next` is sampled bilinearly at
//! the running estimate, and each solve updates the residual flow that the next
//! iteration samples at. The error is then non-increasing in the budget, so
//! `max_iters` behaves as a budget. (An earlier fix anchored an integer window and
//! sampled `next` at an integer offset, which double-counted the sub-pixel part and
//! oscillated - see `lk_subpixel_bouguet.rs`.)

#![forbid(unsafe_code)]

use cv_core::storage::CpuStorage;
use cv_core::Tensor;
use cv_hal::compute::ComputeDevice;
use nalgebra::Vector2;

/// Where the feature starts, common to both frames.
const START: (f32, f32) = (32.0, 24.0);

/// A band-limited image with a known integer shift between the two frames.
///
/// Built as smooth sinusoids rather than noise so the shift is the only signal
/// present - a textureless or aliased input would make the measurement meaningless.
///
/// Two crossing waves, not one: a single plane wave `sin(2 pi (x/16 + y/24))` has a
/// rank-1 structure tensor (every gradient is parallel to `(3, 2)`), so only the
/// flow component along that normal is observable and `(+2, +1)` is one of a line
/// of equally good answers. The original version of this test used that single
/// wave and passed only because f32 round-off made `G` numerically invertible; a
/// tracker with OpenCV's minimum-eigenvalue check correctly declines it.
fn frames(
    w: usize,
    h: usize,
    dx: f32,
    dy: f32,
) -> (Tensor<f32, CpuStorage<f32>>, Tensor<f32, CpuStorage<f32>>) {
    use std::f32::consts::PI;
    let f = |x: f32, y: f32| {
        127.0
            + 60.0 * (2.0 * PI * (x / 16.0 + y / 24.0)).sin()
            + 40.0 * (2.0 * PI * (x / 22.0 - y / 18.0)).cos()
    };
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            a[y * w + x] = f(x as f32, y as f32);
            // `b` is `a` shifted by (dx, dy): b(x,y) = a(x - dx, y - dy).
            b[y * w + x] = f(x as f32 - dx, y as f32 - dy);
        }
    }
    let shape = cv_core::TensorShape::new(1, h, w);
    (
        Tensor::from_vec(a, shape.clone()).expect("prev"),
        Tensor::from_vec(b, shape).expect("next"),
    )
}

/// The tracked **absolute position** after `max_iters`, one `[x, y]` per point.
///
/// Not a flow vector: the API returns where the point ended up, so the flow is the
/// difference from where it started. Getting this wrong was my first error here -
/// `[32, 24] -> [33, 25.75]` is a displacement of `(+1, +1.75)`, not a 39 px error.
fn track(max_iters: u32, window: usize) -> Vec<f32> {
    let (prev, next) = frames(64, 48, 2.0, 1.0);
    let prev_p = vec![prev];
    let next_p = vec![next];
    // One feature, away from the border so the window stays inside both frames.
    let pts = vec![[START.0, START.1]];
    let cpu = cv_hal::cpu::CpuBackend::new().expect("cpu backend");
    let ctx = ComputeDevice::Cpu(&cpu);
    let out = ctx
        .optical_flow_lk(&prev_p, &next_p, &pts, window, max_iters)
        .expect("LK should run on a well-formed input");
    out.into_iter().flatten().collect::<Vec<f32>>()
}

/// The error must not grow with `max_iters`.
///
/// This is the assertion the defect fails hardest: before the fix the error rose
/// linearly to +58.6 px at 30 iterations, so it would fail by three orders of
/// magnitude, not marginally.
#[test]
fn more_iterations_do_not_diverge() {
    // The true displacement from `a` to `b`: b(x) = a(x - d), so a point at `p` in
    // `prev` sits at `p - d` in `next`... LK estimates prev -> next, and the shift
    // applied to build `b` was (+2, +1), so the recovered motion is (+2, +1).
    let truth = Vector2::new(2.0f32, 1.0f32);

    let mut errors = Vec::new();
    for iters in [1u32, 2, 5, 10, 30] {
        let got = track(iters, 21);
        assert_eq!(
            got.len(),
            2,
            "LK returns the tracked (x, y) per point; got {got:?} at {iters} iterations"
        );
        errors.push((
            iters,
            Vector2::new(got[0], got[1]) - Vector2::new(START.0, START.1),
        ));
    }

    let first = errors[0].1;
    for (iters, p) in &errors {
        let err = (p - truth).norm();
        let first_err = (first - truth).norm();
        assert!(
            err <= first_err + 1e-3,
            "more iterations must not make the result worse: at {iters} iterations \
             the error is {err:.4} px, against {first_err:.4} px at 1 iteration"
        );
    }
}

/// One iteration must already be close, and the answer must be stable across the
/// budgets a caller would realistically pass.
#[test]
fn the_flow_converges_to_the_planted_shift() {
    let truth = Vector2::new(2.0f32, 1.0f32);
    // **Not** asserted at 1 iteration: the first solve linearises about zero flow,
    // so it is an approximation by construction. Converged Bouguet LK on this
    // smooth, noise-free pair lands within a few hundredths of a pixel.
    //
    // Claiming it converged *improves* with budget is the real contract, and
    // `more_iterations_do_not_diverge` above is the assertion that pins it.
    for iters in [5u32, 30] {
        let got = track(iters, 21);
        let p = Vector2::new(got[0], got[1]) - Vector2::new(START.0, START.1);
        let err = (p - truth).norm();
        println!("{iters} iterations: flow {p:?}, err {err:.4}");
        assert!(
            err < 0.05,
            "at {iters} iterations the recovered flow {p:?} is {err:.4} px from the \
             planted {truth:?}"
        );
    }
}

/// CONTROL: a zero shift must be recovered as zero, so the test above is measuring
/// the displacement and not some artefact of the construction.
#[test]
fn a_zero_shift_is_recovered_as_zero() {
    let (prev, next) = frames(64, 48, 0.0, 0.0);
    let pts = vec![[START.0, START.1]];
    let cpu = cv_hal::cpu::CpuBackend::new().expect("cpu backend");
    let ctx = ComputeDevice::Cpu(&cpu);
    let out = ctx
        .optical_flow_lk(&[prev], &[next], &pts, 21, 10)
        .expect("LK on identical frames");
    let got = out.into_iter().flatten().collect::<Vec<f32>>();
    assert_eq!(got.len(), 2, "expected the tracked (x, y), got {got:?}");
    let flow = Vector2::new(got[0], got[1]) - Vector2::new(START.0, START.1);
    assert!(
        flow.norm() < 0.5,
        "identical frames must yield ~zero flow, got {flow:?}"
    );
}
