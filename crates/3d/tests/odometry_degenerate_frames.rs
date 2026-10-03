//! `compute_rgbd_odometry` must not panic, and must not report a success it
//! did not achieve.
//!
//! # 1. A frame narrower than 8 pixels panicked
//!
//! `compute_vertex_normal_map_ctx` (`odometry/mod.rs`) built its normal map with
//! `normals.par_chunks_mut(width)`. Rayon panics outright on a zero chunk size:
//! `chunk_size must not be zero`.
//!
//! That zero width is reachable from the public entry point without any invalid
//! input. `compute_rgbd_odometry_ctx` runs a four-level pyramid at scales
//! `[0.125, 0.25, 0.5, 1.0]`, and `scaled_width = (width as f32 * scale) as usize`.
//! For a 7-pixel-wide frame the coarsest level is `(7.0 * 0.125) as usize = 0`, so
//! **every call panicked in a rayon worker**. So did a 1x1 frame and a declared
//! 0x0 frame.
//!
//! Measured before the fix (panic text captured from a `catch_unwind`):
//! ```text
//! compute_rgbd_odometry(&[], &[], .., 0, 0, PointToPlane)  -> PANIC
//!     thread 'default-0' panicked at crates/3d/src/odometry/mod.rs:535:
//     chunk_size must not be zero
//! compute_rgbd_odometry(&[1.0], &[1.0], .., 1, 1, PointToPlane) -> PANIC (same)
//! ```
//!
//! The existing `length_checks::odometry_does_not_panic_on_a_short_depth_slice`
//! test covers a *short slice at a normal frame size* — a different guard. Neither
//! test reached this one, which is why it survived.
//!
//! # 2. A pyramid that matched nothing returned `Some` with the identity
//!
//! The pyramid loop is best-effort: each scale returns `None` on failure and the
//! previous transform is carried forward. If **every** scale fails, the transform
//! that reaches the end is the identity — not because the camera did not move,
//! but because nothing was ever estimated.
//!
//! `compute_rgbd_odometry_ctx` returned that identity unconditionally:
//!
//! ```ignore
//! Some(OdometryResult { transformation, fitness, inlier_rmse: rmse })
//! ```
//!
//! and `evaluate_odometry_ctx` returns `(0.0, 0.0)` when there were zero
//! correspondences. Measured, a source frame with valid depth against a target
//! frame with none:
//!
//! ```text
//! Some(OdometryResult { transformation: identity, fitness: 0.0, inlier_rmse: 0.0 })
//! ```
//!
//! Three things are wrong at once. The camera is reported as having not moved
//! when no motion was ever estimated. The RMSE of an empty match set is reported
//! as a perfect `0.0` — the same value a perfect alignment produces. And
//! `fitness: 0.0` is the *same* value `compute_hybrid` used to fabricate, which
//! is the exact ambiguity that fix was made to remove: a caller reading the
//! score alone cannot separate "hybrid is unimplemented" from "nothing matched".
//!
//! The function returns `Option<OdometryResult>`; `None` is the honest signal,
//! and it is what `compute_intensity` and `compute_hybrid` already return.

use cv_3d::tsdf::CameraIntrinsics;
use cv_3d::{compute_rgbd_odometry, OdometryMethod};

fn intrinsics_for(w: u32, h: u32) -> CameraIntrinsics {
    CameraIntrinsics::new(100.0, 100.0, w as f32 / 2.0, h as f32 / 2.0, w, h)
}

/// Every one of these panicked before the fix.
#[test]
fn a_frame_narrower_than_the_coarsest_pyramid_level_does_not_panic() {
    // The pyramid's coarsest scale is 1/8, so any width below 8 downsamples to
    // zero columns.
    for &n in &[1usize, 2, 3, 4, 7] {
        let depth = vec![1.0f32; n * n];
        let k = intrinsics_for(n as u32, n as u32);
        let r = std::panic::catch_unwind(move || {
            compute_rgbd_odometry(
                &depth,
                &depth,
                None,
                None,
                &k,
                n,
                n,
                OdometryMethod::PointToPlane,
            )
        });
        assert!(
            r.is_ok(),
            "{n}x{n}: compute_rgbd_odometry panicked. The pyramid's coarsest level \
             is scale 0.125, and `(n as f32 * 0.125) as usize == 0` for every n < 8, \
             so `par_chunks_mut(0)` aborted the worker."
        );
    }
}

/// A declared zero-sized frame is the degenerate end of the same path.
#[test]
fn a_zero_sized_frame_does_not_panic() {
    let empty: Vec<f32> = Vec::new();
    let k = intrinsics_for(0, 0);
    let r = std::panic::catch_unwind(move || {
        compute_rgbd_odometry(
            &empty,
            &empty,
            None,
            None,
            &k,
            0,
            0,
            OdometryMethod::PointToPlane,
        )
    });
    assert!(
        r.is_ok(),
        "a 0x0 frame panicked: there are no vertices to differentiate"
    );
    // And nothing was measured, so there is nothing to report.
    assert!(
        r.unwrap().is_none(),
        "a frame with no pixels cannot produce a motion estimate"
    );
}

/// A source frame with no valid surface at all must not produce a result.
#[test]
fn no_source_surface_is_reported_as_no_result() {
    let n = 64usize;
    let k = intrinsics_for(n as u32, n as u32);
    let depth = vec![0.0f32; n * n]; // every pixel invalid

    assert!(
        compute_rgbd_odometry(
            &depth,
            &depth,
            None,
            None,
            &k,
            n,
            n,
            OdometryMethod::PointToPlane
        )
        .is_none(),
        "with no valid depth anywhere, nothing was matched and nothing was estimated; \
         returning the identity as a 'result' would claim the camera did not move"
    );
}

/// The sharpest case: a perfectly good source frame against a target with no
/// surface. This is the one the measurements above came from.
#[test]
fn a_source_with_no_target_surface_is_reported_as_no_result() {
    let n = 64usize;
    let k = intrinsics_for(n as u32, n as u32);
    let source = vec![1.0f32; n * n];
    let target = vec![0.0f32; n * n]; // no surface anywhere in the target

    let r = compute_rgbd_odometry(
        &source,
        &target,
        None,
        None,
        &k,
        n,
        n,
        OdometryMethod::PointToPlane,
    );
    assert!(
        r.is_none(),
        "a valid source frame against a target with no surface matches nothing, so \
         the estimate is undefined. It previously came back as \
         `Some(OdometryResult {{ transformation: identity, fitness: 0.0, \
         inlier_rmse: 0.0 }})` — claiming both that the camera did not move and \
         that the (empty) match was perfect."
    );
}

/// CONTROL: two frames that genuinely do correspond still produce a result with
/// a real fitness, so the `None` above cannot be a blanket refusal.
#[test]
fn control_corresponding_frames_still_produce_a_result() {
    let n = 64usize;
    let k = intrinsics_for(n as u32, n as u32);
    let source = vec![1.0f32; n * n];
    let target = vec![1.2f32; n * n]; // same surface, slightly further away

    let r = compute_rgbd_odometry(
        &source,
        &target,
        None,
        None,
        &k,
        n,
        n,
        OdometryMethod::PointToPlane,
    )
    .expect("CONTROL: two frames of the same scene must still produce a result");

    assert!(
        r.fitness > 0.0,
        "CONTROL: a real correspondence set must score above zero, got {}",
        r.fitness
    );
    assert!(
        r.inlier_rmse > 0.0,
        "CONTROL: the two frames are 0.2 apart, so a zero RMSE would be the \
         empty-match artefact, not a measurement. Got {}",
        r.inlier_rmse
    );
    assert!(
        r.inlier_rmse.is_finite(),
        "CONTROL: the RMSE must be a real number"
    );
}

/// CONTROL: a frame at the smallest size that *does* survive the pyramid coarsest
/// level still returns something sane.
#[test]
fn control_the_smallest_non_degenerate_frame_still_answers() {
    let n = 8usize;
    let k = intrinsics_for(n as u32, n as u32);
    let depth = vec![1.0f32; n * n];
    let r = std::panic::catch_unwind(move || {
        compute_rgbd_odometry(
            &depth,
            &depth,
            None,
            None,
            &k,
            n,
            n,
            OdometryMethod::PointToPlane,
        )
    });
    assert!(r.is_ok(), "an 8x8 frame must not panic");
}