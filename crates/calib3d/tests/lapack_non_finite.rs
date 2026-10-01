//! Non-finite input must not reach LAPACK.
//!
//! Two public entry points hung on a single NaN, found by auditing untested
//! code:
//!
//! - `calibrate_camera_planar` / `solve_dlt_homography`, via
//!   `hartley_normalize`: its only guard was `mean_dist <= 1e-12`, and
//!   `NaN <= 1e-12` is **false**, so an all-NaN normalisation matrix reached the
//!   SVD. Control call returns in ~1 ms; with the NaN it was killed at 300 s and
//!   again at 120 s.
//! - `calibrate_hand_eye`, via `symmetric_eigen` on a Gram matrix containing
//!   NaN: never returned, killed at 60 s.
//!
//! The common cause is that every LAPACK convergence test is a *comparison*, and
//! every comparison against NaN is false. A hang reports nothing, so it is worse
//! than a wrong answer.
//!
//! Two layers of defence, and **the outer one masks the inner**: `solve_dlt_homography`
//! checks its point sets before calling `hartley_normalize`, so reverting only
//! the `mean_dist` guard still passes these tests. That is deliberate - the
//! entry check is the cheap one and covers every caller - but it means the
//! `hartley_normalize` guard is not independently covered here. It matters
//! because `hartley_normalize` is also reachable from the DLT PnP path, which
//! has its own entry check.

use cv_calib3d::{calibrate_hand_eye, solve_dlt_homography};
use nalgebra::{Matrix3, Vector3};
use std::time::Duration;

/// Run `f` on a worker thread and require it to finish inside a deadline.
///
/// A hang must fail the test rather than wedge the suite - which is exactly
/// what an earlier scratch probe did, costing a 360 s timeout.
fn must_terminate_within<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    assert!(
        rx.recv_timeout(Duration::from_secs(30)).is_ok(),
        "{what} did not return within 30s"
    );
}

fn good_pairs(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in 0..n {
        let x = (i % 5) as f64;
        let y = (i / 5) as f64;
        src.push([x, y]);
        dst.push([x + 0.01 * x * y, y - 0.008 * x]);
    }
    (src, dst)
}

#[test]
fn a_non_finite_point_set_is_refused_rather_than_hanging_the_svd() {
    let (mut src, mut dst) = good_pairs(12);
    src[7] = [f64::NAN, f64::NAN];

    must_terminate_within("solve_dlt_homography with a NaN point", move || {
        assert!(
            solve_dlt_homography(&src, &dst).is_none(),
            "a homography through points at infinity does not exist"
        );
    });
}

/// The control case: without the NaN the same call returns quickly and succeeds,
/// so the guard is not simply refusing everything.
#[test]
fn a_finite_point_set_still_solves() {
    let (src, dst) = good_pairs(12);
    must_terminate_within("solve_dlt_homography with finite points", move || {
        assert!(solve_dlt_homography(&src, &dst).is_some());
    });
}

/// One non-finite coordinate in an otherwise perfect set - the realistic shape,
/// since a single bad projection or transcription is enough.
#[test]
fn a_single_non_finite_coordinate_is_enough() {
    let (mut src, dst) = good_pairs(12);
    src[3][1] = f64::INFINITY;
    must_terminate_within(
        "solve_dlt_homography with one infinite coordinate",
        move || {
            assert!(solve_dlt_homography(&src, &dst).is_none());
        },
    );
}

fn hand_eye_inputs(n: usize) -> (Vec<Matrix3<f64>>, Vec<Vector3<f64>>) {
    let mut a = Vec::new();
    let mut t = Vec::new();
    for i in 0..n {
        let ang = 0.1 * i as f64;
        let r = Matrix3::from_row_slice(&[
            ang.cos(),
            -ang.sin(),
            0.0,
            ang.sin(),
            ang.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ]);
        a.push(r);
        t.push(Vector3::new(0.1 * i as f64, 0.0, 0.0));
    }
    (a, t)
}

#[test]
fn calibrate_hand_eye_terminates_on_a_non_finite_rotation() {
    let (mut a, t) = hand_eye_inputs(4);
    a[2][(0, 0)] = f64::NAN;

    must_terminate_within("calibrate_hand_eye with a NaN rotation", move || {
        assert!(
            calibrate_hand_eye(&a, &t, &a, &t, cv_calib3d::HandEyeMethod::Tsai).is_none(),
            "a rotation containing NaN cannot be averaged"
        );
    });
}

#[test]
fn calibrate_hand_eye_terminates_on_a_non_finite_translation() {
    let (a, mut t) = hand_eye_inputs(4);
    t[1].x = f64::INFINITY;

    must_terminate_within(
        "calibrate_hand_eye with an infinite translation",
        move || {
            assert!(calibrate_hand_eye(&a, &t, &a, &t, cv_calib3d::HandEyeMethod::Tsai).is_none());
        },
    );
}

/// Fewer than three samples is refused by the existing length guard; this pins
/// it so the NaN guard cannot be mistaken for the whole validation.
#[test]
fn calibrate_hand_eye_refuses_too_few_samples() {
    let (a, t) = hand_eye_inputs(2);
    assert!(
        calibrate_hand_eye(&a, &t, &a, &t, cv_calib3d::HandEyeMethod::Tsai).is_none(),
        "two samples cannot determine a hand-eye transform"
    );
}
