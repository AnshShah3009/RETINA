//! Non-finite input must not reach an SVD.
//!
//! `solve_dlt_homography` and `solve_dlt_fundamental` build a design matrix and
//! hand it to LAPACK's SVD. With NaN in the points, that matrix contains NaN, and
//! bidiagonalisation's convergence test is a *comparison* - and every comparison
//! against NaN is false, so it never converges.
//!
//! Found by a scratch probe while auditing untested code: a single NaN
//! observation fed to `calibrate_camera_planar` did not return at all. The probe
//! hung, and the workspace suite timed out after 360 s waiting on it. A hang is
//! worse than a wrong answer, because nothing reports it.

use cv_calib3d::{solve_dlt_fundamental, solve_dlt_homography};

/// A square, well-conditioned correspondence set that does solve.
fn good_pairs(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in 0..n {
        let x = (i % 5) as f64;
        let y = (i / 5) as f64;
        src.push([x, y]);
        // A mild perspective-ish warp, invertible and well conditioned.
        dst.push([x + 0.01 * x * y, y - 0.008 * x]);
    }
    (src, dst)
}

/// The control case must still solve, or the tests below prove nothing.
#[test]
fn a_finite_correspondence_set_still_solves() {
    let (src, dst) = good_pairs(12);
    assert!(
        solve_dlt_homography(&src, &dst).is_some(),
        "a finite, non-degenerate set must produce a homography"
    );
}

#[test]
fn nan_points_are_refused_rather_than_hanging_the_svd() {
    let (mut src, mut dst) = good_pairs(12);
    src[7] = [f64::NAN, f64::NAN];
    dst[7] = [f64::NAN, f64::NAN];

    let r = solve_dlt_homography(&src, &dst);
    assert!(
        r.is_none(),
        "a homography through points at infinity does not exist, so this must \
         return None rather than entering the SVD"
    );
}

#[test]
fn a_single_nan_is_enough_to_hang_it() {
    // One bad coordinate, not a whole point: a realistic transcription or
    // projection slip.
    let (mut src, dst) = good_pairs(12);
    src[3][1] = f64::INFINITY;
    assert!(solve_dlt_homography(&src, &dst).is_none());
}

#[test]
fn a_non_finite_destination_is_refused_too() {
    let (src, mut dst) = good_pairs(12);
    dst[9][0] = f64::NAN;
    assert!(solve_dlt_homography(&src, &dst).is_none());
}

#[test]
fn the_fundamental_solver_has_the_same_exposure() {
    let (mut a, mut b) = good_pairs(20);
    assert!(
        solve_dlt_fundamental(&a, &b).is_some(),
        "a finite set must produce a fundamental matrix"
    );

    a[11] = [f64::NAN, f64::NAN];
    assert!(
        solve_dlt_fundamental(&a, &b).is_none(),
        "the fundamental solver enters the same SVD and needs the same guard"
    );

    let (a, mut b2) = good_pairs(20);
    b2[2] = [f64::NEG_INFINITY, 0.0];
    assert!(solve_dlt_fundamental(&a, &b2).is_none());
}
