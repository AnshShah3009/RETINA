//! Degenerate correspondences must make `solve_dlt_homography` return `None`,
//! not a plausible-looking matrix.
//!
//! The Hartley transform is happy with a collinear set: a line has a perfectly
//! well-defined centroid and mean distance, so normalisation succeeds and the
//! design matrix is built. But a homography is 8 degrees of freedom, and the
//! DLT solves a homogeneous system `A h = 0` in **9** unknowns, so the answer
//! is only meaningful when `A` has rank 8 - null space exactly
//! one-dimensional. With 3 collinear points the matrix drops to rank 7 and the
//! homography becomes a 1-parameter family (the vanishing point of the line is
//! never observed), so *every* member fits the correspondences equally well.
//!
//! The SVD still returns a unit-norm vector, so the caller got `Ok` with a
//! plausible-looking matrix. Measured on the unfixed code: `det(H)` of
//! `-2.2e0` / `0.0`, `(2,2)` entry ~1e-14 - which is exactly why the
//! `h[(2,2)].abs() > 1e-12` normalisation guard falls through to the `else`
//! branch and hands back the unscaled rank-1 matrix that projects every input
//! point to infinity. `Some` is not merely useless here; it is a lie.
//!
//! The RANSAC path in `cv-features` hid this because `HomographyEstimator`
//! `compute_error` returns `INFINITY` when `w ~ 0`, scoring these as
//! all-outliers. A *direct* caller of `HomographySolver::estimate` or
//! `estimate_homography_dlt` had no such protection.

use cv_calib3d::solve_dlt_homography;
use nalgebra::{DMatrix, Matrix3, Vector3};

/// The solver's criterion: the design matrix's 8th singular value relative to
/// its largest. Below this the null space is more than one-dimensional and no
/// homography is determined. Mirrored here so the test measures the *actual*
/// quantity rather than a proxy.
const TOLERANCE: f64 = 1e-9;

/// `sigma_8 / sigma_1` of the design matrix `solve_dlt_homography` builds,
/// zero-padded to 9x9 exactly as the solver pads it.
fn design_ratio(src: &[[f64; 2]], dst: &[[f64; 2]]) -> f64 {
    let n = src.len();
    let (_, n1) = cv_calib3d::hartley_normalize(src).expect("src normalisation");
    let (_, n2) = cv_calib3d::hartley_normalize(dst).expect("dst normalisation");
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let (x, y) = (n1[i][0], n1[i][1]);
        let (u, v) = (n2[i][0], n2[i][1]);
        let (r0, r1) = (2 * i, 2 * i + 1);
        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;
        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }
    if a.nrows() < 9 {
        let mut padded = DMatrix::<f64>::zeros(9, 9);
        padded.view_mut((0, 0), (a.nrows(), 9)).copy_from(&a);
        a = padded;
    }
    let sv = a.svd(false, false).singular_values;
    let sigma_max = sv.iter().copied().fold(0.0f64, f64::max);
    // descending order, so `len - 2` is sigma_8 and `len - 1` is the null
    // direction itself - always ~0, carries no information.
    let sigma_8 = if sv.len() >= 2 { sv[sv.len() - 2] } else { 0.0 };
    if sigma_max > 0.0 {
        sigma_8 / sigma_max
    } else {
        0.0
    }
}

/// A known, invertible homography with a real perspective term, used to
/// generate the destination points.
fn known_h() -> Matrix3<f64> {
    Matrix3::new(
        1.2, 0.1, 300.0, //
        -0.05, 0.9, 220.0, //
        0.0002, 0.0001, 1.0,
    )
}

fn apply(m: &Matrix3<f64>, p: [f64; 2]) -> [f64; 2] {
    let v = m * Vector3::new(p[0], p[1], 1.0);
    [v[0] / v[2], v[1] / v[2]]
}

fn destinations(src: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let h = known_h();
    src.iter().map(|p| apply(&h, *p)).collect()
}

/// `H . p`, or `None` when the result lands at infinity.
fn project(h: &Matrix3<f64>, p: [f64; 2]) -> Option<[f64; 2]> {
    let v = h * Vector3::new(p[0], p[1], 1.0);
    if !v[2].is_finite() || v[2].abs() < 1e-300 {
        return None;
    }
    let q = [v[0] / v[2], v[1] / v[2]];
    if q[0].is_finite() && q[1].is_finite() {
        Some(q)
    } else {
        None
    }
}

/// Largest pixel distance between what `h` transfers `src` to and ground truth.
fn transfer_error(src: &[[f64; 2]], h: &Matrix3<f64>) -> f64 {
    let truth = known_h();
    let mut worst = 0.0f64;
    for p in src {
        let Some(q) = project(h, *p) else {
            return f64::INFINITY;
        };
        let t = apply(&truth, *p);
        worst = worst.max(((q[0] - t[0]).powi(2) + (q[1] - t[1]).powi(2)).sqrt());
    }
    worst
}

// ---------------------------------------------------------------------------
// CONTROL
// ---------------------------------------------------------------------------

/// The control is a genuinely non-degenerate minimal sample, not merely a
/// well-spread one.
///
/// "Well spread" is exactly what a weak threshold coasts on. An axis-aligned
/// image-corner quad is beautifully spread yet is an *affine* configuration:
/// every row of its design matrix is proportional to a partner row, so it sits
/// in the same near-degenerate regime as a collinear set. The control below is
/// therefore not a rectangle - its 4th corner is pushed off the grid so that no
/// two points share an x or a y, every one of the 6 triangles has a visibly
/// different area, and the quadrilateral's diagonals are far from parallel -
/// and the destinations carry a genuine perspective term. Measured
/// `sigma_8/sigma_1` of `2.6e-1`: six orders of magnitude above the tolerance,
/// with no accident of geometry doing the work.
#[test]
fn control_a_genuinely_non_degenerate_four_point_set_still_solves_exactly() {
    let src: [[f64; 2]; 4] = [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [37.0, 611.0]];
    let dst = destinations(&src);

    let ratio = design_ratio(&src, &dst);
    assert!(
        ratio > 1e3 * TOLERANCE,
        "the control must sit far inside the threshold; ratio = {ratio:e}"
    );

    let est = solve_dlt_homography(&src, &dst).expect("control must solve");
    let est = est / est[(2, 2)];
    let reference = known_h() / known_h()[(2, 2)];
    assert!(
        (est - reference).norm() < 1e-9,
        "control did not recover H:\n{est}\nexpected\n{reference}"
    );
    assert!(transfer_error(&src, &est) < 1e-6);
}

/// The skinniest configuration that is still legitimately solvable: a trapezium
/// whose short side is 3 px. This is the case a *too strict* criterion would
/// reject, so it guards the other end of the threshold.
#[test]
fn control_an_extreme_but_valid_perspective_set_still_solves() {
    let src: [[f64; 2]; 4] = [[0.0, 0.0], [1000.0, 0.0], [0.0, 3.0], [1000.0, 7.0]];
    let dst = destinations(&src);

    let ratio = design_ratio(&src, &dst);
    assert!(
        ratio > 1e3 * TOLERANCE,
        "this skinny-but-valid set is inside the threshold: {ratio:e}"
    );
    let est = solve_dlt_homography(&src, &dst).expect("skinny valid set must solve");
    let est = est / est[(2, 2)];
    assert!(
        (est - known_h() / known_h()[(2, 2)]).norm() < 1e-8,
        "skinny set did not recover H:\n{est}"
    );
}

/// The criterion must not depend on the units of the input. Same geometry,
/// scaled by `1e-6 .. 1e6`: an absolute cutoff would reject the small end.
///
/// The ratio is not *exactly* scale-invariant - Hartley normalisation is exact
/// only in exact arithmetic, so a 12-order-of-magnitude change in scale moves
/// `sigma_8/sigma_1` by ~38% (measured: `9.84e-4` at `1e-6` vs `1.36e-3` at
/// `1e6`). What matters is that it does not move by the 6 orders of magnitude
/// an absolute threshold would need in order to fail one end, so the assertion
/// is a factor of 10, not exact equality.
#[test]
fn control_the_criterion_is_scale_invariant() {
    let base: [[f64; 2]; 4] = [[0.0, 0.0], [220.0, 1.0], [440.0, 0.0], [300.0, 500.0]];
    let mut ratios = Vec::new();
    for s in [1e-6f64, 1e-3, 1.0, 1e3, 1e6] {
        let src = base.map(|p| [p[0] * s, p[1] * s]);
        let dst = destinations(&src);
        let ratio = design_ratio(&src, &dst);
        assert!(
            ratio > TOLERANCE,
            "scale {s:e} fell below the threshold; ratio = {ratio:e}"
        );
        ratios.push(ratio);
        assert!(
            solve_dlt_homography(&src, &dst).is_some(),
            "scale {s:e} must still solve; ratio = {ratio:e}"
        );
    }
    let hi = ratios.iter().copied().fold(f64::MIN, f64::max);
    let lo = ratios.iter().copied().fold(f64::MAX, f64::min);
    assert!(
        hi / lo < 10.0,
        "sigma_8/sigma_1 moved by a factor of {} across 12 orders of magnitude \
         ({lo:e} .. {hi:e}); the test is not scale-invariant",
        hi / lo
    );
}

// ---------------------------------------------------------------------------
// EXACTLY DEGENERATE
// ---------------------------------------------------------------------------

/// 3 of 4 source points on the line `y = x/2`, the 4th well off it.
///
/// The homography through such a set is a 1-parameter family, so any single
/// matrix returned is arbitrary.
#[test]
fn three_collinear_plus_one_off_line_is_refused() {
    let src: [[f64; 2]; 4] = [[0.0, 0.0], [200.0, 100.0], [400.0, 200.0], [320.0, 480.0]];
    let dst = destinations(&src);

    assert!(
        design_ratio(&src, &dst) < TOLERANCE,
        "this configuration is rank deficient by construction"
    );

    match solve_dlt_homography(&src, &dst) {
        None => {}
        Some(h) => panic!(
            "3 collinear must return None, got H = {h} with det = {:e} and \
             (2,2) = {:e}; transfer error = {:e}",
            h.determinant(),
            h[(2, 2)],
            transfer_error(&src, &h)
        ),
    }
}

#[test]
fn all_four_collinear_is_refused() {
    let src: [[f64; 2]; 4] = [[0.0, 0.0], [160.0, 80.0], [320.0, 160.0], [480.0, 240.0]];
    let dst = destinations(&src);

    assert!(design_ratio(&src, &dst) < TOLERANCE);
    assert!(solve_dlt_homography(&src, &dst).is_none());
}

/// The longer-correspondence version a real feature tracker would hand over:
/// 5 points, 4 of them on one line.
#[test]
fn four_of_five_collinear_is_refused() {
    let mut src: Vec<[f64; 2]> = (0..5)
        .map(|i| [60.0 + i as f64 * 130.0, 30.0 + i as f64 * 65.0])
        .collect();
    src[4] = [300.0, 500.0];
    let dst = destinations(&src);

    assert!(design_ratio(&src, &dst) < TOLERANCE);
    assert!(solve_dlt_homography(&src, &dst).is_none());
}

/// **9** of 10 source points on one line, the 10th well off it.
///
/// This is the larger-sample form of the 3-collinear case, and it is the
/// realistic one for a feature tracker: 8 of 10 collinear is *not* degenerate
/// (measured `sigma_8/sigma_1 = 3.5e-2`, transfer error `2.3e-13 px`) because
/// two off-line points already pin the vanishing point. Nine is the line.
#[test]
fn nine_of_ten_collinear_is_refused() {
    let mut src: Vec<[f64; 2]> = (0..10)
        .map(|i| [20.0 + i as f64 * 70.0, 55.0 + i as f64 * 35.0])
        .collect();
    src[9] = [700.0, 470.0];
    let dst = destinations(&src);

    assert!(
        design_ratio(&src, &dst) < TOLERANCE,
        "9 collinear points must be rank deficient"
    );
    assert!(solve_dlt_homography(&src, &dst).is_none());
}

/// The mirror of the test above, and the reason the criterion cannot be a mere
/// "are any three points collinear" heuristic: 8 of 10 collinear is solvable
/// and must stay solvable, so the solver has to notice that the *last two*
/// points rescue the configuration.
#[test]
fn eight_of_ten_collinear_is_still_solved() {
    let mut src: Vec<[f64; 2]> = (0..10)
        .map(|i| [20.0 + i as f64 * 70.0, 55.0 + i as f64 * 35.0])
        .collect();
    src[8] = [10.0, 500.0];
    src[9] = [700.0, 470.0];
    let dst = destinations(&src);

    let ratio = design_ratio(&src, &dst);
    assert!(
        ratio > 1e3 * TOLERANCE,
        "two off-line points rescue 8 collinear ones: {ratio:e}"
    );
    let est = solve_dlt_homography(&src, &dst).expect("8-of-10 must still solve");
    let est = est / est[(2, 2)];
    assert!(
        transfer_error(&src, &est) < 1e-6,
        "8-of-10 must still be solved accurately, got {:e} px",
        transfer_error(&src, &est)
    );
}

// ---------------------------------------------------------------------------
// NEAR-DEGENERATE: pins the criterion from both sides
// ---------------------------------------------------------------------------

/// 4 points where the middle one is displaced `e` perpendicular to the chord
/// through the other three.
///
/// As `e -> 0` the set becomes collinear and `sigma_8 / sigma_1` decays
/// *linearly* in `e` (measured: `r8 = 1.008e-3 * e` across twelve decades), so
/// this is a clean one-parameter walk down the threshold - the right family for
/// pinning a criterion, because "near-degenerate" has a number attached to it.
fn near_collinear(e: f64) -> ([[f64; 2]; 4], Vec<[f64; 2]>) {
    let src: [[f64; 2]; 4] = [[0.0, 0.0], [220.0, e], [440.0, 0.0], [300.0, 500.0]];
    let dst = destinations(&src);
    (src, dst)
}

/// Just inside and just outside the threshold, both located by bisection
/// rather than hard-coded, so the test pins the *boundary* and cannot drift
/// away from it if the normalisation ever changes.
#[test]
fn near_collinear_is_accepted_inside_the_threshold_and_refused_outside() {
    // r8 is proportional to e, so bisect on log(e).
    let (mut e_inside, mut e_outside) = (1e3f64, 1e-30f64);
    for _ in 0..200 {
        let mid = (e_inside * e_outside).sqrt();
        let (s, d) = near_collinear(mid);
        if design_ratio(&s, &d) > TOLERANCE {
            e_inside = mid;
        } else {
            e_outside = mid;
        }
    }

    // Just inside: a factor of 10 above the threshold, and a real homography.
    let (src_ok, dst_ok) = near_collinear(e_inside);
    let ratio_ok = design_ratio(&src_ok, &dst_ok);
    assert!(
        ratio_ok > TOLERANCE && ratio_ok < 10.0 * TOLERANCE,
        "the inside case is not just inside: {ratio_ok:e}"
    );
    let est = solve_dlt_homography(&src_ok, &dst_ok).expect(
        "a set just inside the threshold must still solve; it is a legitimate \
         homography, merely an ill-conditioned one",
    );
    let est = est / est[(2, 2)];
    let err = transfer_error(&src_ok, &est);
    assert!(
        err < 1e-3,
        "just inside the threshold the solve must still be accurate: {err:e} px"
    );

    // Just outside: a factor of 10 below, and refused.
    let (src_bad, dst_bad) = near_collinear(e_outside);
    let ratio_bad = design_ratio(&src_bad, &dst_bad);
    assert!(
        ratio_bad < TOLERANCE && ratio_bad > 0.01 * TOLERANCE,
        "the outside case is not just outside: {ratio_bad:e}"
    );
    assert!(
        solve_dlt_homography(&src_bad, &dst_bad).is_none(),
        "a set just outside the threshold must be refused, ratio {ratio_bad:e}"
    );
}

/// Past exact degeneracy the SVD's answer becomes noise: the vector it returns
/// is no longer even a good fit to the correspondences it was built from.
///
/// This is the strongest statement of the defect's cost. It is what the direct
/// caller was actually handed.
#[test]
fn past_the_threshold_the_solved_matrix_is_garbage_not_merely_degenerate() {
    // e = 0 is *exactly* 3 collinear points. This is the pre-fix symptom.
    let (src, dst) = near_collinear(0.0);
    let est = solve_dlt_homography(&src, &dst);
    assert!(
        est.is_none(),
        "an exactly collinear set must not yield a matrix at all"
    );
}

/// Whatever survives must be usable: rank 3, finite entries, and it must map
/// its own input to something finite. This is the invariant the RANSAC path
/// was silently enforcing via `INFINITY` in `compute_error`; direct callers had
/// no such protection.
#[test]
fn no_returned_homography_maps_its_own_input_to_infinity() {
    let cases: Vec<Vec<[f64; 2]>> = vec![
        vec![[0.0, 0.0], [200.0, 100.0], [400.0, 200.0], [320.0, 480.0]],
        vec![[0.0, 0.0], [160.0, 80.0], [320.0, 160.0], [480.0, 240.0]],
        vec![[0.0, 0.0], [220.0, 0.0], [440.0, 0.0], [300.0, 500.0]],
        vec![[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [37.0, 611.0]],
    ];
    let mut checked = 0;
    for src in cases {
        let dst = destinations(&src);
        if let Some(h) = solve_dlt_homography(&src, &dst) {
            checked += 1;
            assert!(
                h.iter().all(|v| v.is_finite()),
                "H has a non-finite entry: {h}"
            );
            let sv = h.svd(false, false).singular_values;
            assert!(
                sv[2] > 1e-6 * sv[0],
                "H is not rank 3: singular values {sv:?}"
            );
            for p in src.iter() {
                assert!(
                    project(&h, *p).is_some(),
                    "H projected {p:?} to infinity; H = {h}"
                );
            }
        }
    }
    assert!(
        checked >= 1,
        "at least the control must have produced a matrix"
    );
}

// ---------------------------------------------------------------------------
// `enforce_rank2`: refuse matrices it cannot meaningfully rank-reduce
// ---------------------------------------------------------------------------

#[test]
fn enforce_rank2_refuses_inputs_it_cannot_project() {
    // Already rank 1: U diag(s1, s2, 0) V^T is a no-op, so returning `Some`
    // would claim a rank-2 result the caller can verify is still rank 1.
    let rank1 = Matrix3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
    assert!(
        cv_calib3d::enforce_rank2(&rank1).is_none(),
        "a rank-1 matrix cannot be rank-reduced, and saying so via Some would \
         hand the caller a matrix that is still rank 1"
    );

    // Zero matrix: nothing to project onto, and the caller would divide by it.
    assert!(cv_calib3d::enforce_rank2(&Matrix3::<f64>::zeros()).is_none());

    // Non-finite: checked before the SVD, which does not terminate on NaN.
    let nan = Matrix3::new(f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    assert!(cv_calib3d::enforce_rank2(&nan).is_none());
}

/// ...and must still accept the normal case, including a matrix that is
/// *already* exactly rank 2 - which is what the 8-point solver produces on
/// every well-conditioned input, and which an over-eager guard would reject.
#[test]
fn enforce_rank2_accepts_a_genuine_rank_two_matrix() {
    // An epipolar geometry: F = [0, -1, 0; 1, 0, 0; 0, 0, 0] has exactly two
    // nonzero singular values.
    let f = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let out = cv_calib3d::enforce_rank2(&f).expect("rank 2 must be accepted");
    let sv = out.svd(false, false).singular_values;
    assert!(
        sv[2] < 1e-12 * sv[0],
        "output should still be rank 2: {sv:?}"
    );
    assert!((out.norm() - f.norm()).abs() < 1e-12);
}

/// The end-to-end check that the `enforce_rank2` guard did not break the 8-point
/// solver: a well-conditioned 8-point set still yields a rank-2 F.
#[test]
fn the_eight_point_solver_still_produces_a_rank_two_fundamental_matrix() {
    let (p1, p2) = synthetic_stereo(8);
    let f = cv_calib3d::solve_dlt_fundamental(&p1, &p2).expect("F must be found");
    let sv = f.svd(false, false).singular_values;
    assert!(sv[2] < 1e-9 * sv[0], "F is not rank 2: {sv:?}");
    let mut worst = 0.0f64;
    for (a, b) in p1.iter().zip(p2.iter()) {
        let x1 = Vector3::new(a[0], a[1], 1.0);
        let x2 = Vector3::new(b[0], b[1], 1.0);
        worst = worst.max(x2.dot(&(f * x1)).abs());
    }
    assert!(worst < 1e-6, "epipolar residual {worst}");
}

/// **And** collinear-in-one-image geometry is refused - for a reason worth
/// pinning, because it is *not* the same reason as for `H`.
///
/// The tempting assumption is that collinear points in one image are fine for
/// `F`: scene points on a 3D line project to a line in both images, and the
/// epipolar line is exactly what `F` predicts. Measurement contradicts it. With
/// 8 points collinear in image 1 the design matrix drops to rank 6 and the DLT
/// returns a **rank-1** matrix `F = a b^T`. That matrix satisfies the sample
/// perfectly and means nothing:
///
/// ```text
/// x2' F x1 = x2' a * (b' x1)
/// ```
///
/// and `b' x1 = 0` for every one of them, since they are on the line `b' x = 0`.
/// The residual is *exactly* zero for any `a` and any `b`. So a rank-1 `F`
/// "fits" a collinear sample by construction while encoding no epipolar geometry
/// at all - it just restates that the points are collinear.
///
/// Pre-fix, this solver returned exactly such a matrix, with a measured
/// epipolar residual of `8.9e-16`: a perfect score on information the score
/// cannot see.
#[test]
fn collinear_points_in_one_image_yield_no_fundamental_matrix() {
    let (p1, p2) = stereo_from_3d_line(8);
    assert!(
        cv_calib3d::solve_dlt_fundamental(&p1, &p2).is_none(),
        "a rank-1 F would satisfy these correspondences with zero residual \
         while encoding no epipolar geometry; None is the honest answer"
    );

    // 20 points on the same line: same verdict, so the guard is not a
    // minimal-sample artefact.
    let (p1, p2) = stereo_from_3d_line(20);
    assert!(cv_calib3d::solve_dlt_fundamental(&p1, &p2).is_none());
}

/// The boundary on the `F` side: **3** collinear points in image 1, not 8.
///
/// This is what separates "collinear" from "degenerate" for a fundamental
/// matrix. The 3-dof family the collinearity introduces is removed again by the
/// remaining general points, the design matrix keeps rank 8, and the solve
/// succeeds. A rule borrowed from the homography path - "reject if any three
/// are collinear" - would throw this away.
///
/// Note the epipolar residual is deliberately *not* asserted here. Moving three
/// points onto a line in image 1 without moving their image-2 partners makes
/// the correspondences deliberately inconsistent, so a least-squares fit
/// necessarily has a residual against them; the property under test is that the
/// solver still returns a rank-2 `F` rather than `None`.
#[test]
fn only_three_collinear_points_in_one_image_still_yield_a_fundamental_matrix() {
    let (mut p1, p2) = synthetic_stereo(20);
    let dir = [p1[2][0] - p1[0][0], p1[2][1] - p1[0][1]];
    for i in 0..3 {
        let s = i as f64;
        p1[i] = [p1[0][0] + dir[0] * s, p1[0][1] + dir[1] * s];
    }

    let f = cv_calib3d::solve_dlt_fundamental(&p1, &p2)
        .expect("3 collinear points in image 1 is not a degeneracy for F");
    let sv = f.svd(false, false).singular_values;
    assert!(sv[2] < 1e-9 * sv[0], "F is not rank 2: {sv:?}");
}

/// Coplanar scenes are accepted. `F` is not *uniquely* determined by them, but
/// a valid one is produced and it genuinely satisfies the correspondences;
/// refusing would break the stereo-calibration and essential-matrix paths that
/// share this 8-point step.
#[test]
fn a_coplanar_scene_is_still_accepted() {
    let (p1, p2) = stereo_from_plane(12);
    let f = cv_calib3d::solve_dlt_fundamental(&p1, &p2)
        .expect("coplanar correspondences do determine a fundamental matrix");
    let sv = f.svd(false, false).singular_values;
    assert!(sv[2] < 1e-9 * sv[0], "F is not rank 2: {sv:?}");
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Two pinhole cameras with a fixed relative pose.
fn stereo_rig() -> (Matrix3<f64>, Matrix3<f64>, Vector3<f64>) {
    let k = Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0);
    let angle = 0.15f64;
    let r = Matrix3::new(
        angle.cos(),
        0.0,
        angle.sin(),
        0.0,
        1.0,
        0.0,
        -angle.sin(),
        0.0,
        angle.cos(),
    );
    (k, r, Vector3::new(-0.5, 0.0, 0.0))
}

fn project3d(k: &Matrix3<f64>, x: &Vector3<f64>, r: &Matrix3<f64>, t: &Vector3<f64>) -> [f64; 2] {
    let c2 = k * (r * x + t);
    [c2[0] / c2[2], c2[1] / c2[2]]
}

/// `n` general 3D points.
fn synthetic_stereo(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (k, r, t) = stereo_rig();
    let mut seed = 0xabcdefu64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    let mut i = 0;
    while p1.len() < n && i < 10 * n {
        i += 1;
        let x = Vector3::new(next() * 4.0, next() * 4.0, 5.0 + next() * 3.0);
        let v = k * x;
        p1.push([v[0] / v[2], v[1] / v[2]]);
        p2.push(project3d(&k, &x, &r, &t));
    }
    (p1, p2)
}

/// `n` points on a straight 3D line.
fn stereo_from_3d_line(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (k, r, t) = stereo_rig();
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    for i in 0..n {
        let s = i as f64 * 0.4 - n as f64 * 0.2;
        let x = Vector3::new(s, 0.5, 6.0);
        let v = k * x;
        p1.push([v[0] / v[2], v[1] / v[2]]);
        p2.push(project3d(&k, &x, &r, &t));
    }
    (p1, p2)
}

/// `n` points on a plane at constant depth.
fn stereo_from_plane(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let (k, r, t) = stereo_rig();
    let mut seed = 0x1234_5678u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let x = Vector3::new(next() * 3.0, next() * 2.0, 6.0);
        let v = k * x;
        p1.push([v[0] / v[2], v[1] / v[2]]);
        p2.push(project3d(&k, &x, &r, &t));
    }
    (p1, p2)
}
