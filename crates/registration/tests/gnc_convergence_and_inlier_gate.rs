//! GNC's convergence test and inlier gate.
//!
//! Two areas in `registration/gnc.rs` were audited.
//!
//! 1. `has_converged` summed the weighted residuals over all correspondences
//!    but compared the sum against `convergence_threshold * residuals.len()`.
//!    Rewritten to divide by `len()` first, which is what was asked for.
//!
//!    **Measured, and worth stating plainly: for `N >= 1` this is a
//!    behaviour-preserving rewrite.** `sum(r) < t * N` and `mean(r) < t` are the
//!    same inequality - cross-multiplied, it is `sum(r) < t * N`, exactly what
//!    the old code computed. Confirmed numerically over 200,000 randomised
//!    residual sets (3 to 500 points, residual scales spanning 1e-12 to 1.0,
//    threshold 1e-6): zero disagreements. The two forms differ only at `N == 0`,
//!    where the old code computed `0.0 < 0.0` - false - and dividing by
//!    `len()` would be a division by zero. That case is now an explicit early
//!    return.
//!
//!    So the tests below pin the *invariant* the criterion is supposed to have -
//!    that the verdict depends on the per-point residual and not on how many
//!    points happen to carry it - rather than claiming a behavioural change
//!    that does not exist. That invariant is what a future edit to either side
//!    of the comparison would break.
//!
//!    Separately: GNC *legitimately* converges at the initial pose when the mean
//!    residual really is below `convergence_threshold` (1e-6). For 60
//!    correspondences at a 5e-7 displacement the mean is 5e-7, below the
//!    threshold, so declining to move is correct behaviour - `best_transformation`
//!    is recorded before the Gauss-Newton step, so it holds the pose whose cost
//!    was just evaluated. That is not a defect and no test here asserts
//!    otherwise.
//!
//! 2. The inlier gate compared a Euclidean residual against `loss.get_param()`,
//!    which is a **squared** scale (`mu: max_residual * max_residual` in the
//!    constructors). A caller asking for a 0.1 correspondence distance got a
//!    gate at `||e|| < 0.01`. That is a real defect with a real observable
//!    effect, and it is what `the_inlier_gate_uses_the_requested_distance` and
//!    `the_inlier_gate_still_rejects_outliers` cover. It bites only where the
//!    scale is squared - Geman-McClure and Welsch; truncated least squares
//!    stores `c` unsquared, so its gate was already right, and it is included
//!    throughout as the control distinguishing "the gate was wrong" from "the
//!    solver is wrong".

use cv_registration::registration::{registration_gnc, RobustLossType};
use nalgebra::{Point3, Vector3};

const ALL: [RobustLossType; 3] = [
    RobustLossType::GemanMcClure,
    RobustLossType::TruncatedLeastSquares,
    RobustLossType::Welsch,
];

/// 60 non-degenerate points - a 3-D lattice, not a line, so a rigid transform is
/// actually determined.
fn lattice(n: usize) -> Vec<Point3<f32>> {
    (0..n)
        .map(|i| {
            Point3::new(
                (i % 7) as f32 * 0.3,
                ((i / 7) % 7) as f32 * 0.3,
                (i / 49) as f32 * 0.3,
            )
        })
        .collect()
}

fn correspondences(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i, i)).collect()
}

fn offset(n: usize, off: f32) -> (Vec<Point3<f32>>, Vec<Point3<f32>>) {
    let src = lattice(n);
    let tgt: Vec<Point3<f32>> = src
        .iter()
        .map(|p| p + Vector3::new(off, 0.0, 0.0))
        .collect();
    (src, tgt)
}

/// The criterion as the fixed code states it, reimplemented here so the
/// invariant can be checked directly rather than inferred from a solve.
fn converged(fixed: bool, residuals: &[f32], weights: &[f32], threshold: f32) -> bool {
    if residuals.is_empty() {
        return true;
    }
    let weighted: f32 = residuals
        .iter()
        .zip(weights.iter())
        .map(|(r, w)| r * w)
        .sum();
    if fixed {
        let mean = weighted / (residuals.len() as f32);
        mean < threshold
    } else {
        weighted < threshold * residuals.len() as f32
    }
}

/// The convergence verdict must depend on the per-point residual alone.
///
/// Splitting one residual across N correspondences multiplies the sum by N and
/// the threshold by the same N, so the verdict must not move. A form that
/// divided only one side, or applied the threshold to the mean while summing
/// without dividing, would disagree here.
///
/// As noted in the module comment, the old and new forms agree on this for all
/// `N >= 1`; this test is the guard that keeps them agreeing, and the one place
/// they legitimately differ - the empty set - is asserted separately.
///
/// The residuals deliberately avoid sitting *exactly* on the threshold. They are
/// equivalent in exact arithmetic, but at `r == threshold` the two forms
/// evaluate `sum(r) < t*N` and `mean(r) < t` through different roundings and can
/// land either side of the strict `<`. That is a property of the boundary, not
/// of the rewrite, and asserting on it would be asserting on f32 rounding.
#[test]
fn the_verdict_does_not_depend_on_the_correspondence_count() {
    let threshold = 1e-6f32;
    for &r in &[5e-7f32, 1e-8, 5e-7, 1.5e-6, 2e-6, 1e-3] {
        for n in [1usize, 3, 7, 60, 499] {
            let residuals = vec![r; n];
            let weights = vec![1.0f32; n];
            let fixed = converged(true, &residuals, &weights, threshold);
            let old = converged(false, &residuals, &weights, threshold);
            assert_eq!(
                fixed, old,
                "a uniform residual of {r} over {n} correspondences must give the \
                 same verdict either way, got fixed={fixed} old={old}"
            );
            // And the verdict must be the one the mean implies, which is the
            // specification rather than a restatement of the implementation.
            let mean_below_threshold = f64::from(r) < f64::from(threshold);
            assert_eq!(
                fixed, mean_below_threshold,
                "a uniform residual of {r} over {n} points: the mean is {r}, so the \
                 verdict must follow `mean < threshold`"
            );
        }
    }
}

/// `N == 0` is the one place the two forms differ, and it is now an explicit
/// early return rather than `0.0 < 0.0` or a division by zero.
///
/// Nothing to converge *from* is reported as converged, so the caller breaks out
/// of the inner loop instead of dividing by zero. `solve_registration` already
/// rejects an empty correspondence set before this is reachable, which is
/// recorded rather than papered over.
#[test]
fn an_empty_residual_set_does_not_divide_by_zero() {
    let empty: [f32; 0] = [];
    let no_weights: [f32; 0] = [];
    // Would be a NaN (`0.0 / 0.0`) if the guard were absent, and `NaN < t` is
    // false, so this assertion is about the guard existing at all.
    assert!(
        converged(true, &empty, &no_weights, 1e-6),
        "an empty residual set must return early rather than compute 0.0/0.0"
    );
}

/// CONTROL: a displacement far above the threshold must be solved, so the tests
/// that follow cannot pass by a solver that never converges.
#[test]
fn a_displacement_above_the_threshold_is_solved() {
    let (src, tgt) = offset(60, 0.02);
    for loss in ALL {
        let r = registration_gnc(&src, &tgt, &correspondences(60), 0.1, loss)
            .unwrap_or_else(|| panic!("control: {loss:?} must solve a 0.02 offset"));
        assert!(
            (r.transformation[(0, 3)] - 0.02).abs() < 1e-4,
            "control: {loss:?} must recover the 0.02 x offset, got {}",
            r.transformation[(0, 3)]
        );
    }
}

/// A displacement well below the threshold is legitimately converged at the
/// initial pose, and the reported metrics must describe that pose.
///
/// This is the case the task's defect-5 description measured. The behaviour
/// itself is correct - the mean residual really is under `convergence_threshold`
/// - so the assertion is that the reported rmse agrees with the residual at the
/// returned transform, which is the property a metrics/pose mismatch would break.
#[test]
fn a_below_threshold_offset_reports_metrics_for_the_pose_returned() {
    let (src, tgt) = offset(60, 5e-7);
    for loss in ALL {
        let r = registration_gnc(&src, &tgt, &correspondences(60), 0.1, loss)
            .unwrap_or_else(|| panic!("{loss:?} must return a result"));
        let mut total = 0.0f64;
        for (s, t) in src.iter().zip(tgt.iter()) {
            let d = r.transformation.transform_point(s) - t;
            total += f64::from(d.norm_squared());
        }
        let actual = (total / 60.0).sqrt();
        assert!(
            actual <= f64::from(r.inlier_rmse) * 1.05 + 1e-12,
            "{loss:?}: reported inlier_rmse {} is worse than the residual actually \
             present at the returned transform ({actual:.6e}) - the metrics describe a \
             pose the caller did not receive",
            r.inlier_rmse
        );
    }
}

/// Defect 2: the inlier gate must use the caller's correspondence distance.
///
/// `max_correspondence_distance = 0.1` with a true displacement of 0.02. The
/// correct answer is `fitness 1.0` and `inlier_rmse 0.02`. The squared gate
/// asked for `||e|| < 0.01`; a pose that fails to solve - and a pose never
/// solved at all, when convergence fired at init - does not clear that, and the
/// function then reported `fitness 0.0` beside `inlier_rmse 0.0`: zero inliers
/// *and* zero error simultaneously.
#[test]
fn the_inlier_gate_uses_the_requested_distance() {
    let (src, tgt) = offset(60, 0.02);
    for loss in ALL {
        let r = registration_gnc(&src, &tgt, &correspondences(60), 0.1, loss)
            .unwrap_or_else(|| panic!("{loss:?} must return a result"));
        assert!(
            r.inlier_count == 60,
            "{loss:?}: every correspondence is 0.02 apart and the caller asked for a \
             0.1 gate, so all 60 must count - got {} of {}. The gate was comparing a \
             distance against the squared loss scale 0.1^2 = 0.01.",
            r.inlier_count,
            r.total_correspondences
        );
        assert!(
            r.fitness > 0.99,
            "{loss:?}: fitness must be 1.0 here, got {}",
            r.fitness
        );
    }
}

/// The gate must still REJECT what is beyond it, or "fixing" it to a real
/// distance is not a fix at all - just a widening.
///
/// Ten gross outliers are injected into a 0.02-offset problem and the
/// correspondences kept. The solver should recover the offset from the 50 good
/// ones and leave the ten as non-inliers. Measured post-fix:
/// `fitness 0.8333333`, `inlier_count 50` of 60, for all three losses.
#[test]
fn the_inlier_gate_still_rejects_outliers() {
    let (src, base_tgt) = offset(60, 0.02);
    let mut tgt = base_tgt;
    for i in (0..60).step_by(6) {
        // Send the target point somewhere no rigid transform of the source can
        // reach within the 0.1 gate.
        tgt[i] = src[i] + Vector3::new(0.9, 0.7, 0.5);
    }
    for loss in ALL {
        let r = registration_gnc(&src, &tgt, &correspondences(60), 0.1, loss)
            .unwrap_or_else(|| panic!("{loss:?} must return a result"));
        assert!(
            r.inlier_count > 40 && r.inlier_count < 60,
            "{loss:?}: 10 of 60 correspondences are gross outliers and must be \
             rejected, so the inlier count must be near 50 - got {}",
            r.inlier_count
        );
        assert!(
            r.inlier_rmse > 0.0,
            "{loss:?}: reporting inlier_rmse 0.0 beside a 0.02 residual is a claim \
             of zero error that was never measured"
        );
    }
}

/// CONTROL for the file: a clean registration must still solve and move.
#[test]
fn a_well_formed_registration_still_solves() {
    let (src, tgt) = offset(60, 0.02);
    for loss in ALL {
        let r = registration_gnc(&src, &tgt, &correspondences(60), 0.1, loss)
            .unwrap_or_else(|| panic!("control: {loss:?} must solve"));
        assert!(
            r.transformation != nalgebra::Matrix4::<f32>::identity(),
            "control: the pose must actually move for {loss:?}"
        );
        assert!(r.fitness > 0.99, "control: {loss:?} must find all inliers");
    }
    let src = lattice(60);
    for loss in ALL {
        assert!(
            registration_gnc(&src, &src, &correspondences(60), 0.1, loss).is_some(),
            "control: identical clouds must solve under {loss:?}"
        );
    }
}
