//! GNC's parameter schedule must anneal from the convex surrogate toward the
//! target parameter, and must not leak state between calls.
//!
//! `compute_parameter_schedule` read the scale from `self.loss`, which the
//! previous outer iteration had already overwritten with `update_param`. Every
//! step therefore multiplied the *last* scheduled value by up to 100 instead of
//! interpolating between `100 * mu0` and `mu0`, so mu grew geometrically
//! (Geman-McClure, mu0 = 0.0025: ~2.7e12 after one solve, ~3e27 after a second
//! one on the same optimizer). At that scale every weight is ~2, the solve is
//! plain least squares, and outliers drag the pose.

use cv_registration::registration::GNCOptimizer;
use nalgebra::{Matrix4, Point3, Rotation3, Translation3};

/// Small deterministic LCG so the test needs no `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn ground_truth() -> Matrix4<f32> {
    let r = Rotation3::from_euler_angles(0.05, -0.03, 0.08);
    let t = Translation3::new(0.1, -0.05, 0.07);
    (t * r).to_homogeneous()
}

/// 200 correspondences, the first 40% of which are replaced by random target
/// points (outliers).
fn scene() -> (Vec<Point3<f32>>, Vec<Point3<f32>>, Vec<(usize, usize)>) {
    let mut rng = Lcg(0x5eed);
    let n = 200;
    let gt = ground_truth();
    let source: Vec<Point3<f32>> = (0..n)
        .map(|_| Point3::new(rng.next_f32(), rng.next_f32(), rng.next_f32()))
        .collect();
    let mut target: Vec<Point3<f32>> = source.iter().map(|p| gt.transform_point(p)).collect();
    for t in target.iter_mut().take(n * 2 / 5) {
        *t = Point3::new(rng.next_f32(), rng.next_f32(), rng.next_f32()) * 2.0;
    }
    let corr = (0..n).map(|i| (i, i)).collect();
    (source, target, corr)
}

fn pose_error(a: &Matrix4<f32>, b: &Matrix4<f32>) -> f32 {
    (a - b).abs().max()
}

#[test]
fn geman_mcclure_recovers_the_pose_under_forty_percent_outliers() {
    let (src, tgt, corr) = scene();
    let mut gnc = GNCOptimizer::new_geman_mcclure(0.05);
    let r = gnc
        .solve_registration(&src, &tgt, &corr, &Matrix4::identity())
        .expect("GNC must solve a well-posed problem");
    let err = pose_error(&r.transformation, &ground_truth());
    assert!(
        err < 1e-3,
        "pose error {err:.6} under 40% outliers: the schedule is not annealing \
         toward mu0, so the solve degenerates to plain least squares"
    );
    assert_eq!(
        r.inlier_count, 120,
        "exactly the 120 true inliers must pass"
    );
}

#[test]
fn the_loss_parameter_is_restored_after_a_solve() {
    let (src, tgt, corr) = scene();
    for (mut gnc, p0) in [
        (GNCOptimizer::new_geman_mcclure(0.05), 0.0025f32),
        (GNCOptimizer::new_welsch(0.05), 0.0025),
        (GNCOptimizer::new_tls(0.05), 0.05),
    ] {
        gnc.solve_registration(&src, &tgt, &corr, &Matrix4::identity())
            .expect("solve");
        let p = gnc.loss.get_param();
        assert!(
            (p - p0).abs() <= 1e-9,
            "loss parameter after a solve is {p:e}, expected the target {p0:e}: \
             the schedule leaked its state out of solve_registration"
        );
    }
}

#[test]
fn repeated_calls_give_the_same_answer() {
    let (src, tgt, corr) = scene();
    let mut gnc = GNCOptimizer::new_geman_mcclure(0.05);
    let a = gnc
        .solve_registration(&src, &tgt, &corr, &Matrix4::identity())
        .expect("first solve");
    let b = gnc
        .solve_registration(&src, &tgt, &corr, &Matrix4::identity())
        .expect("second solve");
    assert_eq!(
        a.transformation, b.transformation,
        "the same optimizer gave a different pose on its second call"
    );
    assert_eq!(a.inlier_count, b.inlier_count);
    assert_eq!(a.cost, b.cost);
}

#[test]
fn the_parameter_stays_within_target_and_surrogate_across_calls() {
    // Before the fix the parameter left behind by one solve seeded the next,
    // so it compounded across calls (2.7e12 after one, ~3e27 after two).
    let (src, tgt, corr) = scene();
    let mut gnc = GNCOptimizer::new_geman_mcclure(0.05);
    for _ in 0..3 {
        gnc.solve_registration(&src, &tgt, &corr, &Matrix4::identity())
            .expect("solve");
        let mu = gnc.loss.get_param();
        assert!(
            (0.0025..=0.25).contains(&mu),
            "mu = {mu:e} escaped [mu0, 100 mu0] = [2.5e-3, 2.5e-1]"
        );
    }
}
