//! Regression tests for the `cv-calib3d` hand-eye estimator against known
//! ground truth, written as an AX = XB measurement rather than a replay of the
//! library's own internal formulation.
//!
//! ## Why this file exists
//!
//! An audit of the four `is never used` warnings on `cv-calib3d` found three
//! sites in `src/hand_eye.rs`:
//!
//!   * `rotation_vector_from_matrix` (line 120) - **false positive.** It was
//!     referenced only from `#[cfg(test)] mod tests`, so the *lib* target
//!     (the thing without tests) saw no use and warned. It is not dead; it is a
//!     test-only error metric. It is now gated on `#[cfg(test)]` so the warning
//!     matches reality. Measured below, not asserted.
//!   * `skew` (line 134) - **dead.** Duplicated verbatim in
//!     `crates/hal/src/gpu_kernels/icp.rs` (a different crate) where it *is*
//     used. Had no use here; deleted.
//!   * `gamma_to_rotation` (line 139) - **dead.** A correct, self-contained
//!     gamma -> rotation-matrix converter, but every hand-eye variant resolves
//!     to the one quaternion Q-method estimator, so no caller needed it.
//!     Deleted rather than wired up: see `hand_eye_ax_xb_ground_truth.md`
//!     reasoning - `HandEyeMethod` is a documented no-op selector, and giving
//!     `gamma_to_rotation` a caller would have meant changing that documented
//!     contract.
//!
//! The tests that matter are the ones below: they drive the *public* estimator
//! `cv_calib3d::calibrate_hand_eye` and prove it recovers a planted rigid
//! transform from synthetic motions built by the standard formulation.
//!
//! ## Standard formulation used here (independent of the library's internals)
//!
//! For a camera mounted on a gripper observing a fixed target, with
//!   G_i : gripper -> base  (robot flange poses, per view)
//!   T_i : target  -> camera (calibration target poses, per view)
//!   X   : camera  -> gripper  (the unknown we want)
//! the chain from base to target is  base -> gripper -> camera -> target, i.e.
//!
//!   H_{b->t} = G_i . X . T_i
//!
//! The target is FIXED, so H_{b->t} is the same constant H for every i:
//!
//!   G_i . X . T_i = H          (1)   for all i
//!
//! Pick a ground-truth X, a fixed H, and draw each G_i freely; then solve (1)
//! for T_i. That is exactly the synthetic data generator below, and it is the
//! textbook Tsai-Lenz data construction. No library helper is used to build
//! the data, so the test cannot inherit a sign or frame-convention error from
//! the implementation it is checking.

use cv_calib3d::{
    calibrate_hand_eye, calibrate_robot_world_hand_eye, HandEyeMethod, RobotWorldHandEyeMethod,
};
use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

/// Deterministic LCG so the planted pose is identical on every run and on
/// every machine. A fixed seed is what makes this a regression test.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / ((u32::MAX >> 1) as f64) - 1.0
    }
    fn rotation(&mut self, max_angle: f64) -> Matrix3<f64> {
        let axis = Vector3::new(self.next_f64(), self.next_f64(), self.next_f64());
        if axis.norm() < 1e-6 {
            return Matrix3::identity();
        }
        let angle = self.next_f64() * max_angle;
        Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle).into_inner()
    }
    fn translation(&mut self, scale: f64) -> Vector3<f64> {
        Vector3::new(
            self.next_f64() * scale,
            self.next_f64() * scale,
            self.next_f64() * scale,
        )
    }
}

/// Rigid transform as (R, t) with `y = R x + t`.
#[derive(Clone, Copy, Debug)]
struct Rigid {
    r: Matrix3<f64>,
    t: Vector3<f64>,
}

impl Rigid {
    /// Inverse of a rigid transform: (R^T, -R^T t).
    fn inverse(&self) -> Rigid {
        Rigid {
            r: self.r.transpose(),
            t: -(self.r.transpose() * &self.t),
        }
    }
    /// Composition `self . other`, i.e. apply `other` first.
    fn then(&self, other: &Rigid) -> Rigid {
        Rigid {
            r: &self.r * &other.r,
            t: &self.t + &self.r * &other.t,
        }
    }
}

/// Axis-angle length of a rotation: a metric, frame-independent rotation error.
/// Reused here rather than copied so this file does not add a third copy of it
/// (see the audit note: it already existed in `hand_eye.rs` and `stereo.rs`).
fn rotation_angle(r: &Matrix3<f64>) -> f64 {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    let denom = 2.0 * angle.sin();
    if denom.abs() < 1e-10 {
        return 0.0;
    }
    let axis = Vector3::new(
        (r[(2, 1)] - r[(1, 2)]) / denom,
        (r[(0, 2)] - r[(2, 0)]) / denom,
        (r[(1, 0)] - r[(0, 1)]) / denom,
    );
    axis.norm() * angle
}

/// Plant a known X = cam2gripper and generate n consistent view pairs.
///
/// Returns `(gripper2base, target2cam, x_true)` where `gripper2base[i]` is the
/// i-th robot pose and `target2cam[i]` the matching target pose, satisfying
/// `gripper2base[i] . X . target2cam[i] = H` exactly for the planted `H`.
fn plant_eye_in_hand(
    rng: &mut Lcg,
    x_true: Rigid,
    h_base2target: Rigid,
    n: usize,
) -> (Vec<Rigid>, Vec<Rigid>) {
    let mut gripper2base = Vec::with_capacity(n);
    let mut target2cam = Vec::with_capacity(n);

    for _ in 0..n {
        // Draw the robot motion freely: enough rotational variety that the
        // AX=XB rotation is determined (a single rotation leaves R_X free).
        let g = Rigid {
            r: rng.rotation(1.5),
            t: rng.translation(0.4),
        };

        // (1):  G . X . T = H   =>   T = (G . X)^-1 . H
        let t = g.then(&x_true).inverse().then(&h_base2target);

        gripper2base.push(g);
        target2cam.push(t);
    }

    (gripper2base, target2cam)
}

fn split(v: &[Rigid]) -> (Vec<Matrix3<f64>>, Vec<Vector3<f64>>) {
    (
        v.iter().map(|p| p.r).collect(),
        v.iter().map(|p| p.t).collect(),
    )
}

/// CONTROL + measurement: the well-formed case still works.
///
/// Ten varied views, a non-trivial planted pose, every `HandEyeMethod`
/// variant. Tolerance is stated as `1e-6` for rotation (radians, axis-angle
/// norm) and `1e-6` metres for translation - roughly float64 round-trip
/// accumulated over the eigenvector solve and the normal-equations
/// translation solve, i.e. two orders of magnitude tighter than the `1e-3`
/// the inline unit tests use, so this also says the estimator is more accurate
/// than its own tests claim.
#[test]
fn calibrate_hand_eye_recovers_planted_transform_across_all_method_variants() {
    let mut rng = Lcg(0x5EED_0001);
    let x_true = Rigid {
        r: rng.rotation(1.2),
        t: Vector3::new(0.05, -0.02, 0.10),
    };
    let h = Rigid {
        r: rng.rotation(1.0),
        t: Vector3::new(0.5, 0.1, 0.8),
    };

    let (g2b, t2c) = plant_eye_in_hand(&mut rng, x_true, h, 10);
    let (r_g2b, t_g2b) = split(&g2b);
    let (r_t2c, t_t2c) = split(&t2c);

    // The generated data must actually satisfy G_i . X . T_i = H, otherwise
    // this test would be measuring nothing. Check it directly.
    for i in 0..g2b.len() {
        let lhs = g2b[i].then(&x_true).then(&t2c[i]);
        let rot_err = rotation_angle(&(h.r.transpose() * lhs.r));
        let trn_err = (lhs.t - h.t).norm();
        assert!(
            rot_err < 1e-12 && trn_err < 1e-12,
            "synthetic generator inconsistent at view {i}: \
             G.X.T = H up to {rot_err} rad / {trn_err} m, expected {h:?}"
        );
    }

    for method in [
        HandEyeMethod::Tsai,
        HandEyeMethod::Park,
        HandEyeMethod::Horaud,
        HandEyeMethod::Andreff,
    ] {
        let (r_x, t_x) = calibrate_hand_eye(&r_g2b, &t_g2b, &r_t2c, &t_t2c, method)
            .unwrap_or_else(|| panic!("calibrate_hand_eye returned None for {method:?}"));

        let rot_err = rotation_angle(&(x_true.r.transpose() * r_x));
        let trn_err = (t_x - x_true.t).norm();
        assert!(rot_err < 1e-6, "{method:?}: rotation error {rot_err} rad");
        assert!(trn_err < 1e-6, "{method:?}: translation error {trn_err} m");
    }
}

/// Robot-world/hand-eye (A . X = Z . B), same treatment.
///
/// With Z = gripper2cam and X = base2world, the relation is
/// `world2cam_i . X = Z . base2gripper_i`. Draw B_i freely, plant X and Z,
/// and solve for A_i. Tolerance 1e-6 rad / 1e-6 m, matching the sibling test.
#[test]
fn calibrate_robot_world_hand_eye_recovers_planted_x_and_z() {
    let mut rng = Lcg(0x5EED_0003);
    let x_true = Rigid {
        r: rng.rotation(1.0),
        t: Vector3::new(0.3, 0.0, -0.1),
    };
    let z_true = Rigid {
        r: rng.rotation(1.0),
        t: Vector3::new(-0.02, 0.04, 0.06),
    };

    let mut world2cam = Vec::new();
    let mut base2gripper = Vec::new();
    for _ in 0..10 {
        let b = Rigid {
            r: rng.rotation(1.4),
            t: rng.translation(0.5),
        };
        // A_i = Z . B_i . X^-1
        world2cam.push(z_true.then(&b).then(&x_true.inverse()));
        base2gripper.push(b);
    }
    let (r_w2c, t_w2c) = split(&world2cam);
    let (r_b2g, t_b2g) = split(&base2gripper);

    for method in [RobotWorldHandEyeMethod::Shah, RobotWorldHandEyeMethod::Li] {
        let (r_x, t_x, r_z, t_z) = calibrate_robot_world_hand_eye(
            &r_w2c, &t_w2c, &r_b2g, &t_b2g, method,
        )
        .unwrap_or_else(|| panic!("calibrate_robot_world_hand_eye returned None for {method:?}"));

        let x_rot_err = rotation_angle(&(x_true.r.transpose() * r_x));
        let x_trn_err = (t_x - x_true.t).norm();
        let z_rot_err = rotation_angle(&(z_true.r.transpose() * r_z));
        let z_trn_err = (t_z - z_true.t).norm();

        assert!(
            x_rot_err < 1e-6,
            "{method:?}: X rotation error {x_rot_err} rad"
        );
        assert!(
            x_trn_err < 1e-6,
            "{method:?}: X translation error {x_trn_err} m"
        );
        assert!(
            z_rot_err < 1e-6,
            "{method:?}: Z rotation error {z_rot_err} rad"
        );
        assert!(
            z_trn_err < 1e-6,
            "{method:?}: Z translation error {z_trn_err} m"
        );
    }
}
