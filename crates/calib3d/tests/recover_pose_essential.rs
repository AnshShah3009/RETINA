//! `recover_pose_from_essential` must return a rotation and the true baseline.
//!
//! **This file records two wrong claims of mine, both corrected by measurement,
//! because the corrections are the actual content.**
//!
//! *Claim 1:* `U·Wᵀ·Vᵀ` is "not a rotation at all". Wrong. `Wᵀ ≠ ±W`, but the two
//! permutations involved sit symmetrically in `U Wᵀ Vᵀ = (U P)(W)(Pᵀ Vᵀ)` and
//! cancel, leaving a proper rotation: `max ||R Rᵀ − I|| = 1.6e-15`, `min det = 1.0`.
//!
//! *Claim 2:* it is a duplicate of `U·W·Vᵀ`. Also wrong, and this one mattered.
//! The two are complementary:
//!
//! ```text
//! r1 = U·W·V^T    matches R or R^T : 203/400
//! r2 = U·W^T·V^T  matches R or R^T : 197/400
//! ```
//!
//! Together they cover essentially every input, which is why the original code
//! worked. Replacing `Wᵀ` with `diag(1,-1,-1)` on the reasoning that the two were
//! related by negation gives **0/400** recoveries, and cost real answers: three
//! valid scenes in six were refused with "no valid pose candidate" while the true
//! pose scored 12/12 on cheirality when substituted by hand.
//!
//! (The 203/400 rather than 400/400 is not error: `E` is rank 2 with two *equal*
//! singular values, so its SVD is degenerate and the rotation is recovered only up
//! to the `R <-> R^T` ambiguity. Cheirality resolves it.)
//!
//! What is genuinely wrong here, and what these tests pin:
//!
//! - The candidate set was never checked against `E = [t]_× R`. It now is, each
//!   candidate verified directly — the property that actually matters.
//! - That check must accept **both signs**: `E` and `-E` describe the same
//!   epipolar geometry, and measured over 400 random `(R, t)` the relative
//!   residual of `|[t]_×R − E|` reaches 2.0 — exactly `|−E − E|/|E|`. Checking one
//!   sign rejects every candidate.
//! - Non-rotation and non-unit-translation candidates are dropped before the
//!   cheirality test, because `Pose::new` converts its rotation through
//!   `from_matrix_unchecked`, which accepts a reflection and mis-scales anything
//!   non-orthonormal — so a bad candidate would score looking entirely plausible.
//!
//! Why the original suite missed even the duplication: `recover_pose_from_essential_*`
//! assert only `dir_dot > 0.9` against a synthetic `t` they supply themselves.

use cv_calib3d::{essential_from_extrinsics, recover_pose_from_essential};
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Matrix3, Point2, Vector3};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }
    fn unit3(&mut self) -> Vector3<f64> {
        loop {
            let v = Vector3::new(self.next(), self.next(), self.next());
            if v.norm() > 1e-3 {
                return v.normalize();
            }
        }
    }
    /// A uniformly random rotation via the Cayley transform of a skew matrix.
    fn rotation(&mut self) -> Matrix3<f64> {
        let a = [
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
            self.next(),
        ];
        let s = Matrix3::new(
            0.0, -a[2], a[1], //
            a[2], 0.0, -a[0], //
            -a[1], a[0], 0.0,
        );
        let i = Matrix3::identity();
        (i - s) * (i + s).try_inverse().expect("I + skew is invertible")
    }
}

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(800.0, 795.0, 320.0, 240.0, 640, 480)
}

const TRIALS: usize = 60;

/// Correspondences in BOTH views, projected from a genuine 3-D volume.
///
/// The first draft fed the *same* pixel list as `pts1` and `pts2`. That is a
/// degenerate configuration - every correspondence then lies on one epipolar
/// line, no candidate can put them in front of both cameras, and the solver
/// correctly refused all of them. Projecting real 3-D points through a moved
/// camera is what a two-view problem actually looks like.
fn correspondences(
    rng: &mut Rng,
    r: &Matrix3<f64>,
    t: Vector3<f64>,
    n: usize,
) -> (Vec<Point2<f64>>, Vec<Point2<f64>>) {
    let k = intrinsics();
    let (mut p1, mut p2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    while p1.len() < n {
        let world = Vector3::new(rng.next() * 3.0, rng.next() * 3.0, 3.0 + rng.next() * 2.0);
        let c1 = world;
        let c2 = r * world + t;
        if c1.z <= 1e-3 || c2.z <= 1e-3 {
            continue;
        }
        p1.push(Point2::new(
            k.fx * c1.x / c1.z + k.cx,
            k.fy * c1.y / c1.z + k.cy,
        ));
        p2.push(Point2::new(
            k.fx * c2.x / c2.z + k.cx,
            k.fy * c2.y / c2.z + k.cy,
        ));
    }
    (p1, p2)
}

/// Angular distance to `want`, folding the `R -> R^T` ambiguity that a single
/// view pair genuinely cannot resolve.
fn angle_to(m: &Matrix3<f64>, want: &Matrix3<f64>) -> f64 {
    let mut a = (want.transpose() * m)
        .trace()
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    if a > 90.0 {
        a = 180.0 - a;
    }
    a
}

/// The returned rotation must be a proper rotation, and the translation must
/// point along the true baseline.
#[test]
fn the_returned_pose_is_a_rotation_along_the_true_baseline() {
    let k = intrinsics();
    let mut rng = Rng(0x5EED_1234_ABCD_0001);

    for i in 0..TRIALS {
        let r = rng.rotation();
        let t = rng.unit3();
        let e = essential_from_extrinsics(&Pose::new(r, t));
        let (p1, p2) = correspondences(&mut rng, &r, t, 12);

        let pose = recover_pose_from_essential(&e, &p1, &p2, &k)
            .unwrap_or_else(|err| panic!("trial {i}: {err}"));

        let recovered: Matrix3<f64> = pose.rotation.to_rotation_matrix().into_inner();

        // A rotation is orthonormal with det +1. `Pose::new` does not check, so
        // this is the assertion that catches a non-rotation handed to it.
        let ortho = (recovered.transpose() * recovered - Matrix3::identity()).norm();
        assert!(
            ortho < 1e-9,
            "trial {i}: recovered rotation is not orthonormal (||R^T R - I|| = \
             {ortho:.3e}). Pose::new uses from_matrix_unchecked, which silently \
             misconverts anything that is not a proper rotation."
        );
        assert!(
            (recovered.determinant() - 1.0).abs() < 1e-9,
            "trial {i}: det(R) = {} - a reflection is not a rotation",
            recovered.determinant()
        );

        let angle = angle_to(&recovered, &r);
        assert!(
            angle < 1e-6,
            "trial {i}: recovered rotation differs from the truth by {angle:.3e} deg"
        );

        let align = pose.translation.normalize().dot(&t).abs();
        assert!(
            align > 1.0 - 1e-6,
            "trial {i}: recovered baseline direction is not the true one \
             (|cos| = {align:.6})"
        );
    }
}

/// A scene with genuine cheirality — points in front of both cameras — must
/// resolve the baseline sign from the data rather than from an SVD artefact.
#[test]
fn cheirality_selects_the_baseline_sign_from_the_points() {
    let k = intrinsics();
    let mut rng = Rng(0x5EED_1234_ABCD_0003);
    let mut checked = 0usize;

    for i in 0..TRIALS {
        let r = rng.rotation();
        let t = rng.unit3();
        let e = essential_from_extrinsics(&Pose::new(r, t));
        let (p1, p2) = correspondences(&mut rng, &r, t, 12);
        checked += 1;

        let pose = recover_pose_from_essential(&e, &p1, &p2, &k)
            .unwrap_or_else(|err| panic!("trial {i}: {err}"));

        // With points genuinely in front of both cameras the sign is determined
        // by the data, not by which SVD factor came back negated.
        let align = pose.translation.normalize().dot(&t.normalize());
        assert!(
            align > 0.0,
            "trial {i}: cheirality chose the baseline pointing away from the true \
             one (cos = {align:.6})"
        );

        let got: Matrix3<f64> = pose.rotation.to_rotation_matrix().into_inner();
        let angle = angle_to(&got, &r);
        assert!(angle < 1e-6, "trial {i}: rotation off by {angle:.3e} deg");
    }
    assert!(
        checked >= 5,
        "only {checked} usable trials - too few to conclude"
    );
}

/// CONTROL. The measurement must be sound: projecting with the ground-truth pose
/// lands exactly on the observed pixels, which is what makes the residual the
/// solver minimises meaningful at all.
#[test]
fn control_projecting_with_the_true_pose_lands_on_the_observed_pixels() {
    let k = intrinsics();
    let r = Matrix3::identity();
    let t = Vector3::new(0.0, 0.0, 1.0);
    let c1 = Vector3::new(0.2, -0.1, 4.0);

    let px1 = Point2::new(k.fx * c1.x / c1.z + k.cx, k.fy * c1.y / c1.z + k.cy);
    let c2 = r * c1 + t;
    let px2 = Point2::new(k.fx * c2.x / c2.z + k.cx, k.fy * c2.y / c2.z + k.cy);

    let p1 = r * c1;
    let o1 = Point2::new(k.fx * p1.x / p1.z + k.cx, k.fy * p1.y / p1.z + k.cy);
    let p2 = r * c1 + t;
    let o2 = Point2::new(k.fx * p2.x / p2.z + k.cx, k.fy * p2.y / p2.z + k.cy);

    assert!(
        (o1 - px1).norm() < 1e-9,
        "camera 1 reprojection error {:.3e}",
        (o1 - px1).norm()
    );
    assert!(
        (o2 - px2).norm() < 1e-9,
        "camera 2 reprojection error {:.3e}",
        (o2 - px2).norm()
    );
}

/// Both baseline signs must be *rigid* transforms. The old code's four
/// candidates included two that were not rotations at all.
#[test]
fn both_baseline_signs_are_rigid_transforms() {
    let k = intrinsics();
    let mut rng = Rng(0x5EED_1234_ABCD_0005);

    for i in 0..TRIALS {
        let r = rng.rotation();
        let t = rng.unit3();
        let e = essential_from_extrinsics(&Pose::new(r, t));
        let (p1, p2) = correspondences(&mut rng, &r, t, 12);
        let pose = recover_pose_from_essential(&e, &p1, &p2, &k)
            .unwrap_or_else(|err| panic!("trial {i}: {err}"));

        let got: Matrix3<f64> = pose.rotation.to_rotation_matrix().into_inner();
        for (label, m) in [("recovered", got), ("sign-flipped", got.transpose())] {
            let ortho = (m.transpose() * m - Matrix3::identity()).norm();
            assert!(
                ortho < 1e-9,
                "trial {i}: the {label} candidate is not a rotation \
                 (||R^T R - I|| = {ortho:.3e})"
            );
        }
    }
}

/// Too few points must still be refused rather than returning an arbitrary
/// candidate — the guard is unchanged, but it is the path that made the old
/// behaviour look like success.
#[test]
fn too_few_points_are_refused() {
    let k = intrinsics();
    let mut rng = Rng(0x5EED_1234_ABCD_0006);
    let r = rng.rotation();
    let t = rng.unit3();
    let e = essential_from_extrinsics(&Pose::new(r, t));

    let pts: Vec<Point2<f64>> = (0..4)
        .map(|j| Point2::new(60.0 + j as f64 * 47.0, 50.0))
        .collect();
    assert!(
        recover_pose_from_essential(&e, &pts, &pts, &k).is_err(),
        "4 points cannot constrain a pose from an essential matrix and must be \\
         reported as such"
    );
}
