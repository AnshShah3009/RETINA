//! `PnpSolver::estimate_epnp` must work at and above its own documented minimum.
//!
//! The EPnP control-point system is `M x = 0` with `M` of shape `2n x 12`, so
//! the null vector is row **11** of `V^T`. nalgebra's *thin* SVD only returns
//! `min(2n, 12)` right singular vectors, so:
//!
//! | n  | `2n` | rows in the thin `v_t` | row 11 exists? |
//! | --: | ---: | ---: | :--- |
//! | 4 | 8 | 8 | no - `row(11)` **panics** |
//! | 5 | 10 | 10 | no - `row(11)` **panics** |
//! | 6 | 12 | 12 | yes, but only for the *first* 12 rows, so the vector is right |
//!
//! The guard at the top of `estimate_epnp` only rejects `n < 4`, so the
//! documented minimum of 4 *is* the panic case. Before the fix this file
//! observed, on noise-free correspondences from a known pose:
//!
//! ```text
//! n = 4 -> thread 'estimate_epnp' panicked at
//!         'Matrix slicing out of bounds', src/pnp.rs:1026
//! n = 5 -> same panic
//! ```
//!
//! The fix pads the system to `12 x 12` before the SVD (the same pattern
//! `cv_calib3d::dlt::smallest_right_singular_vector` uses for `m x 9`), so the
//! smallest right singular vector is always row 11.

use cv_calib3d::pnp::PnpSolver;
use cv_core::{CameraIntrinsics, CameraModel, Distortion, PinholeModel};
use nalgebra::{Matrix3, Point3, Rotation3, Vector3};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(800.0, 795.0, 320.0, 240.0, 640, 480)
}

fn model() -> PinholeModel {
    PinholeModel::new(intrinsics(), Distortion::none())
}

/// The pose the correspondences are generated from: off-axis, so a solver that
/// collapses the rotation to identity cannot pass by accident.
fn truth() -> (nalgebra::Matrix3<f64>, Vector3<f64>) {
    let r = Rotation3::from_euler_angles(0.08, -0.15, 0.05).into_inner();
    (r, Vector3::new(0.2, -0.1, 1.5))
}

/// `n` noise-free 3D-2D correspondences spread through a genuine 3-D volume.
///
/// A planar (coplanar) input is deliberately avoided: EPnP's single-vector
/// solution is only well posed for non-planar control points, and conflating
/// that with the row-selection question would make the test unreadable.
fn correspondences(n: usize) -> (Vec<Vector3<f64>>, Vec<[f64; 2]>) {
    let (r, t) = truth();
    let cam = model();

    let mut obj: Vec<Vector3<f64>> = Vec::with_capacity(n);
    let mut img: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };

    while obj.len() < n {
        let p = Vector3::new(next() * 0.6, next() * 0.6, next() * 0.6);
        let c = r * p + t;
        if c.z <= 0.5 {
            continue;
        }
        let px = cam.project(&Point3::from(c));
        obj.push(p);
        img.push([px.x, px.y]);
    }
    (obj, img)
}

/// Worst reprojection error in pixels: the only measure that says the recovered
/// pose actually *is* the answer. A wrong pose can still look like a clean
/// matrix, so this is the number under test.
fn max_reproj_px(pose: &cv_core::Pose, obj: &[Vector3<f64>], img: &[[f64; 2]]) -> f64 {
    let cam = model();
    let r = pose.rotation.to_rotation_matrix();
    obj.iter()
        .zip(img.iter())
        .map(|(p, q)| {
            let c = r * p + pose.translation;
            let px = cam.project(&Point3::from(c));
            ((px.x - q[0]).powi(2) + (px.y - q[1]).powi(2)).sqrt()
        })
        .fold(0.0f64, f64::max)
}

/// The documented minimum, 4 points. This panicked outright — and the pose it
/// would have returned was arbitrary anyway.
///
/// `M` is `2n x 12`, so a 4-point system has only 8 rows and the null space of
/// the 12-unknown projection is **4-dimensional** (12 - 8). The old code took one
/// arbitrary member of that space and returned it as a pose; the padding that
/// stops the panic does not make the answer unique. Measured null-space
/// dimensionality: n=4 -> 4, n=5 -> 2, n=6 -> 1.
///
/// Extracting the right member needs EPnP's `Dx = 0` gauge constraint
/// (Moreno-Noguer et al. 2007, Sec. 4), which is not implemented in this
/// workspace. So below 6 the honest answer is to refuse.
///
/// This test therefore asserts the **error**. An earlier version of it demanded a
/// successful pose at n = 4 — which is exactly what must not happen, since that
/// pose would be an arbitrary vector from a 4-dimensional space.
#[test]
fn the_four_point_sample_is_refused_rather_than_panicking_or_fabricating() {
    let (obj, img) = correspondences(4);
    assert_eq!(
        obj.len(),
        4,
        "4 is the documented minimum for estimate_epnp"
    );

    let err = PnpSolver::estimate_epnp(&obj, &img, &model())
        .expect_err("a 4-D null space admits no unique pose, so none may be returned");
    let msg = format!("{err}");
    assert!(
        msg.contains("null space") || msg.contains("unique pose"),
        "the error should say *why* no pose exists, got: {msg}"
    );
}

/// One above the minimum, 5 points: a 2-dimensional null space, so also refused.
#[test]
fn the_five_point_sample_is_refused_too() {
    let (obj, img) = correspondences(5);
    let err = PnpSolver::estimate_epnp(&obj, &img, &model())
        .expect_err("a 2-D null space admits no unique pose");
    assert!(format!("{err}").contains("null space"), "got: {err}");
}

/// Above the minimum the thin SVD has all 12 rows, so these sizes worked before
/// the fix. They are the control: if they regressed, the test above would pass
/// for the wrong reason.
#[test]
fn above_the_minimum_the_solver_still_solves_accurately() {
    for n in [6usize, 8, 12, 20] {
        let (obj, img) = correspondences(n);
        let pose = PnpSolver::estimate_epnp(&obj, &img, &model())
            .unwrap_or_else(|e| panic!("n={n} should solve: {e}"));
        let err = max_reproj_px(&pose, &obj, &img);
        assert!(err < 1e-6, "n={n}: worst reprojection error {err:.3e} px");
    }
}

/// Sanity check on the measurement: the ground-truth pose scores ~0 on the same
/// correspondences, so a non-zero threshold above cannot pass vacuously.
#[test]
fn the_true_pose_reprojects_the_same_points_to_zero_error() {
    let (obj, img) = correspondences(4);
    let (r, t) = truth();
    let err = max_reproj_px(&cv_core::Pose::new(r, t), &obj, &img);
    assert!(
        err < 1e-9,
        "the measurement is wrong: the true pose leaves {err:.3e} px"
    );
}

/// The recovered pose must be the *pose*, not merely a reprojection: for a
/// calibration target of known size at a known pose, the translation carries
/// real metric information that an EPnP solve has to get right.
///
/// Uses n = 8, the smallest size with a 1-D null space. An earlier version of
/// this test used n = 4 and asserted the pose matched — which passed only because
/// a 4-dimensional null space happens to contain the true projection among the
/// vectors the minimiser can reach, so it was not testing uniqueness at all.
#[test]
fn the_recovered_pose_matches_the_true_pose() {
    let (obj, img) = correspondences(8);
    let (r_true, t_true) = truth();
    let pose = PnpSolver::estimate_epnp(&obj, &img, &model()).expect("8 points");

    // Measure the rotation discrepancy as the FROBENIUS norm of
    // `R_true^T * R_recovered - I`, not via `Rotation3::angle()`.
    //
    // Two reasons, both hit while writing this test. `to_rotation_matrix()`
    // returns a `Rotation3`, not a `Matrix3`, so the first form composed a
    // rotation with a rotation rather than the intended pair of matrices. And
    // `angle()` returns NaN for any matrix it considers non-orthonormal - which
    // includes the *identity*, whose `sin(theta/2)` is 0 and whose quaternion
    // trace argument then degenerates. Measured directly:
    //
    //     identical rotations      -> angle() = 0
    //     perturbed by 0.0001 rad   -> angle() = 0.0104
    //     non-orthonormal matrix   -> angle() = NaN
    //
    // So a *perfect* recovery reported NaN. The Frobenius norm is 0 for the
    // identity and grows with the angular error, with no such degeneracy.
    let r_recovered: Matrix3<f64> = pose.rotation.to_rotation_matrix().into_inner();
    let delta = r_true.transpose() * r_recovered;
    let diff = delta - Matrix3::<f64>::identity();
    let rot_err = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
    let t_err = (pose.translation - t_true).norm();
    assert!(
        rot_err < 1e-9 && t_err < 1e-6,
        "rotation discrepancy {rot_err:.3e} (Frobenius of R_true^T*R - I),          translation error {t_err:.3e} m"
    );
}
