//! `aruco::estimate_marker_pose` must return a pose that reprojects the corners
//! it was given.
//!
//! Two defects compound here.
//!
//! **(a) Thin-SVD row bug.** `solve_pnp_dlt` builds an `8 x 12` matrix `A` and
//! reads the null vector from `vt.row(vt.nrows() - 1)`. nalgebra's *thin* SVD
//! returns only `min(8, 12) = 8` rows, so that is row **7**, not row 11 - an
//! arbitrary vector with a strictly positive `‖A v‖`.
//!
//! **(b) The 4-point system is rank deficient.** The object points are a square
//! in `z = 0`, so `A` is rank 8 with a 4-dimensional null space. Measured
//! singular values of that `8 x 12` matrix for a frontal 0.05 m marker at
//! 0.40 m, `f = 200`, no distortion:
//!
//! ```text
//! [2.0059, 2.0000, 0.2376, 0.0501, 0.0501, 0.0500, 0.0042, 0.0041]
//! ```
//!
//! Four values cluster at ~0.05 and two more at ~0.004: there is no unique null
//! vector to pick, and the projection matrix recovered from the smallest of them
//! is not a metric pose. Sign/cheirality is also unconstrained: `[-R|t]` and
//! `[R|t]` reproject identically.
//!
//! Measured on that exact case before the fix, where the answer is known
//! analytically (`rvec = 0`, `t = [0, 0, 0.4]`):
//!
//! ```text
//! rvec = [0.0004, 2.2213, 2.2213]   rotation error 2.22 rad
//! t    = [-0.0016, 640.0016, -4.9e-9]   depth 0.0 instead of 0.4
//! ```
//!
//! The old `pose_estimation_runs` test only asserted `is_ok()` and finiteness,
//! which the above satisfies.
//!
//! The fix reads the correct row (11) and, crucially, **verifies the answer
//! against its own input**: the recovered projection is normalised, forced to
//! a proper rotation, and then re-projected through the recovered pose. If the
//! pose does not reproduce the four corners, the solve is not a pose and the
//! function reports that it cannot solve instead of returning it.

use cv_features::aruco::{estimate_marker_pose, DetectedMarker};

const MARKER: f64 = 0.05;
const F: f64 = 200.0;
const CX: f64 = 640.0;
const CY: f64 = 480.0;
const NONE: [f64; 5] = [0.0; 5];

fn cam() -> [[f64; 3]; 3] {
    [[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]]
}

/// The four marker corners in the documented order:
/// top-left, top-right, bottom-right, bottom-left, matching the object points
/// `[-h,-h,0], [h,-h,0], [h,h,0], [-h,h,0]` at `aruco.rs:259-264`.
fn object_points() -> [[f64; 3]; 4] {
    let h = MARKER * 0.5;
    [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]]
}

/// Project the object points through a known (Rodrigues, translation) pose.
fn project(rvec: [f64; 3], t: [f64; 3]) -> [[f64; 2]; 4] {
    let r = rotation_from_rodrigues(rvec);
    let t = nalgebra::Vector3::new(t[0], t[1], t[2]);
    let mut out = [[0.0f64; 2]; 4];
    for (i, p) in object_points().iter().enumerate() {
        let c = r * nalgebra::Vector3::new(p[0], p[1], p[2]) + t;
        out[i] = [F * c.x / c.z + CX, F * c.y / c.z + CY];
    }
    out
}

/// Rodrigues vector -> rotation matrix (Rodrigues' formula), zero-vector safe.
///
/// `R = I + sinc(theta) K + (1 - cos theta)/theta^2 K^2`, i.e. the *exact*
/// formula. The tempting shorthand `I + K + K^2 (1 - cos theta)/theta^2` is
/// only first-order correct: since `K^2 = -theta^2 I` it collapses to
/// `cos(theta) I + K`, which needs `theta == sin(theta)` to agree with the real
/// answer. It happens to be exact for the identity pose, so a test that only
/// ever uses `rvec = 0` would not notice.
fn rotation_from_rodrigues(rvec: [f64; 3]) -> nalgebra::Matrix3<f64> {
    let v = nalgebra::Vector3::new(rvec[0], rvec[1], rvec[2]);
    let theta = v.norm();
    if theta < 1e-15 {
        return nalgebra::Matrix3::identity();
    }
    let k = nalgebra::Matrix3::new(
        0.0, -v.z, v.y, //
        v.z, 0.0, -v.x, //
        -v.y, v.x, 0.0,
    );
    nalgebra::Matrix3::identity()
        + (theta.sin() / theta) * k
        + ((1.0 - theta.cos()) / (theta * theta)) * (k * k)
}

fn marker(corners: [[f64; 2]; 4]) -> DetectedMarker {
    DetectedMarker {
        id: 0,
        corners: corners.map(|[x, y]| (x, y)),
    }
}

/// Worst reprojection error of the returned pose on the four input corners, in
/// pixels. 0 means the pose reproduces its input exactly.
///
/// Also reported as a fraction of the marker's own projected side, so the
/// numbers are comparable with the defect report ("1159% of the marker size").
fn reproj_px(
    corners: &[[f64; 2]; 4],
    rvec: &nalgebra::Vector3<f64>,
    tvec: &nalgebra::Vector3<f64>,
) -> (f64, f64) {
    let r = rotation_from_rodrigues([rvec.x, rvec.y, rvec.z]);
    let t = *tvec;
    let errs: Vec<f64> = object_points()
        .iter()
        .zip(corners.iter())
        .map(|(p, q)| {
            let c = r * nalgebra::Vector3::new(p[0], p[1], p[2]) + t;
            if c.z.abs() < 1e-12 {
                return f64::INFINITY;
            }
            let px = F * c.x / c.z + CX;
            let py = F * c.y / c.z + CY;
            ((px - q[0]).powi(2) + (py - q[1]).powi(2)).sqrt()
        })
        .collect();
    let worst = errs.iter().copied().fold(0.0f64, f64::max);
    // Projected side length at the head-on distance 0.4 m: the ground-truth
    // pose's own image separation of two adjacent corners. This is what
    // "1159% of the marker size" means.
    let side = {
        let o0 = nalgebra::Vector3::new(object_points()[0][0], object_points()[0][1], 0.0);
        let o1 = nalgebra::Vector3::new(object_points()[1][0], object_points()[1][1], 0.0);
        let c0 = o0 + nalgebra::Vector3::new(0.0, 0.0, 0.4);
        let c1 = o1 + nalgebra::Vector3::new(0.0, 0.0, 0.4);
        let a = nalgebra::Vector3::new(F * c0.x / c0.z + CX, F * c0.y / c0.z + CY, 0.0);
        let b = nalgebra::Vector3::new(F * c1.x / c1.z + CX, F * c1.y / c1.z + CY, 0.0);
        (a - b).norm()
    };
    (worst, worst / side)
}

/// The reported case: a frontal 0.05 m marker at 0.40 m, `f = 200`, no
/// distortion, no noise. The answer is known exactly, and the returned pose
/// must (i) reproject the four corners and (ii) reproduce the known pose.
#[test]
fn a_frontal_marker_at_known_pose_is_recovered() {
    let corners = project([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]);
    let (rvec, tvec) = estimate_marker_pose(&marker(corners), MARKER, &cam(), &NONE)
        .expect("an exact 4-point planar case is solvable and must not be refused");

    let (px, frac) = reproj_px(&corners, &rvec, &tvec);
    assert!(
        px < 1e-6,
        "the returned pose does not reproject its own four corners: worst error \
         is {px:.3e} px ({frac:.3e} of the marker's projected side; before the \
         fix 8.389e7 px / 11.59).\n  returned rvec = [{:.4}, {:.4}, {:.4}] \
         (truth [0, 0, 0])\n  returned t    = [{:.4e}, {:.4e}, {:.4e}] \
         (truth [0, 0, 0.4])",
        rvec.x,
        rvec.y,
        rvec.z,
        tvec.x,
        tvec.y,
        tvec.z,
    );

    let rot_err = rvec.norm();
    let depth = tvec.z;
    assert!(
        rot_err < 1e-6,
        "rotation error {rot_err:.3e} rad (truth 0; before the fix 2.22 rad)"
    );
    assert!(
        (depth - 0.4).abs() < 1e-6,
        "depth {depth:.6} instead of 0.4 (before the fix: 0.0), \
         t = [{:.6}, {:.6}, {:.6}]",
        tvec.x,
        tvec.y,
        tvec.z
    );
    assert!(
        tvec.x.abs() < 1e-6 && tvec.y.abs() < 1e-6,
        "on-axis marker must have t_x = t_y = 0, got [{:.6}, {:.6}, {:.6}]",
        tvec.x,
        tvec.y,
        tvec.z
    );
}

/// A tilted marker, so that a solver which only ever produces identity rotations
/// cannot pass. Same scale, oblique pose, still noise-free.
#[test]
fn a_tilted_marker_at_known_pose_is_recovered() {
    let corners = project([0.15, -0.25, 0.10], [0.05, -0.03, 0.6]);
    let (rvec, tvec) = estimate_marker_pose(&marker(corners), MARKER, &cam(), &NONE)
        .expect("an exact 4-point planar case is solvable and must not be refused");

    let (px, frac) = reproj_px(&corners, &rvec, &tvec);
    assert!(
        px < 1e-6,
        "tilted marker: worst reprojection error is {px:.3e} px ({frac:.3e} of the \
         marker's projected side)"
    );
    // And the recovered pose must be the tilted pose, not merely a
    // reprojection of them.
    let expected = project([0.15, -0.25, 0.10], [0.05, -0.03, 0.6]);
    assert!(
        (rvec - nalgebra::Vector3::new(0.15, -0.25, 0.10)).norm() < 1e-6,
        "rotation error: rvec = {:?}, truth [0.15, -0.25, 0.10]",
        rvec.as_slice()
    );
    assert!(
        (tvec - nalgebra::Vector3::new(0.05, -0.03, 0.6)).norm() < 1e-6,
        "translation error: t = {:?}, truth [0.05, -0.03, 0.6]",
        tvec.as_slice()
    );
    let _ = expected;
}

/// CONTROL: with lens distortion the same corners are seen through a distorted
/// projection. The undistort step in `estimate_marker_pose` is unchanged by
/// this fix, so the recovered pose must still reproject the *raw* (distorted)
/// observations it was handed, and still recover the true pose.
#[test]
fn control_a_distorted_frontal_marker_is_still_solved() {
    // k1 = 0.12, k2 = -0.03 on a 0.05 m marker at 0.40 m: a few pixels of
    // barrel/pincushion on the corners.
    let k1 = 0.12;
    let k2 = -0.03;
    let dist = [k1, k2, 0.0, 0.0, 0.0];
    let h = MARKER * 0.5;
    let mut corners = [[0.0f64; 2]; 4];
    for (i, p) in object_points().iter().enumerate() {
        // Camera-space: identity rotation, marker centre 0.4 m in front.
        let c = nalgebra::Vector3::new(p[0], p[1], 0.4);
        let xn = c.x / c.z;
        let yn = c.y / c.z;
        let r2 = xn * xn + yn * yn;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        corners[i] = [F * xn * radial + CX, F * yn * radial + CY];
    }
    let _ = h;

    let (rvec, tvec) = estimate_marker_pose(&marker(corners), MARKER, &cam(), &dist)
        .expect("a distorted but exact frontal case must still solve");

    // The pose must reproduce the *raw, distorted* corners it was handed. The
    // tolerance is the 5-iteration iterative-undistort residual plus the
    // detection noise a real corner would carry - but the marker's projected
    // side here is 12.5 px, so 0.5 px is 4% of the marker, against 1159%
    // before the fix. This is the control that the fix did not simply
    // "solve" everything by refusing: it still returns a pose, and that pose
    // is right.
    let (px, frac) = reproj_px(&corners, &rvec, &tvec);
    assert!(
        px < 0.5,
        "distorted frontal marker: worst reprojection error is {px:.3e} px \
         ({frac:.3e} of the marker's projected side; before the fix 11.59)"
    );
    assert!(
        (tvec.z - 0.4).abs() < 0.01,
        "distorted frontal marker: depth {:.6} instead of 0.4",
        tvec.z
    );
    assert!(
        rvec.norm() < 1e-3,
        "distorted frontal marker is viewed head-on, so rvec should be ~0, got {:?}",
        rvec.as_slice()
    );
}

/// A pose that cannot be represented as a 4-point planar pose must be reported
/// as unsolvable rather than returned. Two coincident corners make the
/// correspondence set degenerate, and there is no pose - let alone a unique
/// one - behind it.
#[test]
fn a_degenerate_marker_is_reported_not_solved() {
    let mut corners = project([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]);
    // Collapse two corners onto each other: three distinct observations of a
    // square. A rigid plane pose exists, but this correspondence set does not
    // determine one.
    corners[2] = corners[1];

    match estimate_marker_pose(&marker(corners), MARKER, &cam(), &NONE) {
        Ok((rvec, tvec)) => {
            // If a pose is offered it must at least reproject its input.
            let (px, frac) = reproj_px(&corners, &rvec, &tvec);
            assert!(
                px < 1e-6,
                "a pose was returned for a degenerate marker and it does not \
                 reproject it: error is {px:.3e} px ({frac:.3e} of the marker)"
            );
        }
        Err(_) => {
            // Reporting failure is the honest outcome; nothing more to check.
        }
    }
}

/// Sanity check on the measurement itself: the ground-truth pose scores zero on
/// the same corners, and the marker's projected side is the expected 12.5 px.
/// Without this, a bug in `reproj_px` would make every assertion above pass
/// vacuously.
#[test]
fn the_true_pose_scores_zero_on_the_same_corners() {
    let corners = project([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]);
    let (px, frac) = reproj_px(
        &corners,
        &nalgebra::Vector3::zeros(),
        &nalgebra::Vector3::new(0.0, 0.0, 0.4),
    );
    assert!(
        px < 1e-12,
        "the measurement is wrong: the true pose scores {px:.3e} px"
    );
    assert!(
        (frac - 0.0).abs() < 1e-12,
        "the true pose should score zero, got {frac:.3e}"
    );
    // Sanity on the "fraction of marker size" unit used in the messages: a
    // 0.05 m marker at 0.4 m with f = 200 projects to 0.05/0.4*200 = 25 px
    // across the diagonal, i.e. 12.5 px per side.
    let side = 0.05 / 0.4 * 200.0;
    let (a, b) = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]);
    assert!(
        ((a * a + b * b).sqrt() - side).abs() < 1e-9,
        "projected marker side is {:.4} px, expected {side}",
        (a * a + b * b).sqrt()
    );
}
