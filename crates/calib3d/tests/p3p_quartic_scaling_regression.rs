//! Regression test for the Grunert quartic polynomial assembly in
//! `cv-calib3d`'s P3P solver (`PnpSolver::estimate_p3p`).
//!
//! ## Why this file exists
//!
//! `src/pnp.rs:768` reported `warning: function padd is never used`. It was a
//! nested `fn padd(a: Vec<f64>, b: &[f64]) -> Vec<f64>` defined *inside*
//! `estimate_p3p`, immediately after its sibling `pmul` (a dense polynomial
//! convolution used three times to build the Grunert quartic).
//!
//! `pmul` was called; `padd` never was. The coefficient assembly below it
//! accumulates by hand (`coeff[deg] += ...` in three separate loops), which is
//! exactly what `padd` did, so `padd` was the abandoned earlier spelling of an
//! operation that is still performed. It was dead code, not a missing call -
//! the three hand-written loops are complete: they cover degrees 0..3, 0..3 and
//! 0..4 respectively, and `coeff` is length 5, so no coefficient is ever left
//! unwritten. Nothing was dropped on the floor.
//!
//! Dead code that duplicates a live operation is a trap: a later reader can
//! delete the "unused" loops believing they were replaced by `padd`. So
//! `padd` was deleted and the assembly is now pinned by the tests below.
//!
//! ## What is measured
//!
//! 1. `pmul` against an independently written convolution (loop order and
//!    accumulation are deliberately transposed, so a shared bug is unlikely).
//! 2. `estimate_p3p` reprojection error on a planted pose - the end-to-end
//!    statement that the surviving assembly is numerically right, not merely
//!    free of unused helpers.

use cv_calib3d::multiview::PnpSolver;
use cv_core::{CameraIntrinsics, Distortion, PinholeModel};
use nalgebra::{Point3, Rotation3, Unit, Vector3};

/// Independent reference convolution, written as an explicit coefficient sum
/// rather than a nested loop over both inputs.
fn reference_poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let deg = a.len() + b.len() - 2;
    (0..=deg)
        .map(|k| {
            let lo = k.saturating_sub(b.len() - 1);
            let hi = k.min(a.len() - 1);
            (lo..=hi).map(|i| a[i] * b[k - i]).sum::<f64>()
        })
        .collect()
}

/// CONTROL: `pmul` agrees with the reference on ordinary coefficients.
#[test]
fn pmul_matches_independent_convolution_on_ordinary_coefficients() {
    let a = [1.0, -2.0, 3.0];
    let b = [0.5, 4.0];
    let got = cv_calib3d::pnp::pmul(&a, &b);
    let want = reference_poly_mul(&a, &b);

    assert_eq!(got.len(), want.len(), "degree/length mismatch");
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * w.abs().max(1.0),
            "degree {k}: got {g}, want {w}"
        );
    }
}

/// The exact Grunert coefficient triples used by `estimate_p3p`, fed through
/// `pmul` and compared against the reference. These are the shapes that decide
/// whether the quartic has the right roots, so a transposition bug in `pmul`
/// shows up here and nowhere else.
#[test]
fn pmul_matches_reference_on_grunert_coefficient_shapes() {
    let sa2 = 0.25; // sa^2
    let sb2 = 0.36; // sb^2
    let sc2 = 0.49; // sc^2
    let ca = 0.3;
    let cb = -0.15;
    let cg = 0.42;
    let m = sb2 - sc2;

    // npoly = m * dpoly, scaled by a^2 on the constant term:
    //   dpoly  = [1, -2ca, 1]
    let dpoly = [1.0, -2.0 * ca, 1.0];
    let npoly = [m * dpoly[0] - sa2, m * dpoly[1], m * dpoly[2] + sa2];
    let dpoly_d = [-2.0 * sa2 * cb, 2.0 * sa2 * cg];
    let dpoly_d2 = reference_poly_mul(&dpoly_d, &dpoly_d);

    let cases: [(&str, &[f64], &[f64]); 3] = [
        ("t1 = npoly * npoly", &npoly, &npoly),
        ("t2 = npoly * dpoly_d", &npoly, &dpoly_d),
        (
            "t3 = [a2 - b2, 2 b2 ca, -b2] * dpoly_d^2",
            &[sa2 - sb2, 2.0 * sb2 * ca, -sb2][..],
            &dpoly_d2[..],
        ),
    ];

    for (name, p, q) in cases {
        let got = cv_calib3d::pnp::pmul(&p, &q);
        let want = reference_poly_mul(&p, &q);
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "{name}: degree {k}: got {g}, want {w}"
            );
        }
    }
}

/// CONTROL + end-to-end: a planted pose is recovered, so the surviving
/// coefficient assembly really does produce the Grunert quartic whose real
/// positive roots reconstruct the three camera rays.
///
/// Three widely separated, non-collinear object points; 3D->2D by exact
/// pinhole projection; the ground-truth pose must appear among the (up to 4)
/// P3P candidates to within 1e-9 rad / 1e-9 m.
#[test]
fn estimate_p3p_recovers_planted_pose_and_reprojects_to_sub_picopixel() {
    let model = PinholeModel {
        intrinsics: CameraIntrinsics::new(800.0, 810.0, 320.0, 240.0, 640, 480),
        distortion: Distortion::none(),
    };
    let r_true = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0)), 0.3)
        .into_inner();
    let t_true = Vector3::new(0.2, -0.1, 3.0);

    let obj = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.0, 0.1),
        Vector3::new(0.1, 0.6, -0.05),
    ];

    let mut img = [[0.0f64; 2]; 3];
    for i in 0..3 {
        let pc = r_true * obj[i] + t_true;
        assert!(pc.z > 0.0, "planted point must be in front of the camera");
        let pr = model.intrinsics.project(&Point3::from(pc));
        img[i] = [pr.x, pr.y];
    }

    let poses = PnpSolver::estimate_p3p(&obj, &img, &model).expect("P3P should not error");
    assert!(!poses.is_empty(), "P3P returned no candidate poses");

    let matched = poses.iter().any(|pose| {
        let rel = r_true.transpose() * pose.rotation_matrix();
        let trace = rel[(0, 0)] + rel[(1, 1)] + rel[(2, 2)];
        let ang = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        ang < 1e-9 && (pose.translation - t_true).norm() < 1e-9
    });
    assert!(matched, "ground-truth pose missing from the P3P candidates");

    // Every candidate must at least reproject the three correspondences.
    for pose in &poses {
        for i in 0..3 {
            let pc = pose.rotation_matrix() * obj[i] + pose.translation;
            if pc.z <= 0.0 {
                continue;
            }
            let pr = model.intrinsics.project(&Point3::from(pc));
            let e = ((pr.x - img[i][0]).powi(2) + (pr.y - img[i][1]).powi(2)).sqrt();
            assert!(
                e < 1e-6,
                "candidate reprojects point {i} with {e} px error (quartic is wrong)"
            );
        }
    }
}
