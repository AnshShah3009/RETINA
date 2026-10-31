//! ICP must not report a perfect score for a registration that did not happen.
//!
//! Both defects here were found by auditing untested code, and both produced the
//! same shape of wrong answer: a caller got `Some(ICPResult { fitness: 1.0, .. })`
//! with a transform identical to the one it passed in.
//!
//! The existing tests did not catch either, because `mod_test.rs` asserts
//! `res.fitness > 0.8` and `res.num_iterations > 0` - both of which a result of
//! "I did nothing" satisfies perfectly.

use cv_registration::registration_icp_point_to_plane;
use nalgebra::{Matrix4, Point3, Vector3};

fn cloud(points: &[(f32, f32, f32)], normals: Option<&[(f32, f32, f32)]>) -> cv_core::PointCloud {
    let mut pc = cv_core::PointCloud::default();
    for (i, p) in points.iter().enumerate() {
        pc.points.push(Point3::new(p.0, p.1, p.2));
        if let Some(n) = normals {
            let v = n[i];
            pc.normals
                .get_or_insert_with(Vec::new)
                .push(Vector3::new(v.0, v.1, v.2));
        }
    }
    pc
}

/// A singular `A` must fall back to a pseudo-inverse rather than skipping the
/// solve.
///
/// When `try_inverse()` failed the Gauss-Newton step was simply skipped, so
/// `transformation` stayed at the input while `fitness` - derived purely from
/// *how many* correspondences existed - saturated at 1.0. The caller got a
/// perfect score for a transform that never moved.
///
/// All-identical points make `A` robustly singular (rank 1): the Jacobian is
/// `[n, p x n]`, and with `p` constant the rotational columns span one
/// direction. The source is offset from the target, so the residual is
/// non-zero and there is something a solve *should* recover.
#[test]
fn a_singular_normal_matrix_still_applies_an_update() {
    let normals: Vec<(f32, f32, f32)> = vec![(0.0, 1.0, 0.0); 24];

    // All target points identical, so `A` is rank 1.
    let target_pts: Vec<(f32, f32, f32)> = vec![(0.5, 0.3, 0.2); 24];
    let target = cloud(&target_pts, Some(&normals));

    // The source is the same point shifted along y, which the plane normal can
    // see. This is a displacement the solve must apply.
    let source_pts: Vec<(f32, f32, f32)> = vec![(0.5, 0.33, 0.2); 24];
    let source = cloud(&source_pts, Some(&normals));

    let init = Matrix4::identity();
    let res = registration_icp_point_to_plane(&source, &target, 0.5, &init, 50).expect(
        "a plane offset with singular A is still solvable via the                  pseudo-inverse",
    );

    // What matters is that the residual was *reduced*, not which matrix slot the
    // displacement lands in. The update is applied as `update * transformation`
    // - a left-multiply - so the translation is not at the element one would
    // guess, and asserting on it would be asserting on a representation detail.
    assert!(
        res.inlier_rmse < 0.01,
        "the singular solve did not reduce the residual: rmse is {} against a \
         0.03 input offset",
        res.inlier_rmse
    );
    assert!(
        res.num_iterations > 0,
        "the residual fell, so the solve ran, but it recorded zero iterations"
    );
}

/// Zero correspondences must be `None`, not `Some` with `rmse = f32::MAX`.
///
/// The loop `break`s on too few correspondences and falls through to the tail,
/// which reported `fitness 0.0`, `inlier_rmse f32::MAX` and `num_iterations 0`
/// with the input transform. That reads as "registered successfully, found
/// nothing" - and an `inlier_rmse >= 0.0` assertion passes on `f32::MAX`.
#[test]
fn too_few_correspondences_returns_none() {
    let target_pts: Vec<(f32, f32, f32)> = (0..150).map(|i| (i as f32 * 0.01, 0.0, 0.0)).collect();
    let target_normals: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 1.0); 150];
    let target = cloud(&target_pts, Some(&target_normals));

    // 100 units away, with a 0.05 correspondence distance: nothing matches.
    let source_pts: Vec<(f32, f32, f32)> =
        target_pts.iter().map(|p| (p.0 + 100.0, p.1, p.2)).collect();
    let source = cloud(&source_pts, Some(&target_normals));

    let res = registration_icp_point_to_plane(&source, &target, 0.05, &Matrix4::identity(), 50);
    assert!(
        res.is_none(),
        "a registration with no correspondences must report None, not a result \
         with fitness 0 and rmse f32::MAX"
    );
}

/// An empty source has nothing to register.
#[test]
fn an_empty_source_returns_none() {
    let target = cloud(&[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], None);
    let source = cv_core::PointCloud::default();
    let res = registration_icp_point_to_plane(&source, &target, 1.0, &Matrix4::identity(), 10);
    assert!(res.is_none(), "an empty source cannot be registered");
}

/// The reported metrics must describe the transform that is returned.
///
/// `best_transformation` is only updated on a *strict* fitness improvement, so
/// once fitness saturates at 1.0 the recorded metrics stop matching the
/// transform, which keeps being overwritten. The result overstated the achieved
/// error by 5-6 orders of magnitude.
#[test]
fn the_reported_metrics_describe_the_returned_transform() {
    let n = 8;
    let target_pts: Vec<(f32, f32, f32)> = (0..n)
        .map(|i| {
            (
                i as f32 * 0.3,
                (i as f32 * 0.7) % 1.3,
                (i as f32 * 1.1) % 0.9,
            )
        })
        .collect();
    let target_normals: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 1.0); n];
    let target = cloud(&target_pts, Some(&target_normals));

    let offset = 0.05f32;
    let source_pts: Vec<(f32, f32, f32)> = target_pts
        .iter()
        .map(|p| (p.0, p.1, p.2 + offset))
        .collect();
    let source = cloud(&source_pts, Some(&target_normals));

    let res = registration_icp_point_to_plane(&source, &target, 0.5, &Matrix4::identity(), 50)
        .expect("a cube-like source should register");

    // Re-derive the residual at the transform actually returned.
    let t = res.transformation;
    let mut sum = 0.0f32;
    let mut used = 0usize;
    for (s, g) in source.points.iter().zip(target.points.iter()) {
        let moved = t.transform_point(s);
        let diff = moved - g;
        let residual = diff.z; // the target normals are all +z
        sum += residual * residual;
        used += 1;
    }
    let actual = f64::from((sum / used as f32).sqrt());
    let reported = res.inlier_rmse as f64;

    // The reported rmse must not *overstate* the error at the returned
    // transform. Comparing on best-fitness rather than best-fit is what let
    // them drift apart: fitness saturates at 1.0, so the record froze at
    // iteration 0 while the transform kept moving, and the reported rmse
    // overstated the achieved error by 5-6 orders of magnitude.
    assert!(
        actual <= reported * 1.05 + 1e-9,
        "the reported rmse ({reported:.6}) is worse than the residual actually \
         achieved at the returned transform ({actual:.6}), so the two describe \
         different transforms"
    );
}
