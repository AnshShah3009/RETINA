//! Colored ICP must not report a registration it did not perform, and its
//! reported metrics must describe the transform it returns.
//!
//! Both defects live in `registration/colored.rs` and both are copies of
//! defects already fixed in the CPU sibling `registration_icp_point_to_plane`
//! (`registration/mod.rs`), whose comments are the specification for the
//! correct behaviour.
//!
//! 1. `if valid_points < 10 { break; }` fell through to the tail, which
//!    returned `Some` with `fitness: 0.0` and `inlier_rmse: f32::MAX` - the
//!    sentinels `best_fitness`/`best_rmse` are initialised to, because the code
//!    that lowers them sits *after* the break. A target cloud without normals is
//!    the *ordinary* case, since `PointCloud.normals` is optional.
//!
//! 2. The "best" record was compared on `fitness` alone. `fitness` is
//!    `valid_points / source.points.len()`, which saturates at 1.0 as soon as
//!    everything matches, so the record froze at iteration 0 while
//!    `transformation` kept being overwritten. The function returned the *live*
//!    iterate next to iteration 0's rmse.

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix4, Point3};

fn cloud(points: &[[f32; 3]], colors: &[[f32; 3]], normals: Option<&[[f32; 3]]>) -> PointCloud {
    PointCloud {
        points: points
            .iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect(),
        colors: Some(
            colors
                .iter()
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect(),
        ),
        normals: normals.map(|ns| {
            ns.iter()
                .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
                .collect()
        }),
    }
}

fn cube() -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    for &x in &[0.0f32, 0.4, 0.8, 1.0] {
        for &y in &[0.0f32, 0.4, 0.8, 1.0] {
            for &z in &[0.0f32, 0.4, 0.8, 1.0] {
                v.push([x, y, z]);
            }
        }
    }
    v
}

fn colors_for(pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
    // Colour must vary across points: a constant colour makes every
    // correspondence's photometric residual zero and the term contributes
    // nothing to `A` at all.
    pts.iter()
        .map(|p| {
            [
                (p[0] * 180.0 + p[1] * 40.0) as u8 as f32,
                (p[1] * 180.0) as u8 as f32,
                (p[2] * 180.0) as u8 as f32,
            ]
        })
        .collect()
}

fn face_normals(pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
    // The normal of whichever of the cube's six faces the point sits furthest
    // out on. A single uniform normal (e.g. all +z) cannot observe motion along
    // the perpendicular axes, so the geometric term would be degenerate and the
    // test would prove nothing.
    pts.iter()
        .map(|p| {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for ax in 0..3usize {
                for &s in &[0.0f32, 1.0] {
                    let d = (p[ax] - s).abs();
                    if d < best_d {
                        best_d = d;
                        best = ax;
                    }
                }
            }
            let mut n = [0.0f32; 3];
            n[best] = if p[best] > 0.5 { 1.0 } else { -1.0 };
            n
        })
        .collect()
}

/// A grid of `n` non-degenerate points, source offset from target by `dy`.
fn pair(n: usize, dy: f32, normals: bool) -> (PointCloud, PointCloud) {
    let pts: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            [
                (i % 7) as f32 * 0.3,
                ((i / 7) % 7) as f32 * 0.3,
                (i / 49) as f32 * 0.3,
            ]
        })
        .collect();
    let cols = colors_for(&pts);
    let nrm = normals.then(|| face_normals(&pts));
    let src: Vec<[f32; 3]> = pts.iter().map(|p| [p[0], p[1] + dy, p[2]]).collect();
    (
        cloud(&src, &cols, nrm.as_deref()),
        cloud(&pts, &cols, nrm.as_deref()),
    )
}

/// Too few usable points must be `None`, not `Some` carrying the sentinels.
///
/// Pre-fix this returned
/// `Some(ColoredICPResult { identity, fitness: 0.0, inlier_rmse: 3.4028235e38 })`
/// for a 60-point source/target with a 0.01 correspondence distance. Every one
/// of the returned numbers is the value `best_fitness`/`best_rmse` start at,
/// which is only possible because the loop exited before the code that lowers
/// them - so the result is a registration that provably never happened.
#[test]
fn too_few_points_returns_none_rather_than_sentinel_metrics() {
    let n = 60;
    let (source, target) = pair(n, 0.5, false);

    let r = cv_registration::registration::registration_colored_icp(
        &source,
        &target,
        // Nothing is within 0.01 of its counterpart at a 0.5 offset.
        0.01,
        &Matrix4::identity(),
        10,
        0.5,
    );
    assert!(
        r.is_none(),
        "a registration with no usable points must report None, not \
         Some(fitness 0.0, inlier_rmse f32::MAX): got {:?}",
        r.map(|v| (v.fitness, v.inlier_rmse, v.transformation))
    );
}

/// The same case with a target that *does* have normals still has to return
/// `None` - the guard is about having too few points, not about normals. This
/// is the control for the test above in the sense that it pins the guard to its
/// actual cause: normals are present, so nothing else is wrong.
#[test]
fn too_few_points_with_normals_also_returns_none() {
    let n = 60;
    let (source, target) = pair(n, 0.5, true);

    let r = cv_registration::registration::registration_colored_icp(
        &source,
        &target,
        0.01,
        &Matrix4::identity(),
        10,
        0.5,
    );
    assert!(
        r.is_none(),
        "normals present but still no correspondences: must still be None, got {:?}",
        r.map(|v| (v.fitness, v.inlier_rmse))
    );
}

/// The reported metrics must describe the transform that is returned.
///
/// Pre-fix, `fitness` saturated at 1.0 from iteration 1 so `fitness >
/// best_fitness` was false forever: `best_rmse` was frozen at iteration 0's
/// error (~1e-2 on the crate's own coloured-cube case) while the returned
/// transform was the final iterate, whose actual error is ~1e-6. The two
/// described different poses.
///
/// The CPU sibling's rule is the specification: a lower rmse is unambiguously
/// a better fit, so that decides, with ties going to the higher fitness.
#[test]
fn the_reported_metrics_describe_the_returned_transform() {
    let pts = cube();
    let cols = colors_for(&pts);
    let nrm = face_normals(&pts);
    let src: Vec<[f32; 3]> = pts.iter().map(|p| [p[0], p[1] + 0.02, p[2]]).collect();
    let source = cloud(&src, &cols, Some(&nrm));
    let target = cloud(&pts, &cols, Some(&nrm));

    // CONTROL: a well-formed registration must still work, or the assertions
    // below would pass for the wrong reason.
    let control = cv_registration::registration::registration_colored_icp(
        &source,
        &target,
        1.0,
        &Matrix4::identity(),
        20,
        1.0,
    )
    .expect("control: a coloured cube offset by 0.02 in y must register");
    assert!(
        control.fitness > 0.9,
        "control: a normal registration must report a good fitness, got {}",
        control.fitness
    );
    assert!(
        control.transformation[(1, 3)].abs() > 1e-5,
        "control: the pose must actually move, y stayed at 0: {:?}",
        control.transformation
    );

    for lambda in [0.5f32, 1.0] {
        let r = cv_registration::registration::registration_colored_icp(
            &source,
            &target,
            1.0,
            &Matrix4::identity(),
            20,
            lambda,
        )
        .unwrap_or_else(|| panic!("lambda={lambda} returned None"));

        // Re-derive the metric at the transform actually returned, by
        // re-running the same correspondence/residual definition.
        let actual = residual_rmse(&source, &target, &r.transformation, lambda, 1.0);

        assert!(
            actual <= r.inlier_rmse as f64 * 1.05 + 1e-9,
            "lambda={lambda}: the reported inlier_rmse ({:.6e}) is worse than the \
             residual actually achieved at the returned transform ({actual:.6e}), \
             so the two describe different poses. Reported metrics are frozen at \
             iteration 0 while the transform keeps moving.",
            r.inlier_rmse
        );
        assert!(
            r.inlier_rmse.is_finite() && r.inlier_rmse < 0.1,
            "lambda={lambda}: reported inlier_rmse {} is not a plausible error for \
             a converged registration of a 0.02 offset",
            r.inlier_rmse
        );
    }
}

/// The residual the implementation itself would report at `transformation`:
/// nearest-target distance combined with the grayscale difference, exactly as
/// `registration_colored_icp` accumulates it, under the same correspondence
/// gate.
fn residual_rmse(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    lambda: f32,
    max_dist: f32,
) -> f64 {
    let s_colors = source.colors.as_ref().expect("colors");
    let t_colors = target.colors.as_ref().expect("colors");
    let mut total = 0.0f64;
    let mut n = 0usize;
    for i in 0..source.points.len() {
        let moved = transformation.transform_point(&source.points[i]);
        let mut best = f32::MAX;
        let mut best_idx = 0usize;
        for (j, t) in target.points.iter().enumerate() {
            let d = f64::from((moved - t).norm_squared()).sqrt() as f32;
            if d < best {
                best = d;
                best_idx = j;
            }
        }
        if best > max_dist {
            continue;
        }
        let gray = |c: &Point3<f32>| 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
        let photometric = (gray(&s_colors[i]) - gray(&t_colors[best_idx])).abs();
        let residual = lambda * best + (1.0 - lambda) * photometric * 0.1;
        total += f64::from(residual) * f64::from(residual);
        n += 1;
    }
    assert!(n > 0, "the re-derived residual used no correspondences");
    (total / n as f64).sqrt()
}
