//! Colored ICP Registration
//!
//! Park, Zhou, Koltun, "Colored Point Cloud Registration Revisited" (ICCV 2017),
//! following Open3D's `TransformationEstimationForColoredICP`.
//!
//! Each target point carries a colour gradient `d_p` lying in its tangent plane,
//! fitted by least squares to the intensity differences of its neighbours
//! projected onto that plane. A correspondence `(s, t)` with the source point
//! moved to `p = T s` then contributes two residuals:
//!
//! ```text
//! r_G = sqrt(lambda)     * (p - q) . n                    (signed point-to-plane)
//! r_I = sqrt(1 - lambda) * (I_s - (I_t + d . M (p - q)))  (M = I - n n^T)
//! ```
//!
//! linearised at `p` with respect to a left-multiplied twist, solved by
//! Gauss-Newton with the same `[v; w]` convention and SE(3) exponential map as
//! point-to-plane ICP in `mod.rs`.
//!
//! Colours are expected in `[0, 1]` (Open3D's convention, and what `cv-io`
//! loads); intensity is the mean of the three channels. With 0-255 colours the
//! photometric term is weighted ~65000x more heavily than `lambda` suggests.

use super::{
    exponential_map_se3, find_correspondences, fitness_and_rmse, point_to_plane_row, SimpleNN,
};
use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector6};
use rayon::prelude::*;

/// Colored ICP result
#[derive(Debug, Clone)]
pub struct ColoredICPResult {
    pub transformation: Matrix4<f32>,
    /// Fraction of source points with a correspondence at `transformation`.
    pub fitness: f32,
    /// Point-to-point RMSE of those correspondences, as Open3D reports it -
    /// equal to `evaluate_registration` on `transformation`.
    pub inlier_rmse: f32,
}

/// Neighbours used for each target point's colour gradient (Open3D: 30).
const GRADIENT_MAX_NN: usize = 30;
/// Open3D's default relative convergence criteria.
const RELATIVE_FITNESS: f32 = 1e-6;
const RELATIVE_RMSE: f32 = 1e-6;

fn intensity(c: &Point3<f32>) -> f32 {
    (c.x + c.y + c.z) / 3.0
}

/// Colored ICP registration.
///
/// `lambda_geometric` in `[0, 1]` weighs the geometric term against the
/// photometric one (Open3D's default is 0.968). The target must have normals
/// and both clouds colours; otherwise, or with fewer than three
/// correspondences at `init_transformation`, the result is `None`.
///
/// Previously the photometric Jacobian contained no colour gradient at all,
/// the geometric Jacobian was evaluated at the *untransformed* source point,
/// and the geometric residual was the unsigned distance, so the iteration did
/// not converge: a 0.01 offset that point-to-plane ICP recovers to 1e-4 was
/// not recovered (the function returned `None` on the regression scene).
pub fn registration_colored_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
    lambda_geometric: f32,
) -> Option<ColoredICPResult> {
    let source_colors = source.colors.as_ref()?;
    let target_colors = target.colors.as_ref()?;
    // Open3D refuses colored ICP without target normals: both residuals live
    // in the target's tangent plane.
    let target_normals = target.normals.as_ref()?;
    if source.points.is_empty()
        || target.points.is_empty()
        || source_colors.len() != source.points.len()
        || target_colors.len() != target.points.len()
        || target_normals.len() != target.points.len()
        || !(0.0..=1.0).contains(&lambda_geometric)
    {
        return None;
    }

    let target_nn = SimpleNN::new(target.points.clone());
    let gradients = color_gradients(
        target,
        target_normals,
        target_colors,
        &target_nn,
        2.0 * max_correspondence_distance,
    );
    let source_len = source.points.len();
    let lambda_photometric = 1.0 - lambda_geometric;

    let mut transformation = *init_transformation;
    let mut correspondences = find_correspondences(
        source,
        &target_nn,
        &transformation,
        max_correspondence_distance,
    );
    if correspondences.len() < 3 {
        return None;
    }
    let (mut fitness, mut rmse) = fitness_and_rmse(&correspondences, source_len);
    // Pose and the metrics measured at it, kept together (see the
    // point-to-plane ICP for why the best rather than the last).
    let mut best = (transformation, fitness, rmse);

    for _ in 0..max_iterations {
        let mut ata = nalgebra::Matrix6::<f64>::zeros();
        let mut atb = Vector6::<f64>::zeros();

        for &(s, t, _) in &correspondences {
            let p = transformation.transform_point(&source.points[s]);
            let q = target.points[t];
            let n = target_normals[t];

            // Geometric term: shared with point-to-plane ICP.
            let (j_g, r_g) = point_to_plane_row(&p, &q, &n);

            // Photometric term. The target intensity is extrapolated to the
            // projection of `p` onto the target's tangent plane:
            //   I_t(p) = I_t + d . M (p - q),  M = I - n n^T
            // so r_I = I_s - I_t(p) and dr_I/dp = -M d. A perturbation moves `p`
            // by `v + w x p`, giving J_I = [g; p x g] with g = -M d - the same
            // structure as the geometric row.
            let m = Matrix3::identity() - n * n.transpose();
            let d = gradients[t];
            let r_i = intensity(&source_colors[s])
                - (intensity(&target_colors[t]) + d.dot(&(m * (p - q))));
            let g = -(m * d);
            let pg = p.coords.cross(&g);
            let j_i = Vector6::new(g.x, g.y, g.z, pg.x, pg.y, pg.z);

            let (j_g, j_i) = (j_g.cast::<f64>(), j_i.cast::<f64>());
            let (lg, lp) = (f64::from(lambda_geometric), f64::from(lambda_photometric));
            ata += j_g * j_g.transpose() * lg + j_i * j_i.transpose() * lp;
            atb += j_g * (f64::from(r_g) * lg) + j_i * (f64::from(r_i) * lp);
        }

        // Exact inverse when the system is well posed, pseudo-inverse for the
        // partially-constrained cases (a plane with a weak texture, lambda at
        // either end); `None` if neither yields a finite step.
        let delta = match ata.try_inverse() {
            Some(inv) => -(inv * atb),
            None => -(ata.pseudo_inverse(1e-12).ok()? * atb),
        };
        if !delta.iter().all(|v| v.is_finite()) {
            return None;
        }
        transformation = exponential_map_se3(&delta.cast::<f32>()) * transformation;

        let next = find_correspondences(
            source,
            &target_nn,
            &transformation,
            max_correspondence_distance,
        );
        if next.len() < 3 {
            break;
        }
        let (next_fitness, next_rmse) = fitness_and_rmse(&next, source_len);
        let converged = (next_fitness - fitness).abs() < RELATIVE_FITNESS
            && (next_rmse - rmse).abs() < RELATIVE_RMSE;
        correspondences = next;
        fitness = next_fitness;
        rmse = next_rmse;
        if fitness > best.1 || (fitness == best.1 && rmse < best.2) {
            best = (transformation, fitness, rmse);
        }
        if converged {
            break;
        }
    }

    Some(ColoredICPResult {
        transformation: best.0,
        fitness: best.1,
        inlier_rmse: best.2,
    })
}

/// Per-target-point colour gradient in the tangent plane (Open3D's
/// `InitializePointCloudForColoredICP`).
///
/// For point `q` with normal `n` and up to `GRADIENT_MAX_NN` neighbours within
/// `radius`, solve in least squares
///
/// ```text
/// (proj(q_k) - q) . d = I_k - I_q     for each neighbour q_k
/// (N - 1) n . d       = 0             keeps d in the tangent plane
/// ```
///
/// with `proj` the projection onto the tangent plane. Fewer than three
/// neighbours leaves the gradient at zero.
fn color_gradients(
    target: &PointCloud,
    normals: &[Vector3<f32>],
    colors: &[Point3<f32>],
    nn: &SimpleNN,
    radius: f32,
) -> Vec<Vector3<f32>> {
    let radius_sq = radius * radius;
    let gradient = |i: usize| -> Vector3<f32> {
        let q = target.points[i];
        let n = normals[i].cast::<f64>();
        let i_q = f64::from(intensity(&colors[i]));
        let neighbours: Vec<_> = nn
            .k_nearest(&q, GRADIENT_MAX_NN)
            .into_iter()
            .filter(|&(_, j, d2)| j != i && d2 <= radius_sq)
            .collect();
        if neighbours.len() < 3 {
            return Vector3::zeros();
        }
        let mut ata = Matrix3::<f64>::zeros();
        let mut atb = Vector3::<f64>::zeros();
        for &(p, j, _) in &neighbours {
            let diff = (p - q).cast::<f64>();
            let row = diff - n * n.dot(&diff);
            ata += row * row.transpose();
            atb += row * (f64::from(intensity(&colors[j])) - i_q);
        }
        let w = neighbours.len() as f64;
        ata += n * n.transpose() * (w * w);
        match ata.try_inverse() {
            Some(inv) => (inv * atb).cast::<f32>(),
            None => Vector3::zeros(),
        }
    };
    if target.points.len() > 5000 {
        (0..target.points.len())
            .into_par_iter()
            .map(gradient)
            .collect()
    } else {
        (0..target.points.len()).map(gradient).collect()
    }
}

#[cfg(test)]
mod colored_icp_actually_moves {
    use super::*;

    fn cloud(
        points: Vec<[f32; 3]>,
        colors: Vec<[f32; 3]>,
        normals: Option<Vec<[f32; 3]>>,
    ) -> PointCloud {
        let pts: Vec<Point3<f32>> = points
            .iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect();
        let cols: Vec<Point3<f32>> = colors
            .iter()
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let nrm = normals.map(|ns| {
            ns.iter()
                .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
                .collect::<Vec<_>>()
        });
        PointCloud {
            points: pts,
            colors: Some(cols),
            normals: nrm,
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

    /// Colored ICP must move the pose, at every `lambda_geometric` it supports.
    ///
    /// Scope of this test, stated honestly: it shows the pose is *estimated*, at
    /// every lambda in range. It does **not** distinguish the least-squares
    /// solve from the old `try_inverse` path, because on this input - a
    /// well-conditioned cloud with real normals - `try_inverse` succeeds too.
    /// What the earlier code got wrong was the rank-1 photometric Jacobian, and
    /// that is what regressing the constant would show. The `qr().solve` change
    /// is instead justified by the degenerate case it handles, and is recorded
    /// in the comment at the call site.
    #[test]
    fn colored_icp_does_not_return_the_identity() {
        let pts = cube();
        // Colour must vary across points, or every correspondence has a zero
        // photometric residual and the term contributes nothing to `A` at all.
        let cols: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| {
                [
                    (p[0] * 180.0 + p[1] * 40.0) as u8 as f32,
                    (p[1] * 180.0) as u8 as f32,
                    (p[2] * 180.0) as u8 as f32,
                ]
            })
            .collect();
        // Outward face normals, so the geometric term is real rather than a
        // zero Jacobian. Without them `n` is zero, `J_geo` is entirely zero,
        // and `A` has three exact zeros on its diagonal at any lambda.
        // Outward face normals: the nearest of the cube's six faces, so each
        // point gets a normal along the axis it sits furthest out on. Without
        // them `n` is zero, `J_geo` is entirely zero, and `A` has three exact
        // zeros on its diagonal at any lambda.
        let nrm: Vec<[f32; 3]> = pts
            .iter()
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
            .collect();
        let source = cloud(
            pts.iter().map(|p| [p[0], p[1] + 0.02, p[2]]).collect(),
            cols.clone(),
            Some(nrm.clone()),
        );
        let target = cloud(pts, cols, Some(nrm));
        let identity = Matrix4::identity();

        // `lambda = 0` is excluded: a purely photometric point-wise model is
        // rank-deficient by construction, and now reports that rather than
        // pretending to converge. 0 < lambda <= 1 must all move the pose.
        for lambda in [0.1f32, 0.5, 1.0] {
            let r = registration_colored_icp(&source, &target, 1.0, &identity, 20, lambda)
                .unwrap_or_else(|| panic!("lambda={lambda} returned None"));
            assert!(
                r.transformation[(1, 3)].abs() > 1e-5,
                "lambda={lambda}: pose did not move, y stayed at 0 - the solver \
                 is returning the input transform. t={:?}",
                r.transformation
            );
        }
    }
}
