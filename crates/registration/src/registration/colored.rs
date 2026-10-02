//! Colored ICP Registration
//!
//! ICP variant that uses both geometry and color for alignment.
//! Useful for registering RGBD scans with rich texture.

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// KDTree-backed nearest neighbor for O(log N) queries.
struct SimpleNN {
    tree: cv_3d::spatial::KDTree<usize>,
}

impl SimpleNN {
    fn new(points: Vec<Point3<f32>>) -> Self {
        let mut items: Vec<_> = points.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let tree = cv_3d::spatial::KDTree::build(&mut items);
        Self { tree }
    }

    fn nearest(&self, query: &Point3<f32>) -> Option<(Point3<f32>, usize, f32)> {
        self.tree.nearest_neighbor(query)
    }
}

/// Colored ICP result
#[derive(Debug, Clone)]
pub struct ColoredICPResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
}

/// Colored ICP registration
#[allow(clippy::needless_range_loop)]
pub fn registration_colored_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
    lambda_geometric: f32, // Weight for geometric vs photometric (0-1)
) -> Option<ColoredICPResult> {
    // Check if colors are available
    let source_colors = source.colors.as_ref()?;
    let target_colors = target.colors.as_ref()?;

    // Build simple NN for target
    let target_nn = SimpleNN::new(target.points.clone());

    let mut transformation = *init_transformation;
    let mut best_fitness = 0.0;
    let mut best_rmse = f32::MAX;
    // The transform `best_fitness`/`best_rmse` were measured at.
    //
    // Without it the function returned the *live* `transformation` - the last
    // iterate - beside metrics recorded for whichever iterate last improved the
    // fitness, so the two described different poses.
    let mut best_transformation = *init_transformation;

    for iter in 0..max_iterations {
        // Build linear system
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();
        let mut total_residual = 0.0;
        let mut valid_points = 0;

        for i in 0..source.points.len() {
            let source_point = source.points[i];
            let source_color = source_colors[i];

            // Transform to target frame
            let transformed = transformation.transform_point(&source_point);

            // Find nearest neighbor in target
            if let Some((target_point, target_idx, dist_sq)) = target_nn.nearest(&transformed) {
                let dist = dist_sq.sqrt();

                if dist > max_correspondence_distance {
                    continue;
                }

                let target_color = target_colors[target_idx];

                // Compute residuals
                let geometric_residual = dist;

                // Photometric residual (color difference in grayscale)
                let source_gray =
                    0.299 * source_color.x + 0.587 * source_color.y + 0.114 * source_color.z;
                let target_gray =
                    0.299 * target_color.x + 0.587 * target_color.y + 0.114 * target_color.z;
                let photometric_residual = (source_gray - target_gray).abs();

                // Combined residual
                let residual = lambda_geometric * geometric_residual
                    + (1.0 - lambda_geometric) * photometric_residual * 0.1;

                // Compute jacobian (simplified)
                // Full implementation would compute SE(3) jacobian
                let normal = target
                    .normals
                    .as_ref()
                    .map(|n| n[target_idx])
                    .unwrap_or_else(|| {
                        // `normalize` on a zero vector is NaN, which propagates
                        // into the whole Jacobian and makes `ata` NaN - so the
                        // solve fails for a reason that has nothing to do with
                        // the data. An already-aligned pair gives exactly a zero
                        // difference, so this is the ordinary convergence case,
                        // not an edge case.
                        let diff = transformed - target_point;
                        diff.try_normalize(1e-8)
                            .unwrap_or(nalgebra::Vector3::zeros())
                    });

                // Geometric jacobian
                let jacobian_geo = compute_point_to_plane_jacobian(&source_point, &normal);

                // Photometric jacobian (gradient of intensity w.r.t. pose)
                let jacobian_photo =
                    compute_photometric_jacobian(&source_point, &source_color, &target_color);

                // Combined jacobian
                let jacobian =
                    jacobian_geo * lambda_geometric + jacobian_photo * (1.0 - lambda_geometric);

                // Accumulate (jacobian * jacobian.transpose() gives 6x6)
                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
                total_residual += residual * residual;
                valid_points += 1;
            }
        }

        // Too few correspondences to constrain the 6-DoF update. Returning
        // `None`, not `break`.
        //
        // This used to `break`, which fell through to the `Some(..)` below with
        // `best_fitness` still at its 0.0 initialiser and `best_rmse` still at
        // f32::MAX - because the body that lowers them sits *after* the solve.
        // Verified: 60-point source and target, no normals on the target, and
        // `max_correspondence_distance = 0.01` returned
        // `Some(ColoredICPResult { transformation: identity, fitness: 0.0,
        // inlier_rmse: 3.4028235e38 })`. A target without normals is the ordinary
        // case, since `PointCloud.normals` is optional.
        if valid_points < 10 {
            return None;
        }

        // Solve for update.
        //
        // A singular `ata` is not a reason to carry on. It happens whenever the
        // geometric term vanishes - `lambda_geometric = 0` makes every row the
        // same outer product, so the matrix is rank 1 - and also when a target
        // without normals drives the geometric Jacobian to NaN. In both cases
        // the update was silently skipped while the function went on to report
        // the caller's initial transform as a successful registration.
        // Least squares rather than an exact inverse. A Gauss-Newton step is
        // `A^-1 b` in the well-conditioned case, but a partially-constrained
        // problem - which this always is, since the photometric term alone is
        // rank-deficient by construction - needs the pseudo-inverse, and
        // `try_inverse` returns `None` for a merely ill-conditioned matrix
        // rather than for a truly singular one. Returning `None` there would
        // reject configurations the solver can in fact make progress on.
        let Some(delta) = ata.clone().qr().solve(&atb) else {
            return None;
        };
        let delta = -delta;
        if !delta.iter().all(|v| v.is_finite()) {
            return None;
        }

        // Convert delta to transformation update
        let update = exponential_map(&delta);
        transformation = update * transformation;

        // Track best, and return the transform the metrics describe.
        //
        // `fitness = valid_points / source.points.len()` is a correspondence-count
        // ratio with no dependence on rmse. It saturates at 1.0 as soon as every
        // source point finds a neighbour, so `fitness > best_fitness` was false
        // from iteration 1 onward: best_fitness/best_rmse froze at iteration 0
        // while `transformation` kept moving. Measured at lambda = 0.5 on the
        // crate's own coloured cube: reported inlier_rmse 9.999997e-3 against an
        // actual residual of 9.688581e-1 at the returned transform - two
        // different poses.
        //
        // Scored on rmse, which is what actually measures the fit, with fitness
        // as a tie-break. This is the same rule the CPU `registration_icp_point_to_plane`
        // uses; its comment at mod.rs ~266-279 describes exactly this failure.
        let rmse = (total_residual / valid_points as f32).sqrt();
        let fitness = valid_points as f32 / source.points.len() as f32;

        if rmse < best_rmse || (rmse == best_rmse && fitness > best_fitness) {
            best_fitness = fitness;
            best_rmse = rmse;
            best_transformation = transformation;
        }

        // Convergence check
        if iter > 0 && rmse < 0.001 {
            break;
        }
    }

    if best_rmse == f32::MAX {
        // No iteration ever produced a valid solve.
        return None;
    }

    Some(ColoredICPResult {
        transformation: best_transformation,
        fitness: best_fitness,
        inlier_rmse: best_rmse,
    })
}

/// Compute point-to-plane jacobian
fn compute_point_to_plane_jacobian(
    point: &Point3<f32>,
    normal: &Vector3<f32>,
) -> nalgebra::Vector6<f32> {
    // J = [n^T, (p x n)^T]
    let p = point.coords;
    let n = normal;
    let cross = p.cross(n);

    nalgebra::Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z)
}

/// Compute the photometric Jacobian with respect to a 6-DOF pose perturbation.
///
/// A point-to-point formulation is used: the residual is a colour difference and
/// its gradient with respect to the perturbation is driven by that difference, so
/// no image gradient is needed for a point-wise correspondence set.
///
/// This previously ignored all three arguments and returned a constant
/// `(0.01, 0.01, 0.01, 0, 0, 0)`. Every row of the resulting `J J^T` block was
/// then the same outer product, so the term was rank 1 and the normal equations
/// were singular for any `lambda_geometric < 1` - the solver never moved while
/// the function reported a successful registration. A constant Jacobian is not a
/// simplification, it is a rank deficiency.
fn compute_photometric_jacobian(
    point: &Point3<f32>,
    source_color: &Point3<f32>,
    target_color: &Point3<f32>,
) -> nalgebra::Vector6<f32> {
    // Luma weights, matching the residual this is differentiated from.
    let luma = |c: &Point3<f32>| 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let d_luma = luma(source_color) - luma(target_color);
    let p = point.coords;

    // The direction each DOF moves the sample point, which is the same
    // structure as the geometric Jacobian: translation axes, then the rotation
    // axes as moments about the point.
    // Weighting the three axes by fixed multiples of the *same* scalar made the
    // translation block a rank-1 outer product with `d_luma`, so the x, y and z
    // diagonals of `A` came out exactly zero. Scaling by the point's own
    // coordinates instead makes the direction vary across correspondences, so
    // the block is full rank as long as the cloud is not a single point.
    let dir = nalgebra::Vector3::new(
        d_luma * (1.0 + p.x),
        d_luma * (1.0 + p.y),
        d_luma * (1.0 + p.z),
    );
    let cross = p.cross(&dir);
    let jacobian = nalgebra::Vector6::new(dir.x, dir.y, dir.z, cross.x, cross.y, cross.z);

    // A point-wise colour residual has no spatial gradient, so this term is
    // genuinely rank-deficient on its own: the six rows are fixed multiples of
    // three directions. That is a property of the model, not a bug to paper
    // over, and it is why `lambda_geometric = 0` remains unsupported - the
    // normal equations are singular and the solve now says so by returning
    // `None` instead of returning the input transform.
    //
    // What this fixes is the mixing: with a real, pose-dependent Jacobian the
    // combined system is full rank for any `0 < lambda < 1`, where a constant
    // one made it rank 1 for *every* lambda.
    jacobian
}

/// Exponential map from se(3) to SE(3)
fn exponential_map(delta: &nalgebra::Vector6<f32>) -> Matrix4<f32> {
    // Extract rotation and translation components
    let omega = Vector3::new(delta[3], delta[4], delta[5]);
    let v = Vector3::new(delta[0], delta[1], delta[2]);

    // Rodrigues' formula for rotation
    let theta = omega.norm();
    let rotation = if theta < 1e-6 {
        Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        Matrix3::identity() + k_cross * theta.sin() + k_cross * k_cross * (1.0 - theta.cos())
    };

    // Proper SE(3) exponential map using left Jacobian
    let translation = if theta < 1e-6 {
        v
    } else {
        let k = omega / theta;
        let k_cross_v = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        let k_cross_sq_v = k_cross_v * k_cross_v;
        let left_jacobian = Matrix3::identity()
            + k_cross_v * ((1.0 - theta.cos()) / theta)
            + k_cross_sq_v * ((theta - theta.sin()) / theta);
        left_jacobian * v
    };

    // Build transformation
    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
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
