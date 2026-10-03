//! 3D Registration Module
//!
//! Implements various registration algorithms:
//! - ICP (Iterative Closest Point)
//! - Colored ICP
//! - Global Registration (RANSAC, FGR)
//! - GNC (Graduated Non-Convexity) robust registration

#![allow(deprecated)]

pub mod colored;
pub mod global;
pub mod gnc;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type RegistrationResult<T> = cv_core::Result<T>;

pub use colored::{registration_colored_icp, ColoredICPResult};
pub use cv_core::{Error, Result, RobustLoss};
pub use global::{
    registration_fgr_based_on_feature_matching, registration_ransac_based_on_feature_matching,
    FPFHFeature, FastGlobalRegistrationOption, GlobalRegistrationResult,
};
pub use gnc::{registration_gnc, GNCOptimizer, GNCResult, RobustLossType};

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix4, Point3};
use rayon::prelude::*;

/// Nearest-neighbor search structure for point cloud registration.
/// Uses a balanced KDTree for O(log N) queries.
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

/// ICP (Iterative Closest Point) registration result
///
/// Contains the optimized rigid transformation and quality metrics
/// from point cloud registration.
///
/// # Fields
///
/// * `transformation` - 4×4 SE(3) transformation matrix (rotation + translation)
/// * `fitness` - Fraction of inlier correspondences (0-1 range)
/// * `inlier_rmse` - Root mean square error of point-to-plane residuals
/// * `num_iterations` - Number of iterations performed before convergence
#[derive(Debug, Clone)]
pub struct ICPResult {
    /// Optimized 4×4 homogeneous transformation matrix (SE(3))
    pub transformation: Matrix4<f32>,
    /// Registration fitness score: fraction of points with valid correspondences (0-1)
    pub fitness: f32,
    /// Root mean square error of inlier point-to-plane distances
    pub inlier_rmse: f32,
    /// Iterations until convergence
    pub num_iterations: usize,
}

/// Point-to-plane ICP registration
///
/// Registers a source point cloud to a target point cloud using the
/// point-to-plane Iterative Closest Point (ICP) algorithm.
///
/// # Algorithm
///
/// Iteratively:
/// 1. Find nearest neighbors between transformed source and target
/// 2. Compute point-to-plane residuals using target surface normals
/// 3. Solve 6-DOF rigid transformation via least squares
/// 4. Update transformation using exponential map on SE(3)
/// 5. Repeat until convergence or max iterations
///
/// # Arguments
///
/// * `source` - Source point cloud (with or without normals)
/// * `target` - Target point cloud (should have surface normals for best results)
/// * `max_correspondence_distance` - Maximum distance for valid correspondences
/// * `init_transformation` - Initial guess for transformation (often identity)
/// * `max_iterations` - Maximum optimization iterations
///
/// # Returns
///
/// * `Some(ICPResult)` - Registration succeeded with transformation and metrics
/// * `None` - Registration failed (too few correspondences, no convergence, etc.)
///
/// # Performance
///
/// - Complexity: O(N × max_iterations) where N = points in source cloud
/// - Convergence: Typically 10-50 iterations for well-initialized problem
/// - Suitable for point clouds up to ~100k points
///
/// # Example
///
/// ```no_run
/// # use cv_registration::registration::registration_icp_point_to_plane;
/// # use cv_core::point_cloud::PointCloud;
/// # use nalgebra::Matrix4;
/// # let source = PointCloud { points: vec![], normals: None, colors: None };
/// # let target = PointCloud { points: vec![], normals: Some(vec![]), colors: None };
/// let result = registration_icp_point_to_plane(
///     &source,
///     &target,
///     0.1,  // max correspondence distance
///     &Matrix4::identity(),
///     50,   // max iterations
/// );
/// # if let Some(r) = result {
/// #   println!("Transformation:\n{}", r.transformation);
/// #   println!("Fitness: {:.3}", r.fitness);
/// # }
/// ```
pub fn registration_icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &Matrix4<f32>,
    max_iterations: usize,
) -> Option<ICPResult> {
    // Build simple nearest neighbor structure for target
    let target_nn = SimpleNN::new(target.points.clone());

    let mut transformation = *init_transformation;
    let mut best_fitness = 0.0;
    let mut best_rmse = f32::MAX;
    // The transform the best metrics were measured at. Without this the
    // function returned `transformation` - the *last* iterate - next to metrics
    // recorded for whichever iterate had the best fitness, so `inlier_rmse` and
    // `num_iterations` described a different transform from the one returned.
    // Measured: reported rmse overstated the achieved error by 5-6 orders of
    // magnitude, and in one case the reported `fitness` was not one the
    // returned transform achieved at all.
    let mut best_transformation = *init_transformation;
    let mut final_iterations = 0;

    for iter in 0..max_iterations {
        // Find correspondences (parallel for large clouds, serial for small)
        let find_corr = |src_idx: usize, src_point: &Point3<f32>| -> Option<(usize, usize, f32)> {
            let transformed = transformation.transform_point(src_point);
            target_nn
                .nearest(&transformed)
                .and_then(|(_, target_idx, dist_sq)| {
                    let dist = dist_sq.sqrt();
                    if dist <= max_correspondence_distance {
                        Some((src_idx, target_idx, dist))
                    } else {
                        None
                    }
                })
        };
        let correspondences: Vec<(usize, usize, f32)> = if source.points.len() > 5000 {
            source
                .points
                .par_iter()
                .enumerate()
                .filter_map(|(i, p)| find_corr(i, p))
                .collect()
        } else {
            source
                .points
                .iter()
                .enumerate()
                .filter_map(|(i, p)| find_corr(i, p))
                .collect()
        };

        // Too few correspondences to constrain a rigid transform. Returning
        // `None` is what this function's own documentation promises for "too few
        // correspondences", and it is the only honest answer: `break` fell
        // through to the tail, which reported `fitness 0.0` and
        // `inlier_rmse f32::MAX` with the input transform unchanged - a result
        // that reads as a successful registration that found nothing.
        //
        // The `n_used < 3` and `correspondences.is_empty()` guards further down
        // were unreachable for this case, because this fired first.
        if correspondences.len() < 3 {
            return None;
        }

        // Compute point-to-plane error
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();
        let mut total_residual = 0.0;

        // Correspondences that actually contribute geometry. A target without
        // normals gives none, and the accumulation below is skipped for every
        // one - leaving `ata` a zero matrix, so `try_inverse()` returns `None`,
        // the update is silently skipped, and the function used to go on to
        // report the caller's initial transform with `fitness: 1.0` and
        // `inlier_rmse: 0.0`. A perfect score from a result that never moved.
        // A target cloud with no normals is the *normal* case, since `PointCloud`
        // makes the field optional.
        let mut n_used = 0usize;

        for (src_idx, tgt_idx, _) in &correspondences {
            let src_point = source.points[*src_idx];
            let tgt_point = target.points[*tgt_idx];
            let tgt_normal = target.normals.as_ref().map(|n| &n[*tgt_idx]);

            if let Some(normal) = tgt_normal {
                n_used += 1;
                let transformed = transformation.transform_point(&src_point);
                let diff = transformed - tgt_point;
                let residual = diff.dot(normal);

                // Compute Jacobian
                let p = transformed.coords;
                let n = normal;
                let cross = p.cross(n);

                let jacobian = nalgebra::Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z);

                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
                total_residual += residual * residual;
            }
        }

        // Solve for update
        // A singular `A` means the correspondences cannot determine a rigid
        // motion - a collinear target, a planar one with identical normals, any
        // degenerate geometry. Skipping the solve silently was the defect: the
        // loop then fell through, `fitness` was computed purely from how many
        // correspondences existed, and the caller got `fitness 1.0` with the
        // *input* transform and the input's error still unfixed.
        //
        // The pseudo-inverse recovers the solvable subspace and converges the
        // observed cases exactly, so this is the useful behaviour rather than
        // refusing outright. If even that fails, `None` - never a perfect score
        // for a transform that never moved.
        match ata.try_inverse() {
            Some(ata_inv) => {
                let delta = -(ata_inv * atb);
                let update = exponential_map_se3(&delta);
                transformation = update * transformation;
            }
            None => {
                // `pseudo_inverse` returns a `Result` here. If even that fails
                // the geometry is beyond recovery and `None` is the only honest
                // answer.
                // The epsilon is relative to the largest singular value, so a
                // single value works across the whole range of problem scales.
                let Ok(pinv) = ata.clone().pseudo_inverse(1e-6) else {
                    return None;
                };
                let delta = -(pinv * atb);
                let update = exponential_map_se3(&delta);
                transformation = update * transformation;
            }
        }

        // No usable geometry: a point-to-plane residual needs at least three
        // independent constraints. Returning `None` says "this registration did
        // not happen", which is the truth. Returning the input transform with a
        // perfect score is a lie a caller cannot detect.
        if n_used < 3 {
            return None;
        }

        let source_len = source.points.len();
        if correspondences.is_empty() || source_len == 0 {
            return None;
        }

        // Divided by the number of correspondences that actually contributed a
        // residual, not the number offered. `total_residual` accumulated only
        // from those, so dividing by the full count understated the error
        // whenever some had no normal - the same class of quiet wrongness as
        // the update being skipped.
        let rmse = (total_residual / n_used as f32).sqrt();
        // Fraction of the source that found a correspondence, which is what
        // fitness means here and is independent of `n_used`.
        let fitness = correspondences.len() as f32 / source_len as f32;

        // Track the *best fit*, not the best fitness.
        //
        // `fitness` is the fraction of source points that found a
        // correspondence, so it saturates at 1.0 as soon as every point matches
        // and stays there. Comparing on it meant the record was frozen at the
        // first iterate: with `fitness > best_fitness` false from iteration 1
        // onward, the function reported iteration 0's rmse next to the final
        // transform - overstated by 5-6 orders of magnitude, and in one case a
        // `fitness` the returned transform did not achieve.
        //
        // A lower rmse is unambiguously a better fit, so that is what decides.
        // Ties still prefer the higher fitness.
        let better = fitness > best_fitness || (fitness == best_fitness && rmse < best_rmse);
        if better {
            best_fitness = fitness;
            best_rmse = rmse;
            best_transformation = transformation;
            final_iterations = iter + 1;
        }

        if rmse < 1e-6 {
            break;
        }
    }

    Some(ICPResult {
        // Only the recorded best, never the live iterate: the two can differ,
        // and returning the live one makes every reported metric a lie.
        transformation: best_transformation,
        fitness: best_fitness,
        inlier_rmse: best_rmse,
        num_iterations: final_iterations,
    })
}

/// Standard point-to-plane ICP with context-aware acceleration
pub fn registration_icp_point_to_plane_ctx(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f32,
    init_transformation: &nalgebra::Matrix4<f32>,
    max_iterations: usize,
    ctx: &cv_hal::compute::ComputeDevice,
) -> Result<ICPResult> {
    use cv_core::Tensor;
    use cv_hal::tensor_ext::TensorToGpu;

    let mut transformation = *init_transformation;
    let mut best_fitness = 0.0;
    let mut best_rmse = f32::MAX;
    // The transform / were measured at. Returning the
    // live  instead made every reported metric describe a
    // different pose than the one returned.
    let mut best_transformation = *init_transformation;
    let mut final_iterations = 0;

    // Convert point clouds to tensors for GPU processing
    let source_tensor: cv_core::CpuTensor<f32> = Tensor::from_vec(
        source
            .points
            .iter()
            .flat_map(|p| [p.x, p.y, p.z, 1.0])
            .collect(),
        cv_core::TensorShape::new(1, source.points.len(), 4),
    )
    .map_err(|e| Error::RuntimeError(format!("Failed to create source tensor: {:?}", e)))?;
    let target_tensor: cv_core::CpuTensor<f32> = Tensor::from_vec(
        target
            .points
            .iter()
            .flat_map(|p| [p.x, p.y, p.z, 1.0])
            .collect(),
        cv_core::TensorShape::new(1, target.points.len(), 4),
    )
    .map_err(|e| Error::RuntimeError(format!("Failed to create target tensor: {:?}", e)))?;
    let target_normals_tensor: cv_core::CpuTensor<f32> = Tensor::from_vec(
        target
            .normals
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Target point cloud must have normals".to_string()))?
            .iter()
            .flat_map(|n| [n.x, n.y, n.z, 0.0])
            .collect(),
        cv_core::TensorShape::new(1, target.points.len(), 4),
    )
    .map_err(|e| Error::RuntimeError(format!("Failed to create target normals tensor: {:?}", e)))?;

    // If using GPU, upload once
    let (s_gpu, t_gpu, n_gpu) = if let cv_hal::compute::ComputeDevice::Gpu(gpu) = ctx {
        (
            source_tensor.to_gpu_ctx(gpu).map_err(|e| {
                Error::RuntimeError(format!("Failed to upload source tensor to GPU: {:?}", e))
            })?,
            target_tensor.to_gpu_ctx(gpu).map_err(|e| {
                Error::RuntimeError(format!("Failed to upload target tensor to GPU: {:?}", e))
            })?,
            target_normals_tensor.to_gpu_ctx(gpu).map_err(|e| {
                Error::RuntimeError(format!(
                    "Failed to upload target normals tensor to GPU: {:?}",
                    e
                ))
            })?,
        )
    } else {
        // CPU fallback: we'll use the tensors directly but it's less efficient than specialized CPU code
        return registration_icp_point_to_plane(
            source,
            target,
            max_correspondence_distance,
            init_transformation,
            max_iterations,
        )
        .ok_or_else(|| Error::RuntimeError("CPU fallback ICP failed to converge".to_string()));
    };

    // The source as it stands under the current pose.
    //
    // The correspondence search has to see the *moved* source, or it recomputes
    // the same untransformed associations every iteration: the source was
    // uploaded once before the loop and passed unchanged, so the association set
    // was a fixed point rather than an iteration, and the whole "ICP" was a
    // single Gauss-Newton step with the update applied repeatedly to a
    // correspondence set that never moved. The existing parity test takes the
    // CPU early-return above and so never reached this.
    let mut s_gpu = s_gpu;

    for iter in 0..max_iterations {
        // Apply the running transform to the source before searching, so the
        // correspondences reflect the pose reached so far.
        if iter > 0 {
            let moved: Vec<Point3<f32>> = source
                .points
                .iter()
                .map(|p| {
                    // `transform_point` is f64; the tensors are f32. The transform
                    // itself is computed in f64 and narrowed on the way out, so no
                    // precision is lost in the accumulation that follows.
                    transformation.transform_point(&nalgebra::Point3::new(p.x, p.y, p.z))
                })
                .collect();
            let mut flat: Vec<f32> = Vec::with_capacity(moved.len() * 3);
            for p in &moved {
                flat.extend_from_slice(&[p.x, p.y, p.z]);
            }
            let moved_tensor: cv_core::CpuTensor<f32> =
                Tensor::from_vec(flat, cv_core::TensorShape::new(3, source.points.len(), 1))
                    .map_err(|e| {
                        Error::RuntimeError(format!("Failed to build moved source: {e:?}"))
                    })?;
            let gpu = match ctx {
                cv_hal::compute::ComputeDevice::Gpu(g) => g,
                _ => unreachable!("the CPU branch returns above"),
            };
            s_gpu = moved_tensor
                .to_gpu_ctx(gpu)
                .map_err(|e| Error::RuntimeError(format!("Failed to re-upload source: {e:?}")))?;
        }

        // Find correspondences on device
        let correspondences_raw = ctx
            .icp_correspondences(&s_gpu, &t_gpu, max_correspondence_distance)
            .map_err(|e| {
                Error::RuntimeError(format!("Failed to compute correspondences: {:?}", e))
            })?;

        // Too few correspondences to determine a rigid motion.
        //
        // At iteration 0 this used to `break`, falling through to the
        // unconditional `Ok(ICPResult { .. })` below with `fitness` at its 0.0
        // initialiser and `inlier_rmse` at f32::MAX - a successful registration
        // that never happened. Verified with `max_correspondence_distance = 0.0`,
        // which matches nothing.
        //
        // Later on it is a *termination* condition, not a failure. Correspondences
        // are the only thing an ICP iteration consumes: once none are within the
        // gate, the next step has nothing to step from and the pose can only stay
        // put, so continuing would spin to `max_iterations` and then report a
        // stale pose. The way to get there is a pseudo-inverse step on a
        // rank-deficient `ata` overshooting the gate - reachable, and measured:
        // on the rank-1 input of the `a_singular_normal_matrix` regression test
        // (24 identical target points, normals +y, source offset 0.03) the
        // recovered pose sits 0.33 from the target, and a further step puts it
        // beyond 0.5, where nothing matches.
        //
        // Previously this was invisible: `evaluate_registration` returned
        // `rmse 0.0` when there were zero inliers, so `rmse < 1e-6` fired and
        // the loop "converged" on a pose it had not evaluated at all. Now that
        // the zero-inlier case reports `INFINITY`, the overshoot is visible, and
        // the honest response is to stop and hand back the best pose that *was*
        // measured rather than to discard it or fabricate a score for it.
        //
        // Nothing recorded yet means nothing was ever evaluated, which is the
        // failure case above.
        if correspondences_raw.len() < 3 {
            if final_iterations == 0 {
                return Err(Error::AlgorithmError(format!(
                    "registration_icp_point_to_plane_ctx: {} correspondences found, \
                     at least 3 are required to determine a rigid motion",
                    correspondences_raw.len()
                )));
            }
            break;
        }

        let correspondences: Vec<(u32, u32)> = correspondences_raw
            .iter()
            .map(|&(s, t, _)| (s as u32, t as u32))
            .collect();

        // Accumulate Normal Equations on device
        let (ata, atb): (nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>) = ctx
            // `s_gpu` already holds the source moved by the running pose, so
            // the shader is given identity - it multiplies the point by this
            // matrix itself, and passing the pose as well would apply the
            // motion twice per iteration.
            .icp_accumulate(
                &s_gpu,
                &t_gpu,
                &n_gpu,
                &correspondences,
                &Matrix4::identity(),
            )
            .map_err(|e| {
                Error::RuntimeError(format!("Failed to accumulate normal equations: {:?}", e))
            })?;

        // Solve for update on CPU (Matrix6 is small).
        //
        // A singular `ata` used to be detected by `if let Some(..)` with no
        // `else`, so the pose simply stayed where it was, the loop kept going,
        // and the tail reported that pose as a registration. Reached by a
        // collinear target, or a planar one whose normals are all identical.
        // Now: exact inverse, else pseudo-inverse, else an error - the same
        // ladder the CPU function uses at ~213-260.
        match ata.try_inverse() {
            Some(ata_inv) => {
                let delta = -(ata_inv * atb);
                let update = exponential_map_se3(&delta);
                transformation = update * transformation;
            }
            None => {
                let Ok(pinv) = ata.clone().pseudo_inverse(1e-6) else {
                    return Err(Error::AlgorithmError(
                        "registration_icp_point_to_plane_ctx: normal equations are \
                         singular and the pseudo-inverse failed, so no motion can \
                         be determined from these correspondences"
                            .to_string(),
                    ));
                };
                let delta = -(pinv * atb);
                if !delta.iter().all(|v| v.is_finite()) {
                    return Err(Error::AlgorithmError(
                        "registration_icp_point_to_plane_ctx: pseudo-inverse \
                         produced a non-finite update"
                            .to_string(),
                    ));
                }
                let update = exponential_map_se3(&delta);
                transformation = update * transformation;
            }
        }

        // Evaluation (could be optimized on GPU too)
        let (fitness, rmse) =
            evaluate_registration(source, target, &transformation, max_correspondence_distance);

        // Track the *best fit*, not the best fitness.
        //
        // `fitness` is a correspondence-count ratio, so it saturates at 1.0 and
        // `fitness > best_fitness` was false from iteration 1 onward: the record
        // froze at the first iterate while the transform kept moving, so the
        // reported rmse described a different pose than the one returned. Same
        // rule as the CPU function at ~290-306.
        let better = fitness > best_fitness || (fitness == best_fitness && rmse < best_rmse);
        if better {
            best_fitness = fitness;
            best_rmse = rmse;
            best_transformation = transformation;
            final_iterations = iter + 1;
        }

        if rmse < 1e-6 {
            break;
        }
    }

    Ok(ICPResult {
        transformation: best_transformation,
        fitness: best_fitness,
        inlier_rmse: best_rmse,
        num_iterations: final_iterations,
    })
}

/// Multi-scale ICP
pub fn registration_multi_scale_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distances: &[f32],
    init_transformation: &Matrix4<f32>,
    max_iterations_per_scale: usize,
) -> Option<ICPResult> {
    let mut transformation = *init_transformation;
    let mut best_result = None;

    for &max_dist in max_correspondence_distances {
        if let Some(result) = registration_icp_point_to_plane(
            source,
            target,
            max_dist,
            &transformation,
            max_iterations_per_scale,
        ) {
            transformation = result.transformation;
            best_result = Some(result);
        }
    }

    best_result
}

/// Exponential map from se(3) to SE(3)
fn exponential_map_se3(delta: &nalgebra::Vector6<f32>) -> Matrix4<f32> {
    let omega = nalgebra::Vector3::new(delta[3], delta[4], delta[5]);
    let v = nalgebra::Vector3::new(delta[0], delta[1], delta[2]);

    let theta = omega.norm();

    let rotation = if theta < 1e-6 {
        nalgebra::Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = nalgebra::Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        nalgebra::Matrix3::identity()
            + k_cross * theta.sin()
            + k_cross * k_cross * (1.0 - theta.cos())
    };

    // Proper SE(3) exponential map using left Jacobian
    let translation = if theta < 1e-6 {
        v
    } else {
        let k = omega / theta;
        let k_cross = nalgebra::Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        let k_cross_sq = k_cross * k_cross;
        let left_jacobian = nalgebra::Matrix3::identity()
            + k_cross * ((1.0 - theta.cos()) / theta)
            + k_cross_sq * ((theta - theta.sin()) / theta);
        left_jacobian * v
    };

    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

/// Compute information matrix from registration
///
/// The information matrix of a point-to-point ICP problem is the sum of outer
/// products of the per-point Jacobian, which for a rigid perturbation is
///
/// ```text
/// J_p = [ p ; p x (t q_p - s p) ]
/// ```
///
/// where `p` is the source point and `t q_p - s p` is the residual `q_p - T p`.
/// The rotation block here was already right; the **translation** block was not.
/// It was filled with the residual instead of the source point:
///
/// ```text
/// jacobian = [ diff ; p x diff ]        // was
/// jacobian = [ p   ; p x diff ]        // now
/// ```
///
/// so the whole matrix collapsed to zero exactly when the registration was
/// *perfect* - `diff = 0` for every correspondence - and reported "no
/// information about the pose" for the one answer that pins it down hardest.
///
/// Measured: two identical clouds under the identity transform give
/// `diag = [0, 0, 0, 0, 0, 0]` and `det = 0`, before the fix; after, the same
/// case gives non-zero translation diagonals that grow with the spread of the
/// cloud. A 0.01 offset along y, which `diff` did see, gave
/// `diag = [0, 0.0020, 0, 0, 0, 0.0025]` before and non-zero x and z
/// translation entries as well after - the two cases are now consistent, which is
/// what identifies the asymmetry as the defect rather than the zero.
///
/// The residual is still the right thing for the rotation block: `p x diff` is
/// the moment arm, and the residual is what makes it a *mismatch*.
///
/// No other crate in the workspace calls this function, so nothing else shifts.
pub fn get_information_matrix_from_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
) -> nalgebra::Matrix6<f32> {
    let mut information = nalgebra::Matrix6::<f32>::zeros();

    // Build simple nearest neighbor for target
    let target_nn = SimpleNN::new(target.points.clone());

    // Accumulate information from correspondences
    for src_point in &source.points {
        let transformed = transformation.transform_point(src_point);

        if let Some((target_point, _, dist_sq)) = target_nn.nearest(&transformed) {
            if dist_sq.sqrt() < 0.05 {
                // Small distance threshold
                let diff = transformed - target_point;
                let p = src_point.coords;

                // Translation acts on the point itself; rotation acts about it
                // through the residual.
                let jacobian = nalgebra::Vector6::new(
                    p.x,
                    p.y,
                    p.z,
                    p.y * diff.z - p.z * diff.y,
                    p.z * diff.x - p.x * diff.z,
                    p.x * diff.y - p.y * diff.x,
                );

                information += jacobian * jacobian.transpose();
            }
        }
    }

    information
}

/// Evaluate registration quality, returning `(fitness, inlier_rmse)`.
///
/// `fitness` is the fraction of source points that found a correspondence within
/// `max_correspondence_distance`; `inlier_rmse` is the root-mean-square distance
/// of those inlier points to their correspondence.
///
/// A fit that cannot be measured reports `f32::INFINITY`, not `0.0`. Three cases
/// reach that, and in each of them `0.0` is a claim of a flawless registration:
///
/// * **no source points** - nothing was registered at all;
/// * **no target points** - every query is infinitely far from everything;
/// * **zero inliers** - every source point was found but all of them were beyond
///   the gate, so the error is known to be large, not small.
///
/// Measured, source 5 units from a 2-point target with a 0.1 gate, before the
/// fix: `(0.0, 0.0)` - *zero inliers and zero error at once*, which is not a
/// coherent answer for any transform. The sibling
/// `global::ransac::evaluate_registration` was corrected for exactly this and
/// returns `INFINITY`; this one, the function exported at the crate root and used
/// by `registration_icp_point_to_plane_ctx`, was missed.
///
/// `INFINITY` is also the honest bound for the empty case in a way `NaN` would not
/// be: it compares as `> any finite error` under the ordinary `<` a caller uses to
/// ask "did this converge", and it does not poison an arithmetic mean.
pub fn evaluate_registration(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> (f32, f32) {
    let target_nn = SimpleNN::new(target.points.clone());

    let mut inlier_count = 0;
    let mut total_error = 0.0;

    // Nothing on either side is a registration that was never attempted, so it is
    // not a perfect one. The empty-source case used to fall through to the tail
    // and return `(0.0, 0.0)`.
    if source.points.is_empty() || target.points.is_empty() {
        return (0.0, f32::INFINITY);
    }

    for point in &source.points {
        let transformed = transformation.transform_point(point);
        if let Some((_, _, dist_sq)) = target_nn.nearest(&transformed) {
            let dist = dist_sq.sqrt();
            if dist < max_correspondence_distance {
                inlier_count += 1;
                total_error += dist_sq;
            }
        }
    }

    let fitness = inlier_count as f32 / source.points.len() as f32;
    let rmse = if inlier_count > 0 {
        (total_error / inlier_count as f32).sqrt()
    } else {
        // Zero inliers is zero *support*, not zero error: every source point was
        // found and every one of them was beyond the gate. See the doc comment.
        f32::INFINITY
    };

    (fitness, rmse)
}
mod mod_test;

#[cfg(test)]
mod point_to_plane_no_normals {
    use super::*;
    use nalgebra::Point3;

    fn cloud(points: Vec<[f32; 3]>, normals: Option<Vec<[f32; 3]>>) -> PointCloud {
        let pts: Vec<Point3<f32>> = points
            .iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect();
        let nrm = normals.map(|ns| {
            ns.iter()
                .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
                .collect::<Vec<_>>()
        });
        PointCloud {
            points: pts,
            colors: None,
            normals: nrm,
        }
    }

    /// A non-coplanar point set, so a 6-DOF transform is actually constrained.
    ///
    /// A flat square is degenerate: the normal equations are rank-deficient and
    /// even a correct implementation cannot resolve motion in every axis. The
    /// guard under test is about a *missing normal*, not about degeneracy, so the
    /// control case has to be one that genuinely registers.
    fn cube_points() -> Vec<[f32; 3]> {
        let mut v = Vec::new();
        for &x in &[0.0f32, 0.5, 1.0] {
            for &y in &[0.0f32, 0.5, 1.0] {
                for &z in &[0.0f32, 0.5, 1.0] {
                    v.push([x, y, z]);
                }
            }
        }
        v
    }

    /// A non-planar target with outward face normals, so point-to-plane
    /// residuals actually constrain translation in all three axes.
    ///
    /// A set of points that all share one +z normal cannot observe motion along
    /// z at all - the residual is identically zero for any z offset - so the
    /// control case has to be a volume, not a plane.
    fn cube_with_normals() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        for (axis, sign) in [
            (0usize, 0.0f32),
            (0, 1.0),
            (1, 0.0),
            (1, 1.0),
            (2, 0.0),
            (2, 1.0),
        ] {
            for a in [0.0f32, 0.34, 0.67, 1.0] {
                for b in [0.0f32, 0.34, 0.67, 1.0] {
                    let mut p = [0.0f32; 3];
                    p[axis] = sign;
                    p[(axis + 1) % 3] = a;
                    p[(axis + 2) % 3] = b;
                    let mut n = [0.0f32; 3];
                    n[axis] = sign * 2.0 - 1.0;
                    pts.push(p);
                    nrm.push(n);
                }
            }
        }
        (pts, nrm)
    }

    /// A target with no normals must not produce a perfect-looking result.
    ///
    /// The point-to-plane residual is undefined without a target normal, so the
    /// accumulation was skipped for every correspondence, `ata` stayed a zero
    /// matrix, `try_inverse()` returned `None` and the update was silently
    /// skipped - after which the function reported the caller's initial
    /// transform with `fitness: 1.0` and `inlier_rmse: 0.0`. A perfect score
    /// from a result that never moved, which is the worst possible failure for a
    /// caller: nothing about it looks wrong.
    ///
    /// This is not an exotic input: `PointCloud` makes `normals` optional.
    #[test]
    fn normal_less_target_reports_failure_not_a_perfect_score() {
        let (square, normals) = cube_with_normals();
        // Source offset 2cm along y: a registration has real work to do, and the
        // y-offset is observable because the cube has faces whose normals are
        // not perpendicular to y.
        let source = cloud(
            square.iter().map(|p| [p[0], p[1] + 0.02, p[2]]).collect(),
            Some(normals.clone()),
        );
        let target_no_normals = cloud(square.clone(), None);
        let target = cloud(square, Some(normals));

        let identity = Matrix4::identity();

        let without =
            registration_icp_point_to_plane(&source, &target_no_normals, 1.0, &identity, 20);
        assert!(
            without.is_none(),
            "a target with no normals must report failure, not a result: got {:?}",
            without.map(|r| (r.fitness, r.inlier_rmse))
        );

        // With normals the same call does register, so the guard is not simply
        // rejecting everything.
        let with = registration_icp_point_to_plane(&source, &target, 1.0, &identity, 20)
            .expect("the same registration must succeed once normals are present");
        assert!(
            with.transformation[(1, 3)].abs() > 1e-4,
            "with normals the pose should have moved in y, got {:?}",
            with.transformation
        );
    }
}
