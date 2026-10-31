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
pub(crate) struct SimpleNN {
    tree: cv_3d::spatial::KDTree<usize>,
}

impl SimpleNN {
    pub(crate) fn new(points: Vec<Point3<f32>>) -> Self {
        let mut items: Vec<_> = points.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let tree = cv_3d::spatial::KDTree::build(&mut items);
        Self { tree }
    }

    pub(crate) fn nearest(&self, query: &Point3<f32>) -> Option<(Point3<f32>, usize, f32)> {
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
/// * `inlier_rmse` - Point-to-point RMSE over the inlier correspondences, as
///   Open3D reports it (equal to [`evaluate_registration`] on `transformation`)
/// * `num_iterations` - Number of iterations performed before convergence
#[derive(Debug, Clone)]
pub struct ICPResult {
    /// Optimized 4×4 homogeneous transformation matrix (SE(3))
    pub transformation: Matrix4<f32>,
    /// Registration fitness score: fraction of points with valid correspondences (0-1)
    pub fitness: f32,
    /// Point-to-point RMSE of the inlier correspondences at `transformation`
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
    if source.points.is_empty() || target.points.is_empty() {
        return None;
    }
    let target_nn = SimpleNN::new(target.points.clone());
    let source_len = source.points.len();

    // The loop follows Open3D's `RegistrationICP`: evaluate at the current pose,
    // estimate an update from *those* correspondences, apply it, re-evaluate at
    // the new pose, and stop once fitness and RMSE both change by less than
    // `RELATIVE_FITNESS` / `RELATIVE_RMSE`.
    //
    // Previously fitness and rmse were accumulated while linearising - at the
    // pose *before* the update - but stored next to the pose *after* it, and the
    // rmse was the point-to-plane residual. Neither matched
    // `evaluate_registration` on the returned transform (measured on a wavy
    // surface after one iteration: reported 1.43e-2, actual 1.03e-6), and the only stop
    // was `rmse < 1e-6`, which noisy data never reaches.
    let mut transformation = *init_transformation;
    let mut correspondences = find_correspondences(
        source,
        &target_nn,
        &transformation,
        max_correspondence_distance,
    );

    // Too few correspondences to constrain a rigid transform. Returning `None`
    // is what this function's documentation promises for "too few
    // correspondences"; a result with `fitness 0.0` and the input transform
    // unchanged reads as a successful registration that found nothing.
    if correspondences.len() < 3 {
        return None;
    }
    let (mut fitness, mut rmse) = fitness_and_rmse(&correspondences, source_len);

    // Every candidate below is a pose together with the metrics measured *at
    // that pose*, so whichever is returned, its numbers describe it.
    //
    // The best rather than the last is kept as a guard: a pseudo-inverse step
    // on a rank-deficient system can overshoot (see the `_ctx` sibling), and
    // that must not discard a better pose that was already measured. Ties in
    // fitness go to the lower rmse, since fitness saturates at 1.0.
    let mut best = (transformation, fitness, rmse);
    let mut iterations = 0;

    for iter in 0..max_iterations {
        // Point-to-plane normal equations at the current pose.
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();

        // Correspondences that actually contribute geometry. A target without
        // normals gives none, `ata` stays zero, and the solve would leave the
        // pose untouched while the metrics reported a perfect fit. A target
        // cloud with no normals is the *normal* case, since `PointCloud`
        // makes the field optional, so it is reported as a failure.
        let mut n_used = 0usize;
        if let Some(normals) = target.normals.as_ref() {
            for &(src_idx, tgt_idx, _) in &correspondences {
                n_used += 1;
                let (jacobian, residual) = point_to_plane_row(
                    &transformation.transform_point(&source.points[src_idx]),
                    &target.points[tgt_idx],
                    &normals[tgt_idx],
                );
                ata += jacobian * jacobian.transpose();
                atb += jacobian * residual;
            }
        }
        if n_used < 3 {
            return None;
        }

        // A singular `A` means the correspondences cannot determine a rigid
        // motion - a collinear target, a planar one with identical normals.
        // The pseudo-inverse recovers the solvable subspace; if even that fails,
        // `None` - never a perfect score for a transform that never moved.
        let delta = match ata.try_inverse() {
            Some(ata_inv) => -(ata_inv * atb),
            None => {
                let Ok(pinv) = ata.clone().pseudo_inverse(1e-6) else {
                    return None;
                };
                -(pinv * atb)
            }
        };
        if !delta.iter().all(|v| v.is_finite()) {
            return None;
        }
        transformation = exponential_map_se3(&delta) * transformation;
        iterations = iter + 1;

        // Re-evaluate at the pose just reached.
        let next = find_correspondences(
            source,
            &target_nn,
            &transformation,
            max_correspondence_distance,
        );
        if next.len() < 3 {
            // The step overshot the gate: nothing left to step from. Keep the
            // best pose that was measured rather than inventing a score.
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

    Some(ICPResult {
        transformation: best.0,
        fitness: best.1,
        inlier_rmse: best.2,
        num_iterations: iterations,
    })
}

/// Open3D's default `ICPConvergenceCriteria::relative_fitness`.
const RELATIVE_FITNESS: f32 = 1e-6;
/// Open3D's default `ICPConvergenceCriteria::relative_rmse`.
const RELATIVE_RMSE: f32 = 1e-6;

/// One point-to-plane row for a source point already moved to `p`: the
/// Jacobian with respect to a left-multiplied twist `[v; w]` (translation
/// first, as consumed by [`exponential_map_se3`]) and the signed residual
/// `(p - q) . n`.
///
/// For `p' = exp([v; w]) p ~ p + w x p + v`, the residual changes by
/// `n . v + n . (w x p) = n . v + w . (p x n)`, hence `J = [n; p x n]`.
pub(crate) fn point_to_plane_row(
    p: &Point3<f32>,
    q: &Point3<f32>,
    n: &nalgebra::Vector3<f32>,
) -> (nalgebra::Vector6<f32>, f32) {
    let cross = p.coords.cross(n);
    (
        nalgebra::Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z),
        (p - q).dot(n),
    )
}

/// Nearest-neighbour correspondences of `source` under `transformation`, as
/// `(source index, target index, squared distance)`, keeping those strictly
/// closer than `max_correspondence_distance`. Ordered by source index, so
/// every consumer sums in the same order and gets bit-identical metrics.
pub(crate) fn find_correspondences(
    source: &PointCloud,
    target_nn: &SimpleNN,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> Vec<(usize, usize, f32)> {
    let find = |(i, p): (usize, &Point3<f32>)| {
        let moved = transformation.transform_point(p);
        target_nn
            .nearest(&moved)
            .filter(|&(_, _, d2)| d2.sqrt() < max_correspondence_distance)
            .map(|(_, j, d2)| (i, j, d2))
    };
    if source.points.len() > 5000 {
        source
            .points
            .par_iter()
            .enumerate()
            .filter_map(find)
            .collect()
    } else {
        source.points.iter().enumerate().filter_map(find).collect()
    }
}

/// Open3D's registration metrics for a correspondence set: `fitness` is the
/// fraction of source points with a correspondence, `inlier_rmse` the
/// point-to-point RMSE over them (`INFINITY` when there are none - see
/// [`evaluate_registration`]).
pub(crate) fn fitness_and_rmse(
    correspondences: &[(usize, usize, f32)],
    source_len: usize,
) -> (f32, f32) {
    if source_len == 0 {
        return (0.0, f32::INFINITY);
    }
    let fitness = correspondences.len() as f32 / source_len as f32;
    if correspondences.is_empty() {
        return (fitness, f32::INFINITY);
    }
    let total: f32 = correspondences.iter().map(|c| c.2).sum();
    (fitness, (total / correspondences.len() as f32).sqrt())
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
    let mut prev_metrics: Option<(f32, f32)> = None;

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

        // Open3D's relative criteria, as in the CPU function.
        let converged = prev_metrics.is_some_and(|(f, r): (f32, f32)| {
            (fitness - f).abs() < RELATIVE_FITNESS && (rmse - r).abs() < RELATIVE_RMSE
        });
        prev_metrics = Some((fitness, rmse));
        if converged || rmse < 1e-6 {
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

/// Information matrix of a registration, as Open3D's
/// `GetInformationMatrixFromPointClouds`.
///
/// For every source point `p` whose transformed position `T p` has a nearest
/// target point `q` closer than `max_correspondence_distance`, the
/// correspondence contributes `G^T G` with
///
/// ```text
/// G = [ -[q]x | I3 ]        (3 x 6)
/// ```
///
/// i.e. the Jacobian of `T p` with respect to a left-multiplied twist, linearised
/// at the target point. The twist ordering is Open3D's, **rotation first**:
/// `H[0..3, 0..3]` is the rotation block, `H[3..6, 3..6]` the translation block
/// (`N * I3`), and the coupling block `H[0..3, 3..6] = sum [q]x` follows the
/// centroid of the matched target points (zero for a cloud centred on the
/// origin), exactly as in Open3D. This is the matrix pose-graph edges expect.
///
/// The result depends only on *which* points correspond, not on their residual:
/// the information content of a correspondence set is the same whether or not
/// the alignment is perfect.
///
/// Previously the translation rows held the source coordinates and the rotation
/// rows were multiplied by the residual, with a hard-coded 0.05 gate. Two
/// identical clouds under the identity gave `diag [1043, 1043, 1043, 0, 0, 0]` -
/// no rotation information at the alignment that pins rotation hardest, and a
/// translation block that grew with the distance from the origin.
pub fn get_information_matrix_from_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> nalgebra::Matrix6<f32> {
    let mut information = nalgebra::Matrix6::<f64>::zeros();
    if source.points.is_empty() || target.points.is_empty() {
        return information.cast();
    }

    let target_nn = SimpleNN::new(target.points.clone());

    for src_point in &source.points {
        let transformed = transformation.transform_point(src_point);
        let Some((q, _, dist_sq)) = target_nn.nearest(&transformed) else {
            continue;
        };
        if dist_sq.sqrt() >= max_correspondence_distance {
            continue;
        }
        let (x, y, z) = (f64::from(q.x), f64::from(q.y), f64::from(q.z));
        // Rows of G = [-[q]x | I], written out.
        let rows = [
            nalgebra::Vector6::new(0.0, z, -y, 1.0, 0.0, 0.0),
            nalgebra::Vector6::new(-z, 0.0, x, 0.0, 1.0, 0.0),
            nalgebra::Vector6::new(y, -x, 0.0, 0.0, 0.0, 1.0),
        ];
        for g in &rows {
            information += g * g.transpose();
        }
    }

    information.cast()
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
    // Nothing on either side is a registration that was never attempted, so it is
    // not a perfect one. The empty-source case used to fall through to the tail
    // and return `(0.0, 0.0)`.
    if source.points.is_empty() || target.points.is_empty() {
        return (0.0, f32::INFINITY);
    }
    let target_nn = SimpleNN::new(target.points.clone());
    // Same correspondence search and the same reduction the ICP loops report
    // through, so their metrics and this function agree exactly. Zero inliers
    // is zero *support*, not zero error: `INFINITY` (see the doc comment).
    let correspondences = find_correspondences(
        source,
        &target_nn,
        transformation,
        max_correspondence_distance,
    );
    fitness_and_rmse(&correspondences, source.points.len())
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
