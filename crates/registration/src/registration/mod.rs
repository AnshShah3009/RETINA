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
pub use gnc::{registration_gnc, GNCOptimizer, GNCResult};

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

        if correspondences.len() < 3 {
            break;
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
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);

            // Update transformation using exponential map
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
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

        if fitness > best_fitness {
            best_fitness = fitness;
            best_rmse = rmse;
            final_iterations = iter + 1;
        }

        if rmse < 1e-6 {
            break;
        }
    }

    Some(ICPResult {
        transformation,
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

    for iter in 0..max_iterations {
        // Find correspondences on device
        let correspondences_raw = ctx
            .icp_correspondences(&s_gpu, &t_gpu, max_correspondence_distance)
            .map_err(|e| {
                Error::RuntimeError(format!("Failed to compute correspondences: {:?}", e))
            })?;

        if correspondences_raw.len() < 3 {
            break;
        }

        let correspondences: Vec<(u32, u32)> = correspondences_raw
            .iter()
            .map(|&(s, t, _)| (s as u32, t as u32))
            .collect();

        // Accumulate Normal Equations on device
        let (ata, atb): (nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>) = ctx
            .icp_accumulate(&s_gpu, &t_gpu, &n_gpu, &correspondences, &transformation)
            .map_err(|e| {
                Error::RuntimeError(format!("Failed to accumulate normal equations: {:?}", e))
            })?;

        // Solve for update on CPU (Matrix6 is small)
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
        }

        // Evaluation (could be optimized on GPU too)
        let (fitness, rmse) =
            evaluate_registration(source, target, &transformation, max_correspondence_distance);

        if fitness > best_fitness {
            best_fitness = fitness;
            best_rmse = rmse;
            final_iterations = iter + 1;
        }

        if rmse < 1e-6 {
            break;
        }
    }

    Ok(ICPResult {
        transformation,
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

                // Compute Jacobian (simplified)
                let p = src_point.coords;
                let jacobian = nalgebra::Vector6::new(
                    diff.x,
                    diff.y,
                    diff.z,
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

/// Evaluate registration
pub fn evaluate_registration(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> (f32, f32) {
    let target_nn = SimpleNN::new(target.points.clone());

    let mut inlier_count = 0;
    let mut total_error = 0.0;

    for point in &source.points {
        let transformed = transformation.transform_point(point);
        if let Some((_, _, dist_sq)) = target_nn.nearest(&transformed) {
            let dist = dist_sq.sqrt();
            if dist < max_correspondence_distance {
                inlier_count += 1;
                total_error += dist * dist;
            }
        }
    }

    let fitness = if !source.points.is_empty() {
        inlier_count as f32 / source.points.len() as f32
    } else {
        0.0
    };

    let rmse = if inlier_count > 0 {
        (total_error / inlier_count as f32).sqrt()
    } else {
        0.0
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
