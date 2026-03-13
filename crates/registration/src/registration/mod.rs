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

/// Simple nearest-neighbor search structure for point cloud registration
///
/// Provides O(N) linear search for nearest neighbors. While not optimal
/// for large point clouds, it's suitable for smaller datasets and enables
/// straightforward point-to-plane ICP implementation.
struct SimpleNN {
    /// Reference points for neighbor queries
    points: Vec<Point3<f32>>,
}

impl SimpleNN {
    /// Create a new nearest-neighbor index from points
    ///
    /// # Arguments
    ///
    /// * `points` - Vector of 3D points to build index from
    fn new(points: Vec<Point3<f32>>) -> Self {
        Self { points }
    }

    /// Find the nearest point to a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    ///
    /// # Returns
    ///
    /// * `Some((point, index, distance_squared))` - Nearest point and its index
    /// * `None` - If point set is empty
    fn nearest(&self, query: &Point3<f32>) -> Option<(Point3<f32>, usize, f32)> {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;

        for (i, pt) in self.points.iter().enumerate() {
            let dist = (pt - query).norm_squared();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        if min_dist < f32::MAX {
            Some((self.points[min_idx], min_idx, min_dist))
        } else {
            None
        }
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
        let mut correspondences: Vec<(usize, usize, f32)> = Vec::new();

        // Find correspondences
        for (src_idx, src_point) in source.points.iter().enumerate() {
            let transformed = transformation.transform_point(src_point);

            if let Some((_target_point, target_idx, dist_sq)) = target_nn.nearest(&transformed) {
                let dist = dist_sq.sqrt();

                if dist <= max_correspondence_distance {
                    correspondences.push((src_idx, target_idx, dist));
                }
            }
        }

        if correspondences.len() < 3 {
            break;
        }

        // Compute point-to-plane error
        let mut ata = nalgebra::Matrix6::<f32>::zeros();
        let mut atb = nalgebra::Vector6::<f32>::zeros();
        let mut total_residual = 0.0;

        for (src_idx, tgt_idx, _) in &correspondences {
            let src_point = source.points[*src_idx];
            let tgt_point = target.points[*tgt_idx];
            let tgt_normal = target.normals.as_ref().map(|n| &n[*tgt_idx]);

            if let Some(normal) = tgt_normal {
                let transformed = transformation.transform_point(&src_point);
                let diff = transformed - tgt_point;
                let residual = diff.dot(normal);

                // Compute Jacobian
                let p = src_point.coords;
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
            let delta = ata_inv * atb;

            // Update transformation using exponential map
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
        }

        let source_len = source.points.len();
        if correspondences.is_empty() || source_len == 0 {
            return None;
        }

        let rmse = (total_residual / correspondences.len() as f32).sqrt();
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
            let delta = ata_inv * atb;
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
            + k_cross_sq * ((theta - theta.sin()) / (theta * theta));
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
