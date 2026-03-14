//! Colored ICP Registration
//!
//! ICP variant that uses both geometry and color for alignment.
//! Useful for registering RGBD scans with rich texture.

use cv_core::point_cloud::PointCloud;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// Simple nearest neighbor
struct SimpleNN {
    points: Vec<Point3<f32>>,
}

impl SimpleNN {
    fn new(points: Vec<Point3<f32>>) -> Self {
        Self { points }
    }

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
                let normal = target.normals.as_ref()
                    .map(|n| n[target_idx])
                    .unwrap_or_else(|| {
                        let diff = transformed - target_point;
                        diff.normalize()
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

        if valid_points < 10 {
            break;
        }

        // Solve for update
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);

            // Convert delta to transformation update
            let update = exponential_map(&delta);
            transformation = update * transformation;
        }

        // Track best
        let rmse = (total_residual / valid_points as f32).sqrt();
        let fitness = valid_points as f32 / source.points.len() as f32;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_rmse = rmse;
        }

        // Convergence check
        if iter > 0 && rmse < 0.001 {
            break;
        }
    }

    Some(ColoredICPResult {
        transformation,
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

/// Compute photometric jacobian (simplified)
fn compute_photometric_jacobian(
    _point: &Point3<f32>,
    _source_color: &Point3<f32>,
    _target_color: &Point3<f32>,
) -> nalgebra::Vector6<f32> {
    // Simplified - would need image gradient in practice
    nalgebra::Vector6::new(0.01, 0.01, 0.01, 0.0, 0.0, 0.0)
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
