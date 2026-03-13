//! Graduated Non-Convexity (GNC) Robust Optimization
//!
//! GNC is a robust optimization technique that gradually transforms a convex
//! surrogate into the target robust cost function. This allows optimization
//! to escape local minima and effectively reject outliers.
//!
//! Based on "Graduated Non-Convexity for Robust Spatial Perception" by Yang et al.

use cv_core::RobustLoss;
use nalgebra::{Matrix4, Point3};

/// GNC optimizer for robust registration
pub struct GNCOptimizer {
    pub loss: RobustLoss,
    pub max_iterations: usize,
    pub gnc_iterations: usize,
    pub convergence_threshold: f32,
    pub geometric_cost_threshold: f32,
}

impl GNCOptimizer {
    /// Create new GNC optimizer with Geman-McClure loss
    pub fn new_geman_mcclure(max_residual: f32) -> Self {
        Self {
            loss: RobustLoss::GemanMcClure {
                mu: max_residual * max_residual,
            },
            max_iterations: 50,
            gnc_iterations: 10,
            convergence_threshold: 1e-6,
            geometric_cost_threshold: 1e-4,
        }
    }

    /// Create new GNC optimizer with Truncated Least Squares (best for outlier rejection)
    pub fn new_tls(max_residual: f32) -> Self {
        Self {
            loss: RobustLoss::TruncatedLeastSquares { c: max_residual },
            max_iterations: 50,
            gnc_iterations: 10,
            convergence_threshold: 1e-6,
            geometric_cost_threshold: 1e-4,
        }
    }

    /// Create new GNC optimizer with Welsch loss
    pub fn new_welsch(max_residual: f32) -> Self {
        Self {
            loss: RobustLoss::Welsch {
                mu: max_residual * max_residual,
            },
            max_iterations: 50,
            gnc_iterations: 10,
            convergence_threshold: 1e-6,
            geometric_cost_threshold: 1e-4,
        }
    }

    /// Solve robust registration using GNC
    pub fn solve_registration(
        &mut self,
        source: &[Point3<f32>],
        target: &[Point3<f32>],
        correspondences: &[(usize, usize)],
        init_transform: &Matrix4<f32>,
    ) -> Option<GNCResult> {
        if correspondences.len() < 3 {
            return None;
        }

        let mut transformation = *init_transform;
        let mut best_cost = f32::MAX;
        let mut best_transformation = transformation;

        // GNC outer loop - gradually reduce parameter
        for gnc_iter in 0..self.gnc_iterations {
            // Compute current parameter schedule
            let alpha = (gnc_iter as f32 + 1.0) / self.gnc_iterations as f32;
            let param = self.compute_parameter_schedule(alpha);
            self.loss.update_param(param);

            // Inner loop - weighted least squares
            for _ in 0..self.max_iterations {
                // Compute residuals and weights
                let (residuals, weights): (Vec<f32>, Vec<f32>) = correspondences
                    .iter()
                    .map(|&(src_idx, tgt_idx)| {
                        let src_point = source[src_idx];
                        let tgt_point = target[tgt_idx];
                        let transformed = transformation.transform_point(&src_point);
                        let residual = (transformed - tgt_point).norm();
                        let weight = self.loss.weight(residual);
                        (residual, weight)
                    })
                    .unzip();

                // Check convergence
                let cost: f32 = residuals.iter().map(|&r| self.loss.evaluate(r)).sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_transformation = transformation;
                }

                // Check for convergence
                if self.has_converged(&residuals, &weights) {
                    break;
                }

                // Solve weighted least squares
                if let Some(new_transform) =
                    self.solve_weighted_ls(source, target, correspondences, &weights)
                {
                    transformation = new_transform;
                } else {
                    break;
                }
            }
        }

        // Final evaluation
        let inliers: Vec<_> = correspondences
            .iter()
            .filter(|&&(src_idx, tgt_idx)| {
                let src_point = source[src_idx];
                let tgt_point = target[tgt_idx];
                let transformed = best_transformation.transform_point(&src_point);
                let residual = (transformed - tgt_point).norm();
                residual < self.loss.get_param()
            })
            .copied()
            .collect();

        let fitness = inliers.len() as f32 / correspondences.len() as f32;
        let rmse = if !inliers.is_empty() {
            let squared_errors: f32 = inliers
                .iter()
                .map(|&(src_idx, tgt_idx)| {
                    let src_point = source[src_idx];
                    let tgt_point = target[tgt_idx];
                    let transformed = best_transformation.transform_point(&src_point);
                    let diff = transformed - tgt_point;
                    diff.norm_squared()
                })
                .sum();
            (squared_errors / inliers.len() as f32).sqrt()
        } else {
            0.0
        };

        Some(GNCResult {
            transformation: best_transformation,
            fitness,
            inlier_rmse: rmse,
            inlier_count: inliers.len(),
            total_correspondences: correspondences.len(),
            cost: best_cost,
        })
    }

    /// Compute parameter schedule for GNC
    /// α ∈ [0, 1] where 0 = convex surrogate, 1 = target cost
    fn compute_parameter_schedule(&self, alpha: f32) -> f32 {
        match &self.loss {
            RobustLoss::GemanMcClure { mu } => {
                // Start with large μ (convex), decrease to target
                let max_mu = mu * 100.0;
                let min_mu = *mu;
                max_mu * (1.0 - alpha) + min_mu * alpha
            }
            RobustLoss::Welsch { mu } => {
                // Similar schedule
                let max_mu = mu * 100.0;
                let min_mu = *mu;
                max_mu * (1.0 - alpha) + min_mu * alpha
            }
            RobustLoss::TruncatedLeastSquares { c } => {
                // Start with c = ∞ (quadratic), decrease to target
                let max_c = c * 10.0;
                let min_c = *c;
                max_c * (1.0 - alpha) + min_c * alpha
            }
            RobustLoss::Huber { mu } => {
                let max_mu = mu * 100.0;
                let min_mu = *mu;
                max_mu * (1.0 - alpha) + min_mu * alpha
            }
            RobustLoss::Cauchy { mu } => {
                let max_mu = mu * 100.0;
                let min_mu = *mu;
                max_mu * (1.0 - alpha) + min_mu * alpha
            }
            RobustLoss::Tukey { mu } => {
                let max_mu = mu * 100.0;
                let min_mu = *mu;
                max_mu * (1.0 - alpha) + min_mu * alpha
            }
        }
    }

    /// Check for convergence
    fn has_converged(&self, residuals: &[f32], weights: &[f32]) -> bool {
        // Check if cost change is small
        let weighted_residual: f32 = residuals
            .iter()
            .zip(weights.iter())
            .map(|(r, w)| r * w)
            .sum();

        weighted_residual < self.convergence_threshold * residuals.len() as f32
    }

    /// Solve weighted least squares problem
    fn solve_weighted_ls(
        &self,
        source: &[Point3<f32>],
        target: &[Point3<f32>],
        correspondences: &[(usize, usize)],
        weights: &[f32],
    ) -> Option<Matrix4<f32>> {
        if correspondences.len() < 3 {
            return None;
        }

        // Compute weighted centroids
        let mut source_centroid = Point3::origin();
        let mut target_centroid = Point3::origin();
        let mut total_weight = 0.0;

        for (i, &(src_idx, tgt_idx)) in correspondences.iter().enumerate() {
            let w = weights[i];
            source_centroid += source[src_idx].coords * w;
            target_centroid += target[tgt_idx].coords * w;
            total_weight += w;
        }

        if total_weight < 1e-6 {
            return None;
        }

        source_centroid /= total_weight;
        target_centroid /= total_weight;

        // Compute weighted covariance
        let mut covariance = nalgebra::Matrix3::<f32>::zeros();

        for (i, &(src_idx, tgt_idx)) in correspondences.iter().enumerate() {
            let w = weights[i];
            let src = (source[src_idx] - source_centroid.coords).coords;
            let tgt = (target[tgt_idx] - target_centroid.coords).coords;
            covariance += tgt * src.transpose() * w;
        }

        // SVD to find rotation
        let svd = covariance.svd(true, true);
        let u = svd.u?;
        let vt = svd.v_t?;

        let mut rotation = u * vt;

        // Ensure proper rotation (det = 1)
        if rotation.determinant() < 0.0 {
            let mut u_corrected = u;
            u_corrected.set_column(2, &(u.column(2) * -1.0));
            rotation = u_corrected * vt;
        }

        // Compute translation
        let translation = target_centroid.coords - rotation * source_centroid.coords;

        // Build transformation matrix
        let mut transformation = Matrix4::identity();
        transformation
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation);
        transformation
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&translation);

        Some(transformation)
    }
}

/// Result from GNC optimization
#[derive(Debug, Clone)]
pub struct GNCResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
    pub inlier_count: usize,
    pub total_correspondences: usize,
    pub cost: f32,
}

/// Convenience function for robust registration with GNC
pub fn registration_gnc(
    source: &[Point3<f32>],
    target: &[Point3<f32>],
    correspondences: &[(usize, usize)],
    max_correspondence_distance: f32,
    loss_type: RobustLossType,
) -> Option<GNCResult> {
    let mut optimizer = match loss_type {
        RobustLossType::GemanMcClure => {
            GNCOptimizer::new_geman_mcclure(max_correspondence_distance)
        }
        RobustLossType::TruncatedLeastSquares => GNCOptimizer::new_tls(max_correspondence_distance),
        RobustLossType::Welsch => GNCOptimizer::new_welsch(max_correspondence_distance),
    };

    optimizer.solve_registration(source, target, correspondences, &Matrix4::identity())
}

/// Type of robust loss to use
#[derive(Debug, Clone, Copy)]
pub enum RobustLossType {
    GemanMcClure,
    TruncatedLeastSquares,
    Welsch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geman_mcclure_loss() {
        let loss = RobustLoss::GemanMcClure { mu: 1.0 };

        // Small residual - approximately quadratic
        let small = loss.evaluate(0.1);
        assert!(small < 0.01);

        // Large residual - saturates
        let large = loss.evaluate(10.0);
        assert!(large < 1.0); // Should be close to mu

        // Weight decreases for large residuals
        let w_small = loss.weight(0.1);
        let w_large = loss.weight(10.0);
        assert!(w_large < w_small);
    }

    #[test]
    fn test_tls_loss() {
        let loss = RobustLoss::TruncatedLeastSquares { c: 1.0 };

        // Inlier - quadratic
        let inlier = loss.evaluate(0.5);
        assert_eq!(inlier, 0.25);

        // Outlier - truncated
        let outlier = loss.evaluate(2.0);
        assert_eq!(outlier, 1.0);

        // Outlier weight is zero
        let w_outlier = loss.weight(2.0);
        assert_eq!(w_outlier, 0.0);
    }
}
