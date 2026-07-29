//! RANSAC-based Global Registration
//!
//! Uses FPFH feature matching to establish correspondences and RANSAC
//! (with SVD-based rigid transform estimation) to find the best alignment.
//! Also includes Fast Global Registration (FGR) via graduated non-convexity.

#![allow(deprecated)]

use cv_core::point_cloud::PointCloud;
use cv_core::{Error, Ransac, Result, RobustConfig, RobustModel};
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;

use super::fpfh::FPFHFeature;

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

/// Registration result
#[derive(Debug, Clone)]
pub struct GlobalRegistrationResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
    pub correspondences: Vec<(usize, usize)>,
}

pub struct GlobalRegistrationEstimator<'a> {
    source: &'a PointCloud,
    target: &'a PointCloud,
}

impl<'a> RobustModel<(usize, usize, f32)> for GlobalRegistrationEstimator<'a> {
    type Model = Matrix4<f32>;
    fn min_sample_size(&self) -> usize {
        3
    }
    fn estimate(&self, data: &[&(usize, usize, f32)]) -> Option<Self::Model> {
        let correspondences: Vec<(usize, usize, f32)> = data.iter().map(|&&c| c).collect();
        compute_transformation_from_correspondences(self.source, self.target, &correspondences)
    }
    fn compute_error(&self, model: &Self::Model, data: &(usize, usize, f32)) -> f64 {
        let src_point = self.source.points[data.0];
        let tgt_point = self.target.points[data.1];
        let transformed = model.transform_point(&src_point);
        (transformed - tgt_point).norm() as f64
    }
}

/// Global registration using RANSAC
pub fn registration_ransac_based_on_feature_matching(
    source: &PointCloud,
    target: &PointCloud,
    source_features: &[FPFHFeature],
    target_features: &[FPFHFeature],
    max_correspondence_distance: f32,
    ransac_n: usize,
    max_iterations: usize,
) -> Result<GlobalRegistrationResult> {
    // Find correspondences (parallel brute-force — 33D histograms defeat KDTree)
    let mut correspondences: Vec<(usize, usize, f32)> = source_features
        .par_iter()
        .enumerate()
        .filter_map(|(i, source_feature)| {
            let mut min_dist = f32::MAX;
            let mut min_idx = 0;
            for (j, target_feature) in target_features.iter().enumerate() {
                let dist = source_feature
                    .histogram
                    .iter()
                    .zip(target_feature.histogram.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            if min_dist < max_correspondence_distance {
                Some((i, min_idx, min_dist))
            } else {
                None
            }
        })
        .collect();

    correspondences.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let top_correspondences: Vec<_> = correspondences.into_iter().take(1000).collect();

    if top_correspondences.len() < ransac_n {
        return Err(Error::RuntimeError(
            "Insufficient correspondences".to_string(),
        ));
    }

    let config = RobustConfig {
        threshold: max_correspondence_distance as f64,
        max_iterations,
        confidence: 0.99,
    };

    let estimator = GlobalRegistrationEstimator { source, target };
    let ransac = Ransac::new(config);
    let res = ransac.run(&estimator, &top_correspondences);

    let final_transformation = res
        .model
        .ok_or_else(|| Error::RuntimeError("RANSAC failed to find model".to_string()))?;

    // Compute fitness and RMSE
    let (fitness, rmse) = evaluate_registration(
        source,
        target,
        &final_transformation,
        max_correspondence_distance,
    );

    let inlier_correspondences = top_correspondences
        .iter()
        .zip(res.inliers.iter())
        .filter(|(_, &inlier)| inlier)
        .map(|(c, _)| (c.0, c.1))
        .collect();

    Ok(GlobalRegistrationResult {
        transformation: final_transformation,
        fitness,
        inlier_rmse: rmse,
        correspondences: inlier_correspondences,
    })
}

/// Fast Global Registration (FGR) using graduated non-convexity.
///
/// Implements Zhou, Park & Koltun (ECCV 2016):
/// 1. Find feature correspondences via nearest-neighbour in FPFH space.
/// 2. Graduated optimisation with a scaled Geman-McClure kernel:
///    - Start with a large `mu` (convex surrogate of the robust cost).
///    - At each iteration compute line-process weights
///      `l_ij = mu / (mu + r_ij^2)` and solve a weighted least-squares
///      rigid transform via SVD.
///    - Shrink `mu` by a factor (div_factor) each outer iteration,
///      gradually sharpening outlier rejection.
pub fn registration_fgr_based_on_feature_matching(
    source: &PointCloud,
    target: &PointCloud,
    source_features: &[FPFHFeature],
    target_features: &[FPFHFeature],
    option: FastGlobalRegistrationOption,
) -> Result<GlobalRegistrationResult> {
    // --- 1. Feature matching: parallel nearest neighbour in feature space ---
    let correspondences: Vec<(usize, usize)> = source_features
        .par_iter()
        .enumerate()
        .map(|(i, sf)| {
            let mut min_dist = f32::MAX;
            let mut min_idx = 0;
            for (j, tf) in target_features.iter().enumerate() {
                let dist: f32 = sf
                    .histogram
                    .iter()
                    .zip(tf.histogram.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            (i, min_idx)
        })
        .collect();

    if correspondences.len() < 3 {
        return Err(Error::RuntimeError(
            "Insufficient feature correspondences for FGR".to_string(),
        ));
    }

    // Limit to top correspondences by feature distance to keep it tractable
    let max_corr = option.maximum_tuple_count.min(correspondences.len());
    // Re-sort by feature distance to trim
    let mut scored: Vec<(usize, usize, f32)> = correspondences
        .iter()
        .map(|&(si, ti)| {
            let dist: f32 = source_features[si]
                .histogram
                .iter()
                .zip(target_features[ti].histogram.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (si, ti, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max_corr);

    let corr_pairs: Vec<(usize, usize)> = scored.iter().map(|&(s, t, _)| (s, t)).collect();

    // --- 2. Graduated optimisation ---
    let mut transformation = Matrix4::<f32>::identity();
    let div_factor = 1.4_f64; // mu shrink factor per outer iteration
    let max_corr_dist = option.maximum_correspondence_distance;

    // Initialise mu: start large so the surrogate is nearly convex.
    // Use the max pairwise distance squared as the initial scale.
    let mut mu: f64 = {
        let mut max_sq: f64 = 0.0;
        for &(si, ti) in &corr_pairs {
            let sp = source.points[si];
            let tp = target.points[ti];
            let d = ((sp.x - tp.x).powi(2) + (sp.y - tp.y).powi(2) + (sp.z - tp.z).powi(2)) as f64;
            if d > max_sq {
                max_sq = d;
            }
        }
        max_sq.max(1.0)
    };

    let outer_iterations = option.iteration_number;

    for _outer in 0..outer_iterations {
        // Compute residuals and line-process weights
        let mut weighted_correspondences: Vec<(usize, usize, f32)> = Vec::new();

        for &(si, ti) in &corr_pairs {
            let sp = transformation.transform_point(&source.points[si]);
            let tp = target.points[ti];
            let r_sq =
                ((sp.x - tp.x).powi(2) + (sp.y - tp.y).powi(2) + (sp.z - tp.z).powi(2)) as f64;

            // Line process weight: l_ij = mu / (mu + r_ij^2)
            let weight = (mu / (mu + r_sq)) as f32;

            if weight > 1e-4 {
                weighted_correspondences.push((si, ti, weight));
            }
        }

        if weighted_correspondences.len() < 3 {
            break;
        }

        // Solve weighted least squares for the rigid transform
        if let Some(new_transform) = compute_weighted_transformation(
            source,
            target,
            &weighted_correspondences,
            &transformation,
        ) {
            transformation = new_transform;
        }

        // Shrink mu
        mu /= div_factor;

        // Early exit if mu is tiny
        if mu < max_corr_dist * max_corr_dist * 1e-6 {
            break;
        }
    }

    // --- 3. Evaluate final result ---
    let (fitness, rmse) =
        evaluate_registration(source, target, &transformation, max_corr_dist as f32);

    // Collect inlier correspondences
    let inlier_correspondences: Vec<(usize, usize)> = corr_pairs
        .iter()
        .filter(|&&(si, ti)| {
            let sp = transformation.transform_point(&source.points[si]);
            let tp = target.points[ti];
            (sp - tp).norm() < max_corr_dist as f32
        })
        .copied()
        .collect();

    Ok(GlobalRegistrationResult {
        transformation,
        fitness,
        inlier_rmse: rmse,
        correspondences: inlier_correspondences,
    })
}

/// Compute a rigid transform from weighted correspondences via SVD.
///
/// Each correspondence `(src_idx, tgt_idx, weight)` contributes to the
/// covariance matrix with the given weight. The source points are first
/// transformed by `current_transform` so that incremental updates work.
fn compute_weighted_transformation(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize, f32)],
    current_transform: &Matrix4<f32>,
) -> Option<Matrix4<f32>> {
    if correspondences.len() < 3 {
        return None;
    }

    let mut total_weight: f32 = 0.0;
    let mut source_centroid = Vector3::<f32>::zeros();
    let mut target_centroid = Vector3::<f32>::zeros();

    for &(src_idx, tgt_idx, w) in correspondences {
        let sp = current_transform
            .transform_point(&source.points[src_idx])
            .coords;
        let tp = target.points[tgt_idx].coords;
        source_centroid += sp * w;
        target_centroid += tp * w;
        total_weight += w;
    }

    if total_weight < 1e-6 {
        return None;
    }

    source_centroid /= total_weight;
    target_centroid /= total_weight;

    // Weighted covariance
    let mut covariance = nalgebra::Matrix3::<f32>::zeros();
    for &(src_idx, tgt_idx, w) in correspondences {
        let sp = current_transform
            .transform_point(&source.points[src_idx])
            .coords
            - source_centroid;
        let tp = target.points[tgt_idx].coords - target_centroid;
        covariance += (tp * sp.transpose()) * w;
    }

    // SVD to find rotation
    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;

    let mut rotation = u * vt;
    if rotation.determinant() < 0.0 {
        let mut u_corrected = u;
        u_corrected.set_column(2, &(u.column(2) * -1.0));
        rotation = u_corrected * vt;
    }

    let translation = target_centroid - rotation * source_centroid;

    // Build the full 4x4 transformation.
    // Since source points were already transformed by `current_transform`, the
    // returned matrix maps the *original* source directly to the target frame.
    let mut delta = Matrix4::identity();
    delta.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    delta.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);

    Some(delta * current_transform)
}

/// Options for Fast Global Registration
#[derive(Debug, Clone)]
pub struct FastGlobalRegistrationOption {
    pub maximum_correspondence_distance: f64,
    pub iteration_number: usize,
    pub maximum_tuple_count: usize,
    pub tuple_scale: f64,
    pub maximum_iterations: usize,
}

impl Default for FastGlobalRegistrationOption {
    fn default() -> Self {
        Self {
            maximum_correspondence_distance: 0.075,
            iteration_number: 64,
            maximum_tuple_count: 1000,
            tuple_scale: 0.95,
            maximum_iterations: 1000,
        }
    }
}

/// Compute rigid transformation from correspondences using SVD
fn compute_transformation_from_correspondences(
    source: &PointCloud,
    target: &PointCloud,
    correspondences: &[(usize, usize, f32)],
) -> Option<Matrix4<f32>> {
    if correspondences.len() < 3 {
        return None;
    }

    // Compute centroids
    let mut source_centroid = Point3::origin();
    let mut target_centroid = Point3::origin();

    for &(src_idx, tgt_idx, _) in correspondences {
        source_centroid += source.points[src_idx].coords;
        target_centroid += target.points[tgt_idx].coords;
    }

    let n = correspondences.len() as f32;
    source_centroid /= n;
    target_centroid /= n;

    // Compute covariance matrix
    let mut covariance = nalgebra::Matrix3::<f32>::zeros();

    for &(src_idx, tgt_idx, _) in correspondences {
        let src = (source.points[src_idx] - source_centroid.coords).coords;
        let tgt = (target.points[tgt_idx] - target_centroid.coords).coords;
        covariance += tgt * src.transpose();
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

/// Evaluate registration quality
fn evaluate_registration(
    source: &PointCloud,
    target: &PointCloud,
    transformation: &Matrix4<f32>,
    max_correspondence_distance: f32,
) -> (f32, f32) {
    // Build simple NN for target
    let target_nn = SimpleNN::new(target.points.clone());

    let mut inlier_count = 0;
    let mut total_error = 0.0;

    for point in &source.points {
        let transformed = transformation.transform_point(point);
        if let Some((_, _, dist)) = target_nn.nearest(&transformed) {
            if dist.sqrt() < max_correspondence_distance {
                inlier_count += 1;
                total_error += dist;
            }
        }
    }

    if source.points.is_empty() {
        return (0.0, 0.0);
    }

    let fitness = inlier_count as f32 / source.points.len() as f32;
    let rmse = if inlier_count > 0 {
        (total_error / inlier_count as f32).sqrt()
    } else {
        0.0
    };

    (fitness, rmse)
}

/// Random sample without replacement
#[allow(dead_code)]
fn random_sample(n: usize, max: usize) -> Vec<usize> {
    use std::collections::HashSet;
    let mut rng = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut indices = HashSet::new();
    while indices.len() < n && indices.len() < max {
        // Simple LCG random number generator
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let idx = (rng % max as u64) as usize;
        indices.insert(idx);
    }

    indices.into_iter().collect()
}
