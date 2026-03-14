//! Global Registration
//!
//! RANSAC-based global registration that doesn't require initial alignment.
//! Uses FPFH (Fast Point Feature Histograms) for feature matching.

#![allow(deprecated)]

use cv_core::point_cloud::PointCloud;
use cv_core::{Error, Result};
use nalgebra::{Matrix4, Point3, Vector3};

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
        self.tree
            .nearest_neighbor(query)
            .map(|(pt, idx, dist_sq)| (pt, idx, dist_sq))
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

/// FPFH (Fast Point Feature Histogram) feature
#[derive(Debug, Clone)]
pub struct FPFHFeature {
    pub histogram: [f32; 33], // 33-dimensional histogram
}

/// Compute FPFH features for a point cloud.
///
/// Implements the full FPFH pipeline from Rusu (ICRA 2009):
/// 1. Estimate normals if not already present (via PCA over kNN).
/// 2. Build a voxel-grid spatial index for radius search.
/// 3. Compute SPFH (Simplified Point Feature Histograms) for every point.
/// 4. Weight neighbour SPFHs to produce FPFH for every point.
///
/// Returns one 33-bin feature vector per point.
pub fn compute_fpfh_features(cloud: &PointCloud, radius: f32) -> Result<Vec<FPFHFeature>> {
    let n = cloud.points.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // --- normals ---
    // If the cloud already has normals, use them; otherwise estimate via PCA.
    let normals: Vec<Vector3<f32>> = if let Some(ref normals) = cloud.normals {
        normals.clone()
    } else {
        estimate_normals_pca(&cloud.points, radius)
    };

    if normals.len() != n {
        return Err(Error::RuntimeError(
            "Normal count does not match point count".to_string(),
        ));
    }

    let points = &cloud.points;
    let radius_sq = radius * radius;

    // --- voxel-grid spatial index for radius search ---
    let voxel_size = radius;
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 4 + 1);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    // Helper: collect all neighbours within `radius` of `center` (excluding self).
    let radius_search = |center: &Point3<f32>, self_idx: usize| -> Vec<usize> {
        let vx = (center.x / voxel_size).floor() as i32;
        let vy = (center.y / voxel_size).floor() as i32;
        let vz = (center.z / voxel_size).floor() as i32;

        let mut result = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != self_idx {
                                let p = &points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= radius_sq {
                                    result.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    };

    // --- Step 1: Compute SPFH for every point ---
    let all_spfh: Vec<[f32; 33]> = (0..n)
        .map(|i| {
            let neighbors = radius_search(&points[i], i);
            compute_spfh(&points[i], &normals[i], &neighbors, points, &normals)
        })
        .collect();

    // --- Step 2: Weight SPFHs to produce FPFH ---
    let fpfh_features: Vec<FPFHFeature> = (0..n)
        .map(|i| {
            let neighbors = radius_search(&points[i], i);
            let histogram = weight_spfh(&points[i], &all_spfh[i], &neighbors, points, &all_spfh);
            FPFHFeature { histogram }
        })
        .collect();

    Ok(fpfh_features)
}

/// Estimate normals via PCA over radius-based neighbours.
///
/// This is a self-contained fallback so the registration crate does not
/// depend on `cv-3d`.
fn estimate_normals_pca(points: &[Point3<f32>], radius: f32) -> Vec<Vector3<f32>> {
    let n = points.len();
    let mut normals = vec![Vector3::z(); n];

    if n == 0 {
        return normals;
    }

    let voxel_size = radius;
    let radius_sq = radius * radius;

    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 4 + 1);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    for (i, center) in points.iter().enumerate() {
        let vx = (center.x / voxel_size).floor() as i32;
        let vy = (center.y / voxel_size).floor() as i32;
        let vz = (center.z / voxel_size).floor() as i32;

        let mut cov = nalgebra::Matrix3::<f32>::zeros();
        let mut centroid = Vector3::zeros();
        let mut count = 0u32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            let p = &points[idx];
                            let dist_sq = (center.x - p.x).powi(2)
                                + (center.y - p.y).powi(2)
                                + (center.z - p.z).powi(2);
                            if dist_sq <= radius_sq {
                                centroid += p.coords;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        if count < 3 {
            continue; // keep default (0,0,1)
        }

        centroid /= count as f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            let p = &points[idx];
                            let dist_sq = (center.x - p.x).powi(2)
                                + (center.y - p.y).powi(2)
                                + (center.z - p.z).powi(2);
                            if dist_sq <= radius_sq {
                                let diff = p.coords - centroid;
                                cov += diff * diff.transpose();
                            }
                        }
                    }
                }
            }
        }

        // The normal is the eigenvector corresponding to the smallest eigenvalue.
        let eig = cov.symmetric_eigen();
        let mut min_idx = 0;
        let mut min_val = eig.eigenvalues[0].abs();
        for k in 1..3 {
            if eig.eigenvalues[k].abs() < min_val {
                min_val = eig.eigenvalues[k].abs();
                min_idx = k;
            }
        }

        let mut normal = eig.eigenvectors.column(min_idx).into_owned();
        let norm = normal.norm();
        if norm > 1e-6 {
            normal /= norm;
        } else {
            normal = Vector3::z();
        }

        // Orient towards positive-z half-space (convention)
        if normal.z < 0.0 {
            normal = -normal;
        }

        normals[i] = normal;
    }

    normals
}

/// Compute Simple Point Feature Histogram for a single point.
///
/// Implements the SPFH computation from Rusu (ICRA 2009) using the Darboux
/// frame to compute three angular features (alpha, phi, theta) for each
/// pair (source_point, neighbor), then bins them into an 11-bin histogram
/// per feature (33 bins total).
fn compute_spfh(
    point: &Point3<f32>,
    normal: &Vector3<f32>,
    neighbors: &[usize],
    points: &[Point3<f32>],
    normals: &[Vector3<f32>],
) -> [f32; 33] {
    let mut histogram = [0.0f32; 33];
    let mut count = 0u32;

    for &neighbor_idx in neighbors {
        if neighbor_idx >= points.len() {
            continue;
        }

        let neighbor = &points[neighbor_idx];
        let d = neighbor - point;
        let dist = d.norm();

        if dist < 1e-6 {
            continue;
        }

        let n_target = &normals[neighbor_idx];

        // Darboux frame (Rusu ICRA 2009):
        //   u = n_source
        //   v = u x (p_target - p_source) / ||p_target - p_source||
        //   w = u x v
        let u = *normal;
        let v_raw = u.cross(&d);
        let v_norm = v_raw.norm();
        if v_norm < 1e-6 {
            continue;
        }
        let v = v_raw / v_norm;
        let w = u.cross(&v);

        // Angular features:
        //   alpha = v . n_target
        //   phi   = u . d / ||d||
        //   theta = atan2(w . n_target, u . n_target)
        let alpha = v.dot(n_target);
        let phi = u.dot(&d) / dist;
        let theta = (w.dot(n_target)).atan2(u.dot(n_target));

        // Bin into 11 bins per feature
        // alpha in [-1, 1], phi in [-1, 1], theta in [-PI, PI]
        let alpha_bin = ((alpha + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
        let phi_bin = ((phi + 1.0) * 5.5).floor().clamp(0.0, 10.0) as usize;
        let theta_bin = ((theta + std::f32::consts::PI) * (11.0 / (2.0 * std::f32::consts::PI)))
            .floor()
            .clamp(0.0, 10.0) as usize;

        histogram[alpha_bin] += 1.0;
        histogram[11 + phi_bin] += 1.0;
        histogram[22 + theta_bin] += 1.0;
        count += 1;
    }

    // Normalize by count (each bin becomes a fraction)
    if count > 0 {
        let inv_k = 1.0 / count as f32;
        for h in &mut histogram {
            *h *= inv_k;
        }
    }

    histogram
}

/// Weight SPFH with neighbors to produce FPFH (Rusu ICRA 2009).
///
/// FPFH(p) = SPFH(p) + (1/k) * sum_{i=1}^{k} (1/||p - p_i||) * SPFH(p_i)
///
/// where k is the number of neighbors and p_i are the neighbor points.
fn weight_spfh(
    point: &Point3<f32>,
    own_spfh: &[f32; 33],
    neighbors: &[usize],
    points: &[Point3<f32>],
    all_spfh: &[[f32; 33]],
) -> [f32; 33] {
    let mut fpfh = *own_spfh;

    let k = neighbors.len();
    if k == 0 {
        return fpfh;
    }

    let inv_k = 1.0 / k as f32;

    for &neighbor_idx in neighbors {
        let dist = (points[neighbor_idx] - point).norm();
        if dist < 1e-6 {
            continue;
        }

        let w = inv_k / dist;
        let neighbor_spfh = &all_spfh[neighbor_idx];
        for bin in 0..33 {
            fpfh[bin] += neighbor_spfh[bin] * w;
        }
    }

    // Normalize to sum to 100 (standard FPFH convention)
    let sum: f32 = fpfh.iter().sum();
    if sum > 1e-6 {
        let scale = 100.0 / sum;
        for h in &mut fpfh {
            *h *= scale;
        }
    }

    fpfh
}

use cv_core::{Ransac, RobustConfig, RobustModel};

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
    // Find correspondences (Brute force)
    let mut correspondences: Vec<(usize, usize, f32)> = Vec::new();
    for (i, source_feature) in source_features.iter().enumerate() {
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
            correspondences.push((i, min_idx, min_dist));
        }
    }

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
    // --- 1. Feature matching: nearest neighbour in feature space ---
    let mut correspondences: Vec<(usize, usize)> = Vec::new();

    for (i, sf) in source_features.iter().enumerate() {
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
        correspondences.push((i, min_idx));
    }

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

/// ISS (Intrinsic Shape Signatures) feature detector
/// Alternative to FPFH, good for scenes with repeated structures
#[derive(Debug, Clone)]
pub struct ISSFeature {
    pub keypoint_index: usize,
    pub descriptor: [f32; 16], // Simplified 16D descriptor
    pub covariance_eigenvalues: [f32; 3],
}

impl Default for ISSFeature {
    fn default() -> Self {
        Self {
            keypoint_index: 0,
            descriptor: [0.0; 16],
            covariance_eigenvalues: [0.0; 3],
        }
    }
}

/// ISS feature detector parameters
#[derive(Debug, Clone)]
pub struct ISSDetector {
    /// Radius for computing scatter matrix
    pub saliency_radius: f32,
    /// Minimum eigenvalue threshold
    pub min_eigenvalue: f32,
    /// Radius for non-maximum suppression
    pub non_max_radius: f32,
    /// Minimum number of neighbors
    pub min_neighbors: usize,
}

impl Default for ISSDetector {
    fn default() -> Self {
        Self {
            saliency_radius: 0.1,
            min_eigenvalue: 0.001,
            non_max_radius: 0.05,
            min_neighbors: 5,
        }
    }
}

/// Compute ISS keypoints and features
pub fn compute_iss_features(cloud: &PointCloud, detector: ISSDetector) -> Vec<ISSFeature> {
    let points = &cloud.points;
    let n = points.len();

    if n == 0 {
        return Vec::new();
    }

    let saliency_radius_sq = detector.saliency_radius * detector.saliency_radius;
    let non_max_radius_sq = detector.non_max_radius * detector.non_max_radius;

    // Build spatial index
    let voxel_size = detector.saliency_radius;
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::with_capacity(n / 10);

    for (i, p) in points.iter().enumerate() {
        let vx = (p.x / voxel_size).floor() as i32;
        let vy = (p.y / voxel_size).floor() as i32;
        let vz = (p.z / voxel_size).floor() as i32;
        voxel_grid.entry((vx, vy, vz)).or_default().push(i);
    }

    // Compute saliency (determinant of scatter matrix) for each point
    let mut saliencies: Vec<(usize, f32)> = Vec::with_capacity(n);

    for (i, center) in points.iter().enumerate() {
        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        // Gather neighbors
        let mut neighbors: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != i {
                                let p = points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= saliency_radius_sq {
                                    neighbors.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        if neighbors.len() < detector.min_neighbors {
            saliencies.push((i, 0.0));
            continue;
        }

        // Compute scatter matrix
        let mut scatter = nalgebra::Matrix3::zeros();
        let mut centroid = nalgebra::Vector3::zeros();

        for &idx in &neighbors {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }
        centroid /= neighbors.len() as f32;

        for &idx in &neighbors {
            let p = points[idx];
            let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
            scatter += diff * diff.transpose();
        }

        scatter /= neighbors.len() as f32;

        // Compute eigenvalues
        let eigenvals = scatter.eigenvalues();
        if let Some(eigs) = eigenvals {
            // Use determinant as saliency
            let det = eigs[0] * eigs[1] * eigs[2];
            saliencies.push((i, det));
        } else {
            saliencies.push((i, 0.0));
        }
    }

    // Non-maximum suppression
    let mut keypoints: Vec<usize> = Vec::new();

    // Sort by saliency (descending)
    saliencies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; n];

    for (idx, _saliency) in &saliencies {
        if suppressed[*idx] {
            continue;
        }

        keypoints.push(*idx);
        let center = points[*idx];

        // Suppress neighbors within non_max_radius
        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &j in indices {
                            let pj = points[j];
                            let dist_sq = (center.x - pj.x).powi(2)
                                + (center.y - pj.y).powi(2)
                                + (center.z - pj.z).powi(2);
                            if dist_sq <= non_max_radius_sq {
                                suppressed[j] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute descriptors for keypoints
    let mut features = Vec::with_capacity(keypoints.len());

    for &keypoint_idx in &keypoints {
        let center = points[keypoint_idx];

        let (vx, vy, vz) = (
            (center.x / voxel_size).floor() as i32,
            (center.y / voxel_size).floor() as i32,
            (center.z / voxel_size).floor() as i32,
        );

        // Get neighbors for descriptor
        let mut neighbors: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = voxel_grid.get(&(vx + dx, vy + dy, vz + dz)) {
                        for &idx in indices {
                            if idx != keypoint_idx {
                                let p = points[idx];
                                let dist_sq = (center.x - p.x).powi(2)
                                    + (center.y - p.y).powi(2)
                                    + (center.z - p.z).powi(2);
                                if dist_sq <= saliency_radius_sq {
                                    neighbors.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Compute covariance for descriptor
        let mut cov = nalgebra::Matrix3::zeros();
        let mut centroid = nalgebra::Vector3::zeros();

        for &idx in &neighbors {
            let p = points[idx];
            centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
        }

        if !neighbors.is_empty() {
            centroid /= neighbors.len() as f32;

            for &idx in &neighbors {
                let p = points[idx];
                let diff = nalgebra::Vector3::new(p.x, p.y, p.z) - centroid;
                cov += diff * diff.transpose();
            }
        }

        let eigenvals = cov.eigenvalues();
        let mut eigenvalues = [0.0f32; 3];
        if let Some(eigs) = eigenvals {
            eigenvalues = [eigs[0], eigs[1], eigs[2]];
        }

        // Create simple descriptor from eigenvalues
        let mut descriptor = [0.0f32; 16];
        if !neighbors.is_empty() {
            let e1 = eigenvalues[0].max(1e-10);
            let e2 = eigenvalues[1].max(1e-10);
            let e3 = eigenvalues[2].max(1e-10);
            let sum = (e1 + e2 + e3).max(1e-10);

            // Normalized eigenvalues as descriptor
            descriptor[0] = e1 / sum;
            descriptor[1] = e2 / sum;
            descriptor[2] = e3 / sum;
            descriptor[3] = neighbors.len() as f32;

            // Add spatial features
            let dx = centroid[0] - center.x;
            let dy = centroid[1] - center.y;
            let dz = centroid[2] - center.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-10);

            descriptor[4] = dx / dist;
            descriptor[5] = dy / dist;
            descriptor[6] = dz / dist;

            // Fill remaining with eigenvalue ratios
            descriptor[7] = (e1 * e2 / (e3 * e3)).min(100.0);
            descriptor[8] = (e1 * e3 / (e2 * e2)).min(100.0);
            descriptor[9] = (e2 * e3 / (e1 * e1)).min(100.0);
            descriptor[10] = (e1 / e3).min(100.0);
            descriptor[11] = (e2 / e3).min(100.0);
            descriptor[12] = (e1 - e2).abs() / e3.max(1e-10);
            descriptor[13] = (e1 - e3).abs() / e2.max(1e-10);
            descriptor[14] = (e2 - e3).abs() / e1.max(1e-10);
            descriptor[15] = (e1 - e2 - e3).abs() / e1.max(1e-10);
        }

        features.push(ISSFeature {
            keypoint_index: keypoint_idx,
            descriptor,
            covariance_eigenvalues: eigenvalues,
        });
    }

    features
}
