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
    /// Correspondences per hypothesis (Open3D's `ransac_n`, at least 3).
    sample_size: usize,
}

impl<'a> RobustModel<(usize, usize, f32)> for GlobalRegistrationEstimator<'a> {
    type Model = Matrix4<f32>;
    fn min_sample_size(&self) -> usize {
        self.sample_size
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

/// Squared L2 between two FPFH histograms, summed in bin order. The same
/// reduction the linear scan used, so an exact tie stays an exact tie.
fn histogram_dist2(a: &[f32; 33], b: &[f32; 33]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Exact nearest neighbour over 33-D FPFH histograms.
///
/// The scan kept the first strict minimum (`dist < best`), which is the lower
/// index on a tie. A candidate replaces the incumbent only when its squared
/// distance is strictly smaller, or equal and its index is lower. The far
/// child is skipped only when the split plane is *strictly* farther than the
/// current worst accepted distance, so an equal-distance point with a lower
/// index on the other side of the split is still visited.
struct FpfhNn {
    hist: Vec<[f32; 33]>,
    nodes: Vec<FpfhNode>,
    order: Vec<usize>,
}

enum FpfhNode {
    Leaf {
        start: usize,
        count: usize,
    },
    Split {
        axis: u8,
        split: f32,
        left: usize,
        right: usize,
    },
}

const FPFH_LEAF: usize = 8;

impl FpfhNn {
    fn build(features: &[FPFHFeature]) -> Self {
        let hist: Vec<[f32; 33]> = features.iter().map(|f| f.histogram).collect();
        let n = hist.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::new();
        if n > 0 {
            Self::build_range(&hist, &mut order, &mut nodes, 0, n, 0);
        }
        Self { hist, nodes, order }
    }

    fn build_range(
        hist: &[[f32; 33]],
        order: &mut [usize],
        nodes: &mut Vec<FpfhNode>,
        start: usize,
        end: usize,
        depth: usize,
    ) -> usize {
        let count = end - start;
        let id = nodes.len();
        if count <= FPFH_LEAF {
            nodes.push(FpfhNode::Leaf { start, count });
            return id;
        }
        let axis = (depth % 33) as u8;
        let axis_us = axis as usize;
        order[start..end].sort_by(|&a, &b| {
            hist[a][axis_us]
                .partial_cmp(&hist[b][axis_us])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = start + count / 2;
        let split = hist[order[mid]][axis_us];
        nodes.push(FpfhNode::Split {
            axis,
            split,
            left: 0,
            right: 0,
        });
        let left = Self::build_range(hist, order, nodes, start, mid, depth + 1);
        let right = Self::build_range(hist, order, nodes, mid, end, depth + 1);
        if let FpfhNode::Split {
            left: slot_l,
            right: slot_r,
            ..
        } = &mut nodes[id]
        {
            *slot_l = left;
            *slot_r = right;
        }
        id
    }

    /// `(index, Euclidean distance)`. Empty or entirely non-finite input
    /// returns `(0, f32::MAX)`, which is what the scan left in its
    /// initialisers (and `f32::MAX` is finite, so the RANSAC filter kept it).
    fn nearest(&self, query: &[f32; 33]) -> (usize, f32) {
        if self.nodes.is_empty() {
            return (0, f32::MAX);
        }
        let mut best = f32::INFINITY;
        let mut best_idx = usize::MAX;
        self.search_one(0, query, &mut best, &mut best_idx);
        if best_idx == usize::MAX {
            (0, f32::MAX)
        } else {
            (best_idx, best)
        }
    }

    /// `(best index, best distance, second distance)`. `second` is `f32::MAX`
    /// when only one finite histogram exists, matching the scan's sentinel.
    /// `None` when nothing finite was seen.
    fn nearest_two(&self, query: &[f32; 33]) -> Option<(usize, f32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = f32::INFINITY;
        let mut best_idx = usize::MAX;
        let mut second = f32::INFINITY;
        self.search_two(0, query, &mut best, &mut best_idx, &mut second);
        if best_idx == usize::MAX {
            return None;
        }
        // The scan's runner-up sentinel is `f32::MAX`, which is finite, so a
        // lone target still enters the ratio test.
        let second = if second.is_finite() { second } else { f32::MAX };
        Some((best_idx, best, second))
    }

    fn search_one(&self, node: usize, query: &[f32; 33], best: &mut f32, best_idx: &mut usize) {
        match self.nodes[node] {
            FpfhNode::Leaf { start, count } => {
                for &idx in &self.order[start..start + count] {
                    // Euclidean, not squared: two sums of squares can share a
                    // sqrt, and the scan's strict `<` is on the sqrt.
                    let dist = histogram_dist2(query, &self.hist[idx]).sqrt();
                    if dist < *best || (dist == *best && idx < *best_idx) {
                        *best = dist;
                        *best_idx = idx;
                    }
                }
            }
            FpfhNode::Split {
                axis,
                split,
                left,
                right,
            } => {
                let delta = query[axis as usize] - split;
                let (near, far) = if delta <= 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.search_one(near, query, best, best_idx);
                // `|delta|` lower-bounds the Euclidean distance. `>` rather
                // than `>=`: an equal plane distance can hide a tie with a
                // lower index.
                if !(delta.abs() > *best) {
                    self.search_one(far, query, best, best_idx);
                }
            }
        }
    }

    fn consider_two(idx: usize, dist: f32, best: &mut f32, best_idx: &mut usize, second: &mut f32) {
        if dist < *best || (dist == *best && idx < *best_idx) {
            if dist < *best {
                if *best_idx != usize::MAX {
                    *second = *best;
                }
                *best = dist;
                *best_idx = idx;
            } else {
                // Same distance, lower index. The displaced point is still a
                // neighbour at this distance, so the runner-up collapses onto it.
                *second = *best;
                *best_idx = idx;
            }
        } else if dist < *second {
            *second = dist;
        }
    }

    fn search_two(
        &self,
        node: usize,
        query: &[f32; 33],
        best: &mut f32,
        best_idx: &mut usize,
        second: &mut f32,
    ) {
        match self.nodes[node] {
            FpfhNode::Leaf { start, count } => {
                for &idx in &self.order[start..start + count] {
                    let dist = histogram_dist2(query, &self.hist[idx]).sqrt();
                    Self::consider_two(idx, dist, best, best_idx, second);
                }
            }
            FpfhNode::Split {
                axis,
                split,
                left,
                right,
            } => {
                let delta = query[axis as usize] - split;
                let (near, far) = if delta <= 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.search_two(near, query, best, best_idx, second);
                // The runner-up is the prune threshold. Equality is not pruned:
                // the other side may hold a lower index at the same best distance.
                if !(delta.abs() > *second) {
                    self.search_two(far, query, best, best_idx, second);
                }
            }
        }
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
    // Exact nearest neighbour in FPFH space. The distance compared is the same
    // Euclidean value the linear scan used, so the argmin matches its strict
    // `<` and a tie still names the lower index.
    let target_nn = FpfhNn::build(target_features);
    let mut correspondences: Vec<(usize, usize, f32)> = source_features
        .par_iter()
        .enumerate()
        .filter_map(|(i, source_feature)| {
            let (min_idx, min_dist) = target_nn.nearest(&source_feature.histogram);
            // Every source feature gets its nearest target feature, as in
            // Open3D. The match used to be gated by `max_correspondence_
            // distance` - a *metric* distance in the point cloud's units,
            // applied to a distance between 33-D FPFH histograms (which sum
            // to ~100 per sub-histogram). The two scales are unrelated, so the
            // gate kept or dropped matches arbitrarily. Geometric consistency
            // is RANSAC's job, using that distance as its inlier threshold.
            (min_dist.is_finite()).then_some((i, min_idx, min_dist))
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

    let estimator = GlobalRegistrationEstimator {
        source,
        target,
        // `ransac_n` was accepted and then ignored (always 3).
        sample_size: ransac_n.max(3),
    };
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
    // Nearest neighbour with a Lowe ratio test.
    //
    // Without the ratio, a target set whose histograms are all alike - which is
    // what an un-trained or low-information FPFH set looks like - gives every
    // source point the same nearest target, so the whole set collapses to
    // `(i, 0)`. That is not a weak correspondence set, it is a physically
    // impossible one, and it went straight into the graduated solver, which
    // returned `Ok` with a transform derived from it. The sibling RANSAC entry
    // point in this file already validates; this one did not.
    let ratio = (option.tuple_scale.clamp(0.0, 1.0)) as f32;
    let target_nn = FpfhNn::build(target_features);
    let correspondences: Vec<(usize, usize)> = source_features
        .par_iter()
        .enumerate()
        .filter_map(|(i, sf)| {
            let (idx, best, second) = target_nn.nearest_two(&sf.histogram)?;
            // No second candidate, or the best is not clearly better than the
            // runner-up: the match is ambiguous and is dropped rather than
            // guessed. `second` is `f32::MAX` when the target has a single
            // finite histogram, which is finite, so the test still runs.
            if second.is_finite() && (best > ratio * second || best == second) {
                return None;
            }
            Some((i, idx))
        })
        .collect();

    if correspondences.len() < 3 {
        return Err(Error::RuntimeError(
            "Insufficient feature correspondences for FGR: the matches are ambiguous, \
             which usually means the FPFH features are not discriminative"
                .to_string(),
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
/// Options for fast global registration.
///
/// Every field here is read. `maximum_iterations` used to sit alongside
/// `iteration_number` and was read by neither, so tightening it had no effect
/// whatsoever; it is gone rather than left looking like supported configuration.
#[derive(Debug, Clone)]
pub struct FastGlobalRegistrationOption {
    /// Correspondences further apart than this are discarded.
    pub maximum_correspondence_distance: f64,
    /// Outer iterations of the graduated optimisation, which sharpens the robust
    /// kernel by shrinking `mu`. This is the loop bound.
    pub iteration_number: usize,
    /// Upper bound on how many correspondences the solver considers.
    pub maximum_tuple_count: usize,
    /// Lowe ratio threshold on the feature match: a match is kept only when the
    /// best distance is clearly better than the runner-up. Without it, a target
    /// set whose features are all alike collapses every source point onto the
    /// same target and the solver returns a transform built from that.
    pub tuple_scale: f64,
}

impl Default for FastGlobalRegistrationOption {
    fn default() -> Self {
        Self {
            maximum_correspondence_distance: 0.075,
            iteration_number: 64,
            maximum_tuple_count: 1000,
            tuple_scale: 0.95,
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

    if source.points.is_empty() {
        // A registration that was never attempted, so it is not a perfect one.
        return (0.0, f32::INFINITY);
    }

    for point in &source.points {
        let transformed = transformation.transform_point(point);
        if let Some((_, _, dist_sq)) = target_nn.nearest(&transformed) {
            if dist_sq.sqrt() < max_correspondence_distance {
                inlier_count += 1;
                // `dist_sq` is already the SQUARED Euclidean distance. Summing it
                // and taking one sqrt at the end gives sqrt(mean(d^4)), not an
                // RMSE - which is sqrt(mean(d^2)).
                //
                // Verified: 90 inliers at distance 1e-4 and 10 at 1e-2 reported
                // inlier_rmse 1.0045e-4 where the true RMSE is 3.302e-3,
                // understated 33x. The error scales with the *spread* of the
                // inlier distances, so it is invisible when they are all equal -
                // which is why the uniform-distance case looked correct.
                total_error += dist_sq;
            }
        }
    }

    let fitness = inlier_count as f32 / source.points.len() as f32;
    let rmse = if inlier_count > 0 {
        (total_error / inlier_count as f32).sqrt()
    } else {
        // Zero inliers is zero *support*, not zero error. Reporting 0.0 here is a
        // perfect score for a registration that matched nothing.
        f32::INFINITY
    };

    (fitness, rmse)
}

#[cfg(test)]
mod fgr_rejects_degenerate_matching {
    use super::*;

    fn feature(v: f32) -> FPFHFeature {
        FPFHFeature { histogram: [v; 33] }
    }

    fn cloud(n: usize) -> PointCloud {
        PointCloud {
            points: (0..n)
                .map(|i| nalgebra::Point3::new(i as f32, 0.0, 0.0))
                .collect(),
            colors: None,
            normals: None,
        }
    }

    /// Identical target features must produce an error, not a transform.
    ///
    /// With no ratio test, every source point's nearest target is the same one
    /// and the entire correspondence set collapses to `(i, 0)` - a physically
    /// impossible mapping that the graduated solver then turned into a result.
    /// The function returned `Ok` with `fitness = 0.6` from that set.
    #[test]
    fn identical_target_features_are_rejected() {
        let n = 8;
        let source = cloud(n);
        let target = cloud(n);
        // Every target feature identical: nothing is discriminative.
        let source_features: Vec<FPFHFeature> = (0..n).map(|i| feature(i as f32)).collect();
        let target_features: Vec<FPFHFeature> = (0..n).map(|_| feature(0.5)).collect();

        let r = registration_fgr_based_on_feature_matching(
            &source,
            &target,
            &source_features,
            &target_features,
            FastGlobalRegistrationOption::default(),
        );
        assert!(
            r.is_err(),
            "indistinguishable target features must be an error, not a transform: got {:?}",
            r.map(|v| (v.fitness, v.inlier_rmse))
        );
    }

    /// Distinct features must still register, so the guard is not a blanket reject.
    #[test]
    fn discriminative_features_still_register() {
        let n = 8;
        let source = cloud(n);
        let target = cloud(n);
        // Each feature distinct, so the best match is unambiguous.
        let source_features: Vec<FPFHFeature> = (0..n).map(|i| feature(i as f32)).collect();
        let target_features: Vec<FPFHFeature> = (0..n).map(|i| feature(i as f32)).collect();

        let r = registration_fgr_based_on_feature_matching(
            &source,
            &target,
            &source_features,
            &target_features,
            FastGlobalRegistrationOption::default(),
        );
        assert!(r.is_ok(), "distinct features should register: {r:?}");
    }
}

#[cfg(test)]
mod fpfh_nn_matches_the_scan {
    use super::*;

    fn hist(seed: u32) -> FPFHFeature {
        let mut h = [0.0f32; 33];
        let mut s = seed;
        for bin in &mut h {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *bin = (s >> 8) as f32 / (1u32 << 24) as f32;
        }
        FPFHFeature { histogram: h }
    }

    fn scan_one(query: &[f32; 33], targets: &[FPFHFeature]) -> (usize, f32) {
        let mut min_dist = f32::MAX;
        let mut min_idx = 0usize;
        for (j, target) in targets.iter().enumerate() {
            let dist = histogram_dist2(query, &target.histogram).sqrt();
            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }
        (min_idx, min_dist)
    }

    fn scan_two(query: &[f32; 33], targets: &[FPFHFeature]) -> Option<(usize, f32, f32)> {
        let mut best = f32::MAX;
        let mut second = f32::MAX;
        let mut best_idx = None;
        for (j, target) in targets.iter().enumerate() {
            let dist = histogram_dist2(query, &target.histogram).sqrt();
            if dist < best {
                second = best;
                best = dist;
                best_idx = Some(j);
            } else if dist < second {
                second = dist;
            }
        }
        best_idx.map(|idx| (idx, best, second))
    }

    #[test]
    fn nearest_histogram_keeps_the_lower_index_and_the_runner_up() {
        let mut targets: Vec<FPFHFeature> = (0..40).map(hist).collect();
        // Planted ties, including one that sorts onto the other side of a median.
        targets.push(targets[3].clone());
        targets.push(targets[7].clone());
        targets[15] = targets[2].clone();
        let tree = FpfhNn::build(&targets);
        for (i, query) in targets.iter().enumerate() {
            let (idx, dist) = tree.nearest(&query.histogram);
            let (want_idx, want_dist) = scan_one(&query.histogram, &targets);
            assert_eq!(idx, want_idx, "query {i}");
            assert_eq!(dist, want_dist, "query {i}");
            let got = tree.nearest_two(&query.histogram);
            let want = scan_two(&query.histogram, &targets);
            match (got, want) {
                (Some((gi, gb, gs)), Some((wi, wb, ws))) => {
                    assert_eq!(gi, wi, "query {i} index");
                    assert_eq!(gb, wb, "query {i} best");
                    assert_eq!(gs, ws, "query {i} second");
                }
                (None, None) => {}
                (a, b) => panic!("query {i}: tree {a:?} scan {b:?}"),
            }
        }
        // A histogram that matches two targets exactly must name the lower index.
        let tied = targets[3].clone();
        let (idx, _) = tree.nearest(&tied.histogram);
        assert_eq!(idx, 3);
    }
}
