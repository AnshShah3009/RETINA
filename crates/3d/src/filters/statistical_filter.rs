use nalgebra::Point3;
use rayon::prelude::*;

use super::knn_tree;

/// Statistical outlier removal (Open3D's `remove_statistical_outlier`).
///
/// For each point, computes the mean distance to its `nb_neighbors` nearest
/// points, **the point itself included** (Open3D searches `KNN(nb_neighbors)`,
/// whose first hit is the query point at distance 0, and averages over all of
/// them). Points whose mean distance is not below
/// `global_mean + std_ratio * global_std` are outliers, where `global_std` is
/// the sample standard deviation (`n - 1`) of the per-point means.
///
/// One deliberate difference: Open3D also drops every point whose mean is
/// exactly 0 (all of its neighbours coincide with it), because it uses
/// "mean > 0" as its validity flag. Such a point sits inside a cluster of
/// duplicates and is kept here.
///
/// # Returns
/// `(inlier_points, inlier_indices)`.
pub fn statistical_outlier_removal(
    points: &[Point3<f64>],
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    if points.len() < 2 {
        let indices: Vec<usize> = (0..points.len()).collect();
        return (points.to_vec(), indices);
    }

    // The neighbourhood includes the point. This used to be `nb_neighbors`
    // *other* points averaged over `nb_neighbors`, the population standard
    // deviation and an inclusive threshold, so the same parameters kept a
    // different set from Open3D: measured on 500 spread points, (5, 0.5) kept
    // 358 where Open3D keeps 364, and (20, 0.3) kept 343 where it keeps 340.
    let k = nb_neighbors.max(1).min(points.len());

    // Mean distance over the k nearest points of every point (parallel), via
    // an f64 KD-tree: O(n log n) instead of an O(n) scan per point. A
    // duplicate of the query can come back in its place; both are at distance
    // 0, so the mean is the same.
    let tree = knn_tree(points);
    let mean_dists: Vec<f64> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let p = &points[i];
            let nn = tree.query(&[p.x, p.y, p.z], k);
            if nn.is_empty() {
                return f64::NAN;
            }
            nn.iter().map(|&(_, d)| d).sum::<f64>() / nn.len() as f64
        })
        .collect();

    // Global mean and sample standard deviation.
    let n = mean_dists.len() as f64;
    let global_mean = mean_dists.iter().sum::<f64>() / n;
    let variance = mean_dists
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let global_std = variance.sqrt();

    let threshold = global_mean + std_ratio * global_std;

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (i, &md) in mean_dists.iter().enumerate() {
        if md < threshold || md == 0.0 {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}
