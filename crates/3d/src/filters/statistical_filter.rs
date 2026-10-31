use nalgebra::Point3;
use rayon::prelude::*;

use super::{knn_excluding_self, knn_tree};

/// Statistical outlier removal (Open3D equivalent).
///
/// For each point, computes the mean distance to its `nb_neighbors` nearest
/// neighbours.  Points whose mean distance exceeds
/// `global_mean + std_ratio * global_std` are classified as outliers.
///
/// # Returns
/// `(inlier_points, inlier_indices)`.
pub fn statistical_outlier_removal(
    points: &[Point3<f64>],
    nb_neighbors: usize,
    std_ratio: f64,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    if points.len() <= nb_neighbors {
        let indices: Vec<usize> = (0..points.len()).collect();
        return (points.to_vec(), indices);
    }

    let k = nb_neighbors.max(1).min(points.len() - 1);

    // Mean distance to the k nearest neighbours of every point (parallel),
    // via an f64 KD-tree: O(n log n) instead of an O(n) scan plus a Vec per
    // point. The neighbour distances are bit-identical to the brute-force
    // scan's; they are now summed in ascending order.
    let tree = knn_tree(points);
    let mean_dists: Vec<f64> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let sum: f64 = knn_excluding_self(&tree, points, i, k)
                .iter()
                .map(|&(_, d)| d)
                .sum();
            sum / k as f64
        })
        .collect();

    // Global mean and standard deviation.
    let n = mean_dists.len() as f64;
    let global_mean = mean_dists.iter().sum::<f64>() / n;
    let variance = mean_dists
        .iter()
        .map(|d| (d - global_mean).powi(2))
        .sum::<f64>()
        / n;
    let global_std = variance.sqrt();

    let threshold = global_mean + std_ratio * global_std;

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (i, &md) in mean_dists.iter().enumerate() {
        if md <= threshold {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}
