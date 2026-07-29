use nalgebra::Point3;
use rayon::prelude::*;

/// Radius outlier removal (Open3D equivalent).
///
/// Removes points that have fewer than `min_neighbors` neighbours within the
/// given `radius`. Uses HashGrid for O(n) amortized instead of O(n²) brute force.
///
/// # Returns
/// `(inlier_points, inlier_indices)`.
pub fn radius_outlier_removal(
    points: &[Point3<f64>],
    radius: f64,
    min_neighbors: usize,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    use crate::spatial::hash_grid::HashGrid;

    // Build HashGrid in f32 (sufficient precision for spatial hashing)
    let pts_f32: Vec<nalgebra::Point3<f32>> = points
        .iter()
        .map(|p| nalgebra::Point3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let grid = HashGrid::build(&pts_f32, radius as f32);

    let counts: Vec<usize> = pts_f32
        .par_iter()
        .map(|p| {
            // radius_search returns all within radius including self
            let neighbors = grid.radius_search(p, radius as f32);
            // Subtract 1 for self (self is always found)
            neighbors.len().saturating_sub(1)
        })
        .collect();

    let mut inlier_points = Vec::new();
    let mut inlier_indices = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        if c >= min_neighbors {
            inlier_points.push(points[i]);
            inlier_indices.push(i);
        }
    }

    (inlier_points, inlier_indices)
}
