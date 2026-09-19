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
        .enumerate()
        .map(|(self_idx, p)| {
            // `radius_search` returns every point within the radius including
            // self, and can report the same index more than once when distinct
            // cells collide in the hash table. Deduplicate and drop self so
            // each neighbour is counted exactly once (a duplicated self-hit
            // would otherwise keep true outliers).
            let neighbors = grid.radius_search(p, radius as f32);
            let mut unique: std::collections::HashSet<usize> =
                std::collections::HashSet::with_capacity(neighbors.len());
            for (idx, _) in neighbors {
                if idx != self_idx {
                    unique.insert(idx);
                }
            }
            unique.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radius_outlier_removal_keeps_dense_cluster() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.05, 0.0, 0.0),
            Point3::new(0.0, 0.05, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ];
        // Each cluster point has two neighbours within 0.2; the isolated point
        // has none, so it must be reported as an outlier.
        let (inliers, indices) = radius_outlier_removal(&points, 0.2, 2);
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(inliers.len(), 3);
    }
}
