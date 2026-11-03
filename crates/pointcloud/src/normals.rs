//! Canonical `f64` k-nearest-neighbour normal estimation.
//!
//! The single implementation behind `cv_3d::filters::estimate_normals_knn`
//! (which orients the result into `z >= 0`) and the `f32`
//! [`crate::point_cloud::estimate_normals`] (which converts to `f64` and keeps
//! the sign as computed).

use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use crate::filters::{FiniteKnn, finite_indices, is_finite_point};

/// The normal given to a point whose neighbourhood cannot determine one.
const DEFAULT_NORMAL: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);

/// Estimate point normals by PCA over the `k` nearest neighbours (Open3D's
/// `estimate_normals(KDTreeSearchParamKNN(k))`).
///
/// The neighbourhood of a point is its `k` nearest points *including the
/// point itself*; `k` larger than the cloud uses the whole cloud, and a
/// neighbourhood of fewer than three points (`k < 3`, or fewer than three
/// finite points) gives the default `(0, 0, 1)`, as Open3D. The normal is the
/// minimum eigenvector of the neighbourhood's covariance, from the analytic
/// solver in `cv_math` (Open3D / Geometric Tools `RobustEigenSymmetric3x3`),
/// with the sign as the solver returns it: no orientation is applied.
///
/// Neighbours come from an `f64` KD-tree, so the neighbour sets and distances
/// are those of a brute-force `f64` scan.
///
/// # Non-finite input
///
/// A point with a NaN or infinite coordinate is left out of the search tree,
/// so it is never in another point's neighbourhood, and gets the default
/// `(0, 0, 1)` itself.
pub fn estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    if points.is_empty() {
        return Vec::new();
    }
    let finite = finite_indices(points);
    // Neighbourhood size, the point included.
    let m = k.min(finite.len());
    if m < 3 {
        return vec![DEFAULT_NORMAL; points.len()];
    }
    let Some(tree) = FiniteKnn::new(points, &finite) else {
        return vec![DEFAULT_NORMAL; points.len()];
    };

    points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            if !is_finite_point(p) {
                return DEFAULT_NORMAL;
            }
            let others = knn_excluding_self(&tree, p, i, m - 1);

            // Covariance of the point and its m - 1 nearest others.
            let mut cov = Matrix3::<f64>::zeros();
            let mut centroid = p.coords;
            for &(j, _) in &others {
                centroid += points[j].coords;
            }
            centroid /= m as f64;

            for &(j, _) in &others {
                let d = points[j].coords - centroid;
                cov += d * d.transpose();
            }
            let d = p.coords - centroid;
            cov += d * d.transpose();

            cv_math::linalg::min_eigenvector_3x3(&cov)
        })
        .collect()
}

/// The `k` nearest neighbours of point `i` (at `p`) other than `i` itself, as
/// `(index, euclidean distance)` in ascending distance; `k` is less than the
/// number of points in the tree.
///
/// Queries `k + 1` neighbours and drops `i`. If `i` is not among them, the
/// `k + 1` results all lie at distance 0 (`i` itself is at 0 and lost a tie),
/// so they coincide with `p` and dropping the last one leaves the same
/// distances and coordinates as dropping `i` would.
fn knn_excluding_self(tree: &FiniteKnn, p: &Point3<f64>, i: usize, k: usize) -> Vec<(usize, f64)> {
    let mut nn = tree.query(p, k + 1);
    match nn.iter().position(|&(j, _)| j == i) {
        Some(pos) => {
            nn.remove(pos);
        }
        None => {
            nn.pop();
        }
    }
    nn.truncate(k);
    nn
}
