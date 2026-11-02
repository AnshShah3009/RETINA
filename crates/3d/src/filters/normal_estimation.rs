use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use super::{knn_excluding_self, knn_tree};

/// Estimate point cloud normals using PCA over k nearest neighbours.
///
/// The neighbourhood of a point is its `k` nearest points *including the
/// point itself*, as Open3D's `estimate_normals(KDTreeSearchParamKNN(k))`;
/// `k` larger than the cloud uses the whole cloud, and a neighbourhood of
/// fewer than three points (`k < 3`, or a cloud of one or two points) gives
/// the default `(0, 0, 1)`, as Open3D. This used to take the point plus `k`
/// others, one more than Open3D (see `tests/normals_knn_open3d.rs`).
///
/// This is a standalone, CPU-only implementation that does not depend on the
/// HAL or GPU.  Uses the analytic 3x3 eigensolver (trigonometric eigenvalues +
/// best cross-product eigenvector) for the minimum eigenvector.
///
/// Normals are oriented into the `z >= 0` half-space. Open3D leaves the sign
/// of a freshly estimated normal to its eigensolver, so the two agree up to
/// sign.
pub fn estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    if points.is_empty() {
        return Vec::new();
    }
    // Neighbourhood size, the point included.
    let m = k.min(points.len());
    if m < 3 {
        return vec![Vector3::new(0.0, 0.0, 1.0); points.len()];
    }

    // k nearest neighbours from an f64 KD-tree: O(n log n) instead of an O(n)
    // scan plus a Vec per point (see `knn_excluding_self`).
    let tree = knn_tree(points);
    points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let dists = knn_excluding_self(&tree, points, i, m - 1);

            // Covariance of the point and its m - 1 nearest others.
            let mut cov = Matrix3::<f64>::zeros();
            let mut centroid = p.coords;
            for &(j, _) in &dists {
                centroid += points[j].coords;
            }
            centroid /= m as f64;

            for &(j, _) in &dists {
                let d = points[j].coords - centroid;
                cov += d * d.transpose();
            }
            let d = p.coords - centroid;
            cov += d * d.transpose();

            // Analytic min-eigenvector via symmetric 3x3 eigensolver.
            let normal = min_eigenvector_3x3(&cov);

            if normal.z < 0.0 { -normal } else { normal }
        })
        .collect()
}

/// Analytic minimum eigenvector of a symmetric 3x3 matrix.
///
/// Delegates to the single shared implementation in `cv-math`
/// (Open3D / Geometric Tools `RobustEigenSymmetric3x3`), which the f32
/// GPU / point-cloud paths use as well.
fn min_eigenvector_3x3(m: &Matrix3<f64>) -> Vector3<f64> {
    cv_math::linalg::min_eigenvector_3x3(m)
}
