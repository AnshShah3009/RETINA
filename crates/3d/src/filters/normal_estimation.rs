use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use super::dist_sq;

/// Estimate point cloud normals using PCA over k nearest neighbours.
///
/// This is a standalone, CPU-only implementation that does not depend on the
/// HAL or GPU.  Uses the analytic 3x3 eigensolver (trigonometric eigenvalues +
/// best cross-product eigenvector) for the minimum eigenvector.
///
/// Normals are oriented towards the positive-z half-space by default (the
/// convention used by Open3D when no viewpoint is specified).
pub fn estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    if points.is_empty() {
        return Vec::new();
    }
    // A single point has no neighbours to estimate a plane from; return the
    // default +Z orientation rather than panicking on an empty neighbour list.
    if points.len() == 1 {
        return vec![Vector3::new(0.0, 0.0, 1.0)];
    }
    let k = k.min(points.len() - 1).max(1);

    points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            // Find k nearest neighbours (brute force).
            let mut dists: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, q)| (j, dist_sq(p, q)))
                .collect();
            dists.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Build covariance matrix from k nearest.
            let mut cov = Matrix3::<f64>::zeros();
            let mut centroid = Vector3::zeros();
            for &(j, _) in &dists[..k] {
                centroid += points[j].coords;
            }
            centroid += p.coords;
            centroid /= (k + 1) as f64;

            for &(j, _) in &dists[..k] {
                let d = points[j].coords - centroid;
                cov += d * d.transpose();
            }
            let d = p.coords - centroid;
            cov += d * d.transpose();

            // Analytic min-eigenvector via symmetric 3x3 eigensolver.
            let normal = min_eigenvector_3x3(&cov);

            // Orient towards +Z (Open3D default when no viewpoint).
            if normal.z < 0.0 {
                -normal
            } else {
                normal
            }
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
