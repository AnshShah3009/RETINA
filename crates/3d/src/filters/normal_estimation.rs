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
/// Uses the Geometric Tools / Open3D `RobustEigenSymmetric3x3` method:
/// trigonometric eigenvalues + best cross-product eigenvector.
fn min_eigenvector_3x3(m: &Matrix3<f64>) -> Vector3<f64> {
    // Eigenvalues of a 3x3 symmetric matrix via Cardano's formula.
    let a00 = m[(0, 0)];
    let a01 = m[(0, 1)];
    let a02 = m[(0, 2)];
    let a11 = m[(1, 1)];
    let a12 = m[(1, 2)];
    let a22 = m[(2, 2)];

    let c0 = a00 * a11 * a22 + 2.0 * a01 * a02 * a12
        - a00 * a12 * a12
        - a11 * a02 * a02
        - a22 * a01 * a01;
    let c1 = a00 * a11 - a01 * a01 + a00 * a22 - a02 * a02 + a11 * a22 - a12 * a12;
    let c2 = a00 + a11 + a22;

    let c2_over_3 = c2 / 3.0;
    let a_val = c1 / 3.0 - c2_over_3 * c2_over_3;
    let half_b = 0.5 * (c0 + c2_over_3 * (2.0 * c2_over_3 * c2_over_3 - c1));

    // Clamp to avoid NaN from sqrt of negative due to numerical noise.
    let q = (a_val * a_val * a_val).min(0.0);
    let sqrt_neg_q = (-q).sqrt();
    let magnitude = if sqrt_neg_q > 1e-30 {
        sqrt_neg_q
    } else {
        1e-30
    };
    let angle = (-half_b / magnitude).clamp(-1.0, 1.0).acos() / 3.0;
    let two_sqrt_neg_a = 2.0 * (-a_val).max(0.0).sqrt();

    let mut evals = [
        c2_over_3 + two_sqrt_neg_a * angle.cos(),
        c2_over_3 - two_sqrt_neg_a * (angle + std::f64::consts::FRAC_PI_3).cos(),
        c2_over_3 - two_sqrt_neg_a * (angle - std::f64::consts::FRAC_PI_3).cos(),
    ];

    // Sort to find minimum eigenvalue.
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lambda_min = evals[0];

    // Eigenvector: best cross-product of rows of (M - lambda_min * I).
    let shifted = Matrix3::new(
        a00 - lambda_min,
        a01,
        a02,
        a01,
        a11 - lambda_min,
        a12,
        a02,
        a12,
        a22 - lambda_min,
    );

    let r0 = Vector3::new(shifted[(0, 0)], shifted[(0, 1)], shifted[(0, 2)]);
    let r1 = Vector3::new(shifted[(1, 0)], shifted[(1, 1)], shifted[(1, 2)]);
    let r2 = Vector3::new(shifted[(2, 0)], shifted[(2, 1)], shifted[(2, 2)]);

    // Pick the cross product with the largest magnitude.
    let c01 = r0.cross(&r1);
    let c02 = r0.cross(&r2);
    let c12 = r1.cross(&r2);

    let n01 = c01.norm_squared();
    let n02 = c02.norm_squared();
    let n12 = c12.norm_squared();

    let best = if n01 >= n02 && n01 >= n12 {
        c01
    } else if n02 >= n12 {
        c02
    } else {
        c12
    };

    let len = best.norm();
    if len > 1e-15 {
        best / len
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}
