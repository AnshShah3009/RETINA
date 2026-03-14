use crate::Result;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{DMatrix, Matrix3, Point2, Vector3};

/// Compute Essential matrix from camera extrinsics (rotation and translation).
///
/// The essential matrix is derived from: E = [t]_x * R
/// where [t]_x is the skew-symmetric matrix of the translation vector.
pub fn essential_from_extrinsics(extrinsics: &Pose) -> Matrix3<f64> {
    use cv_core::skew_symmetric;
    skew_symmetric(&extrinsics.translation) * extrinsics.rotation_matrix()
}

/// Compute Fundamental matrix from Essential matrix using camera intrinsics.
///
/// F = K2^-T * E * K1^-1
/// where K1 and K2 are the inverse camera intrinsic matrices.
pub fn fundamental_from_essential(
    essential: &Matrix3<f64>,
    intrinsics1: &CameraIntrinsics,
    intrinsics2: &CameraIntrinsics,
) -> Matrix3<f64> {
    let k1_inv = intrinsics1.inverse_matrix();
    let k2_inv_t = intrinsics2.inverse_matrix().transpose();
    k2_inv_t * essential * k1_inv
}

/// Estimate Essential matrix from two sets of corresponding points.
///
/// Uses the 8-point algorithm with point normalization via intrinsics.
/// Requires at least 8 point pairs.
pub fn find_essential_mat(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(cv_core::Error::CalibrationError(
            "find_essential_mat needs >=8 paired points".to_string(),
        ));
    }
    let (n1, n2) = normalize_with_intrinsics(pts1, pts2, intrinsics);
    estimate_essential_8_point(&n1, &n2)
}

/// Estimate Fundamental matrix from two sets of corresponding points.
///
/// Uses the 8-point algorithm with Hartley normalization for numerical stability.
/// Requires at least 8 point pairs.
pub fn find_fundamental_mat(pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(cv_core::Error::CalibrationError(
            "find_fundamental_mat needs >=8 paired points".to_string(),
        ));
    }
    estimate_fundamental_8_point(pts1, pts2)
}

use cv_core::{Ransac, RobustConfig, RobustModel};

pub struct EssentialEstimator;

impl RobustModel<(Point2<f64>, Point2<f64>)> for EssentialEstimator {
    type Model = Matrix3<f64>;
    fn min_sample_size(&self) -> usize {
        8
    }
    fn estimate(&self, data: &[&(Point2<f64>, Point2<f64>)]) -> Option<Self::Model> {
        let pts1: Vec<Point2<f64>> = data.iter().map(|p| p.0).collect();
        let pts2: Vec<Point2<f64>> = data.iter().map(|p| p.1).collect();
        estimate_essential_8_point(&pts1, &pts2).ok()
    }
    fn compute_error(&self, model: &Self::Model, data: &(Point2<f64>, Point2<f64>)) -> f64 {
        sampson_error(model, &data.0, &data.1)
    }
}

pub struct FundamentalEstimator;

impl RobustModel<(Point2<f64>, Point2<f64>)> for FundamentalEstimator {
    type Model = Matrix3<f64>;
    fn min_sample_size(&self) -> usize {
        8
    }
    fn estimate(&self, data: &[&(Point2<f64>, Point2<f64>)]) -> Option<Self::Model> {
        let pts1: Vec<Point2<f64>> = data.iter().map(|p| p.0).collect();
        let pts2: Vec<Point2<f64>> = data.iter().map(|p| p.1).collect();
        estimate_fundamental_8_point(&pts1, &pts2).ok()
    }
    fn compute_error(&self, model: &Self::Model, data: &(Point2<f64>, Point2<f64>)) -> f64 {
        sampson_error(model, &data.0, &data.1)
    }
}

/// Estimate Essential matrix with RANSAC for robust outlier rejection.
pub fn find_essential_mat_ransac(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    threshold_px: f64,
    max_iters: usize,
) -> Result<(Matrix3<f64>, Vec<bool>)> {
    let (n1, n2) = normalize_with_intrinsics(pts1, pts2, intrinsics);
    let data: Vec<(Point2<f64>, Point2<f64>)> = n1.into_iter().zip(n2).collect();

    let f = 0.5 * (intrinsics.fx + intrinsics.fy);
    let thresh_norm = threshold_px / f.max(1e-12);

    let config = RobustConfig {
        threshold: thresh_norm * thresh_norm,
        max_iterations: max_iters,
        confidence: 0.99,
    };

    let ransac = Ransac::new(config);
    let res = ransac.run(&EssentialEstimator, &data);

    let model = res
        .model
        .ok_or_else(|| cv_core::Error::CalibrationError("RANSAC failed".into()))?;
    Ok((model, res.inliers))
}

/// Estimate Fundamental matrix with RANSAC for robust outlier rejection.
pub fn find_fundamental_mat_ransac(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    threshold_px: f64,
    max_iters: usize,
) -> Result<(Matrix3<f64>, Vec<bool>)> {
    let data: Vec<(Point2<f64>, Point2<f64>)> =
        pts1.iter().cloned().zip(pts2.iter().cloned()).collect();

    let config = RobustConfig {
        threshold: threshold_px * threshold_px,
        max_iterations: max_iters,
        confidence: 0.99,
    };

    let ransac = Ransac::new(config);
    let res = ransac.run(&FundamentalEstimator, &data);

    let model = res
        .model
        .ok_or_else(|| cv_core::Error::CalibrationError("RANSAC failed".into()))?;
    Ok((model, res.inliers))
}

// Helper functions

/// Normalize pixel coordinates using camera intrinsics.
///
/// Converts from pixel coordinates to normalized image coordinates
/// using the inverse camera intrinsic matrix.
fn normalize_with_intrinsics(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> (Vec<Point2<f64>>, Vec<Point2<f64>>) {
    let k_inv = intrinsics.inverse_matrix();
    let n1: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    let n2: Vec<Point2<f64>> = pts2
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    (n1, n2)
}

/// Estimate Essential matrix using the 8-point algorithm.
///
/// Solves the linear system and applies rank-2 constraint enforcement
/// (two equal singular values, third is zero).
fn estimate_essential_8_point(pts1: &[Point2<f64>], pts2: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if pts1.len() != pts2.len() || pts1.len() < 8 {
        return Err(cv_core::Error::CalibrationError(
            "estimate_essential_8_point needs >=8 paired points".to_string(),
        ));
    }

    let n = pts1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = pts1[i].x;
        let y1 = pts1[i].y;
        let x2 = pts2[i].x;
        let y2 = pts2[i].y;
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD failed in estimate_essential_8_point".to_string())
    })?;
    let evec = vt.row(vt.nrows() - 1);
    let e = Matrix3::new(
        evec[(0, 0)],
        evec[(0, 1)],
        evec[(0, 2)],
        evec[(0, 3)],
        evec[(0, 4)],
        evec[(0, 5)],
        evec[(0, 6)],
        evec[(0, 7)],
        evec[(0, 8)],
    );
    enforce_essential_constraints(&e)
}

/// Enforce the rank-2 constraint on the Essential matrix.
///
/// Uses SVD to set the smallest singular value to zero,
/// ensuring the matrix has rank 2 (required property of Essential matrix).
fn enforce_essential_constraints(e: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = e.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD U missing in essential constraints".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD V^T missing in essential constraints".to_string())
    })?;
    let s = 0.5 * (svd.singular_values[0] + svd.singular_values[1]);
    let sigma = Matrix3::new(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0);
    Ok(u * sigma * vt)
}

/// Estimate Fundamental matrix using the 8-point algorithm.
///
/// Includes Hartley normalization for numerical stability and
/// enforces the rank-2 constraint on the resulting matrix.
fn estimate_fundamental_8_point(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
) -> Result<Matrix3<f64>> {
    let (n1, t1) = normalize_points_hartley(pts1)?;
    let (n2, t2) = normalize_points_hartley(pts2)?;
    let n = n1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let x1 = n1[i].x;
        let y1 = n1[i].y;
        let x2 = n2[i].x;
        let y2 = n2[i].y;
        a[(i, 0)] = x2 * x1;
        a[(i, 1)] = x2 * y1;
        a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;
        a[(i, 4)] = y2 * y1;
        a[(i, 5)] = y2;
        a[(i, 6)] = x1;
        a[(i, 7)] = y1;
        a[(i, 8)] = 1.0;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD failed in estimate_fundamental_8_point".to_string())
    })?;
    let fvec = vt.row(vt.nrows() - 1);
    let f0 = Matrix3::new(
        fvec[(0, 0)],
        fvec[(0, 1)],
        fvec[(0, 2)],
        fvec[(0, 3)],
        fvec[(0, 4)],
        fvec[(0, 5)],
        fvec[(0, 6)],
        fvec[(0, 7)],
        fvec[(0, 8)],
    );
    let f_rank2 = enforce_rank2(&f0)?;
    let f = t2.transpose() * f_rank2 * t1;
    Ok(f)
}

/// Hartley normalization: translate to centroid and scale for unit mean distance.
///
/// Improves numerical stability during matrix estimation by centering points
/// at the origin and scaling so that the mean distance is sqrt(2).
fn normalize_points_hartley(pts: &[Point2<f64>]) -> Result<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if pts.len() < 2 {
        return Err(cv_core::Error::CalibrationError(
            "normalize_points_hartley requires at least 2 points".to_string(),
        ));
    }

    let mx = pts.iter().map(|p| p.x).sum::<f64>() / pts.len() as f64;
    let my = pts.iter().map(|p| p.y).sum::<f64>() / pts.len() as f64;
    let mean_dist = pts
        .iter()
        .map(|p| ((p.x - mx) * (p.x - mx) + (p.y - my) * (p.y - my)).sqrt())
        .sum::<f64>()
        / pts.len() as f64;
    if mean_dist <= 1e-12 {
        return Err(cv_core::Error::CalibrationError(
            "degenerate points in normalize_points_hartley".to_string(),
        ));
    }

    let s = (2.0f64).sqrt() / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts
        .iter()
        .map(|p| {
            let v = t * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0], v[1])
        })
        .collect();
    Ok((out, t))
}

/// Enforce rank-2 constraint using SVD.
///
/// Sets the smallest singular value to zero, ensuring the matrix
/// has rank 2 (required property of Fundamental matrix).
fn enforce_rank2(m: &Matrix3<f64>) -> Result<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD U missing in enforce_rank2".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD V^T missing in enforce_rank2".to_string())
    })?;
    let sigma = Matrix3::new(
        svd.singular_values[0],
        0.0,
        0.0,
        0.0,
        svd.singular_values[1],
        0.0,
        0.0,
        0.0,
        0.0,
    );
    Ok(u * sigma * vt)
}

/// Compute the Sampson error for a point pair against a matrix.
///
/// The Sampson distance is a geometric error measure used in RANSAC
/// for robust outlier rejection in epipolar geometry.
fn sampson_error(e: &Matrix3<f64>, p1: &Point2<f64>, p2: &Point2<f64>) -> f64 {
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let x2 = Vector3::new(p2.x, p2.y, 1.0);
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let x2tex1 = x2.dot(&ex1);
    let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    if denom <= 1e-18 {
        f64::INFINITY
    } else {
        (x2tex1 * x2tex1) / denom
    }
}

/// Sample unique random indices for RANSAC.
#[allow(dead_code)]
fn sample_unique_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    let mut used = vec![false; n];
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    while out.len() < k {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (state as usize) % n;
        if !used[idx] {
            used[idx] = true;
            out.push(idx);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::{CameraIntrinsics, Pose};
    use nalgebra::{Matrix3, Point2, Point3, Vector3};

    #[test]
    fn test_fundamental_epipolar_constraint() {
        // Generate synthetic point pairs from a known camera motion,
        // compute the fundamental matrix, and verify x2^T * F * x1 ~ 0.
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

        // Camera 2 is rotated slightly around Y and translated along X.
        let angle = 0.1_f64; // ~5.7 degrees
        let r = Matrix3::new(
            angle.cos(),
            0.0,
            angle.sin(),
            0.0,
            1.0,
            0.0,
            -angle.sin(),
            0.0,
            angle.cos(),
        );
        let t = Vector3::new(0.5, 0.0, 0.0);
        let extrinsics = Pose::new(r, t);

        // Derive the ground-truth fundamental matrix.
        let e = essential_from_extrinsics(&extrinsics);
        let f_gt = fundamental_from_essential(&e, &intrinsics, &intrinsics);

        // Generate 3D world points in front of the cameras.
        let world_points = vec![
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 0.5, 4.0),
            Point3::new(-0.5, -0.3, 6.0),
            Point3::new(0.3, -0.8, 3.5),
            Point3::new(-1.0, 0.7, 7.0),
            Point3::new(0.8, 0.2, 4.5),
            Point3::new(-0.2, -0.5, 5.5),
            Point3::new(0.5, 1.0, 3.0),
            Point3::new(-0.7, 0.3, 6.5),
            Point3::new(1.2, -0.4, 4.2),
        ];

        // Project into camera 1 (at identity) and camera 2 (at extrinsics).
        let rot = extrinsics.rotation_matrix();
        let pts1: Vec<Point2<f64>> = world_points.iter().map(|p| intrinsics.project(p)).collect();
        let pts2: Vec<Point2<f64>> = world_points
            .iter()
            .map(|p| {
                let p_cam2 = Point3::from(rot * p.coords + t);
                intrinsics.project(&p_cam2)
            })
            .collect();

        // Verify epipolar constraint: x2^T * F * x1 should be ~ 0.
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = Vector3::new(p1.x, p1.y, 1.0);
            let x2 = Vector3::new(p2.x, p2.y, 1.0);
            let val = x2.dot(&(f_gt * x1));
            assert!(
                val.abs() < 1e-6,
                "Epipolar constraint violated: x2^T*F*x1 = {} for p1={:?}, p2={:?}",
                val,
                p1,
                p2
            );
        }

        // Now also estimate F from the point correspondences and verify
        // the estimated F also satisfies the constraint (looser tolerance
        // due to numerical estimation).
        let f_est = find_fundamental_mat(&pts1, &pts2).expect("should estimate F");
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = Vector3::new(p1.x, p1.y, 1.0);
            let x2 = Vector3::new(p2.x, p2.y, 1.0);
            let val = x2.dot(&(f_est * x1));
            assert!(
                val.abs() < 0.01,
                "Estimated F epipolar constraint violated: x2^T*F*x1 = {} for p1={:?}, p2={:?}",
                val,
                p1,
                p2
            );
        }
    }
}
