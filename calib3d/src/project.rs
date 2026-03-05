//! 3D-to-2D projection functions with optional Jacobian computation
//!
//! This module provides functions to project 3D points to 2D image coordinates
//! using camera intrinsics and extrinsics. Supports optional distortion and
//! Jacobian computation for optimization tasks.

use crate::Result;
use cv_core::{CameraIntrinsics, Distortion, Pose};
use nalgebra::{DMatrix, Matrix3, Point2, Point3, Vector3};
use rayon::prelude::*;

/// Options for projection with optional Jacobian computation
#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectPointsOptions {
    /// Compute Jacobian matrices (adds computational cost)
    pub compute_jacobians: bool,
}

/// Result from projection with optional Jacobians
#[derive(Debug, Clone)]
pub struct ProjectPointsResult {
    /// Projected 2D image points
    pub image_points: Vec<Point2<f64>>,

    /// Jacobian w.r.t. rotation vector (3 params) - shape: 2N × 3
    /// Each point contributes 2 rows: [∂u/∂rx ∂u/∂ry ∂u/∂rz; ∂v/∂rx ∂v/∂ry ∂v/∂rz]
    pub jacobian_rotation: Option<DMatrix<f64>>,

    /// Jacobian w.r.t. translation vector (3 params) - shape: 2N × 3
    pub jacobian_translation: Option<DMatrix<f64>>,

    /// Jacobian w.r.t. intrinsics [fx, fy, cx, cy] - shape: 2N × 4
    pub jacobian_intrinsics: Option<DMatrix<f64>>,

    /// Jacobian w.r.t. distortion [k1, k2, p1, p2, k3] - shape: 2N × 5
    pub jacobian_distortion: Option<DMatrix<f64>>,
}

/// Project 3D points to 2D without distortion
///
/// This is the simplest projection: camera coordinates -> normalized coordinates -> pixel coordinates.
/// Uses only focal length and principal point from camera intrinsics.
///
/// # Arguments
/// * `object_points` - 3D points in world coordinates
/// * `intrinsics` - Camera intrinsic matrix (focal length, principal point)
/// * `extrinsics` - Camera extrinsic matrix (rotation, translation)
///
/// # Returns
/// Vector of projected 2D points in pixel coordinates
///
/// # Errors
/// Returns error if any point has non-finite coordinates or is behind the camera (depth <= 0)
pub fn project_points(
    object_points: &[Point3<f64>],
    intrinsics: &CameraIntrinsics,
    extrinsics: &Pose,
) -> Result<Vec<Point2<f64>>> {
    object_points
        .par_iter()
        .map(|p| {
            let pc = extrinsics.rotation * p.coords + extrinsics.translation;
            if !pc.iter().all(|v: &f64| v.is_finite()) || pc[2].abs() <= 1e-12 {
                return Err(cv_core::Error::CalibrationError(
                    "project_points encountered non-finite or near-zero depth point".to_string(),
                ));
            }
            Ok(Point2::new(
                intrinsics.fx * (pc[0] / pc[2]) + intrinsics.cx,
                intrinsics.fy * (pc[1] / pc[2]) + intrinsics.cy,
            ))
        })
        .collect()
}

/// Project 3D points to 2D with distortion
///
/// This projection applies lens distortion (radial and tangential) after normalization.
/// More realistic for real camera models.
///
/// # Arguments
/// * `object_points` - 3D points in world coordinates
/// * `intrinsics` - Camera intrinsic matrix
/// * `extrinsics` - Camera extrinsic matrix
/// * `distortion` - Distortion coefficients
///
/// # Returns
/// Vector of projected 2D points in pixel coordinates with distortion applied
///
/// # Errors
/// Returns error if any point has non-finite coordinates or is behind the camera
pub fn project_points_with_distortion(
    object_points: &[Point3<f64>],
    intrinsics: &CameraIntrinsics,
    extrinsics: &Pose,
    distortion: &Distortion,
) -> Result<Vec<Point2<f64>>> {
    object_points
        .par_iter()
        .map(|p| {
            let pc = extrinsics.rotation * p.coords + extrinsics.translation;
            if !pc.iter().all(|v: &f64| v.is_finite()) || pc[2].abs() <= 1e-12 {
                return Err(cv_core::Error::CalibrationError(
                    "project_points_with_distortion encountered non-finite or near-zero depth point"
                        .to_string(),
                ));
            }
            let x = pc[0] / pc[2];
            let y = pc[1] / pc[2];
            let (xd, yd) = distortion.apply(x, y);
            Ok(Point2::new(
                intrinsics.fx * xd + intrinsics.cx,
                intrinsics.fy * yd + intrinsics.cy,
            ))
        })
        .collect()
}

/// Project 3D points to 2D with distortion and optional Jacobian computation
///
/// This is the full projection pipeline with analytical Jacobians for optimization.
/// Used in bundle adjustment, calibration refinement, and covariance estimation.
///
/// # Arguments
/// * `object_points` - 3D points in world coordinates
/// * `intrinsics` - Camera intrinsic matrix
/// * `extrinsics` - Camera extrinsic matrix
/// * `distortion` - Distortion coefficients
/// * `options` - Options including whether to compute Jacobians
///
/// # Returns
/// ProjectPointsResult with projected points and optional Jacobian matrices
///
/// # Errors
/// Returns error if any point has non-finite coordinates or is behind the camera
pub fn project_points_with_jacobian(
    object_points: &[Point3<f64>],
    intrinsics: &CameraIntrinsics,
    extrinsics: &Pose,
    distortion: &Distortion,
    options: ProjectPointsOptions,
) -> Result<ProjectPointsResult> {
    let n = object_points.len();
    let mut image_points = Vec::with_capacity(n);

    // Initialize Jacobian matrices if requested
    let mut jac_rot = if options.compute_jacobians {
        Some(DMatrix::zeros(2 * n, 3))
    } else {
        None
    };
    let mut jac_trans = if options.compute_jacobians {
        Some(DMatrix::zeros(2 * n, 3))
    } else {
        None
    };
    let mut jac_intr = if options.compute_jacobians {
        Some(DMatrix::zeros(2 * n, 4))
    } else {
        None
    };
    let mut jac_dist = if options.compute_jacobians {
        Some(DMatrix::zeros(2 * n, 5))
    } else {
        None
    };

    // Convert rotation matrix to axis-angle for Jacobian computation
    let rvec = if options.compute_jacobians {
        Some(rotation_matrix_to_rodrigues(&extrinsics.rotation_matrix()))
    } else {
        None
    };

    for (i, p_obj) in object_points.iter().enumerate() {
        // Transform to camera coordinates
        let p_cam = extrinsics.rotation * p_obj.coords + extrinsics.translation;

        if !p_cam.iter().all(|v| v.is_finite()) || p_cam[2].abs() <= 1e-12 {
            return Err(cv_core::Error::CalibrationError(
                "Projection encountered non-finite or near-zero depth point".to_string(),
            ));
        }

        // Normalized image coordinates
        let x = p_cam[0] / p_cam[2];
        let y = p_cam[1] / p_cam[2];

        // Apply distortion
        let (xd, yd) = distortion.apply(x, y);

        // Project to pixel coordinates
        let u = intrinsics.fx * xd + intrinsics.cx;
        let v = intrinsics.fy * yd + intrinsics.cy;
        image_points.push(Point2::new(u, v));

        // Compute Jacobians if requested
        if let (
            Some(ref mut jr),
            Some(ref mut jt),
            Some(ref mut jk),
            Some(ref mut jd),
            Some(ref rv),
        ) = (
            &mut jac_rot,
            &mut jac_trans,
            &mut jac_intr,
            &mut jac_dist,
            &rvec,
        ) {
            // Chain rule: ∂pixel/∂params = ∂pixel/∂distorted × ∂distorted/∂normalized × ∂normalized/∂camera × ∂camera/∂params
            compute_projection_jacobians(
                i, p_obj, &p_cam, x, y, xd, yd, intrinsics, extrinsics, distortion, rv, jr, jt, jk,
                jd,
            );
        }
    }

    Ok(ProjectPointsResult {
        image_points,
        jacobian_rotation: jac_rot,
        jacobian_translation: jac_trans,
        jacobian_intrinsics: jac_intr,
        jacobian_distortion: jac_dist,
    })
}

/// Convert rotation matrix to Rodrigues axis-angle representation
fn rotation_matrix_to_rodrigues(r: &Matrix3<f64>) -> Vector3<f64> {
    // Use Rodrigues formula: θ = arccos((trace(R) - 1) / 2)
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

    if theta.abs() < 1e-6 {
        // Small angle approximation
        Vector3::new(
            (r[(2, 1)] - r[(1, 2)]) / 2.0,
            (r[(0, 2)] - r[(2, 0)]) / 2.0,
            (r[(1, 0)] - r[(0, 1)]) / 2.0,
        )
    } else {
        let k = theta / (2.0 * theta.sin());
        Vector3::new(
            k * (r[(2, 1)] - r[(1, 2)]),
            k * (r[(0, 2)] - r[(2, 0)]),
            k * (r[(1, 0)] - r[(0, 1)]),
        )
    }
}

/// Convert Rodrigues axis-angle to rotation matrix
fn rodrigues_to_rotation_matrix(rvec: &Vector3<f64>) -> Matrix3<f64> {
    let theta = rvec.norm();

    if theta < 1e-6 {
        // Small angle: R ≈ I + [r]×
        let rx = rvec[0];
        let ry = rvec[1];
        let rz = rvec[2];
        Matrix3::new(1.0, -rz, ry, rz, 1.0, -rx, -ry, rx, 1.0)
    } else {
        // Rodrigues formula: R = I + sin(θ)/θ [r]× + (1-cos(θ))/θ² [r]×²
        let r = rvec / theta;
        let c = theta.cos();
        let s = theta.sin();
        let t = 1.0 - c;

        let rx = r[0];
        let ry = r[1];
        let rz = r[2];

        Matrix3::new(
            t * rx * rx + c,
            t * rx * ry - s * rz,
            t * rx * rz + s * ry,
            t * rx * ry + s * rz,
            t * ry * ry + c,
            t * ry * rz - s * rx,
            t * rx * rz - s * ry,
            t * ry * rz + s * rx,
            t * rz * rz + c,
        )
    }
}

/// Compute all projection Jacobians for a single point
#[allow(clippy::too_many_arguments)]
fn compute_projection_jacobians(
    point_idx: usize,
    p_obj: &Point3<f64>,
    _p_cam: &Vector3<f64>,
    x: f64,
    y: f64,
    xd: f64,
    yd: f64,
    intrinsics: &CameraIntrinsics,
    extrinsics: &Pose,
    distortion: &Distortion,
    rvec: &Vector3<f64>,
    jac_rot: &mut DMatrix<f64>,
    jac_trans: &mut DMatrix<f64>,
    jac_intr: &mut DMatrix<f64>,
    jac_dist: &mut DMatrix<f64>,
) {
    // Use numerical differentiation for now (can optimize to analytical later)
    let eps = 1e-7;
    let row = 2 * point_idx;

    // Jacobian w.r.t. rotation (3 parameters)
    for k in 0..3 {
        let mut rvec_pert = *rvec;
        rvec_pert[k] += eps;
        let r_pert = rodrigues_to_rotation_matrix(&rvec_pert);
        let ext_pert = Pose::new(r_pert, extrinsics.translation);

        let p_cam_pert = ext_pert.rotation * p_obj.coords + ext_pert.translation;
        let x_pert = p_cam_pert[0] / p_cam_pert[2];
        let y_pert = p_cam_pert[1] / p_cam_pert[2];
        let (xd_pert, yd_pert) = distortion.apply(x_pert, y_pert);
        let u_pert = intrinsics.fx * xd_pert + intrinsics.cx;
        let v_pert = intrinsics.fy * yd_pert + intrinsics.cy;

        let u = intrinsics.fx * xd + intrinsics.cx;
        let v = intrinsics.fy * yd + intrinsics.cy;

        jac_rot[(row, k)] = (u_pert - u) / eps;
        jac_rot[(row + 1, k)] = (v_pert - v) / eps;
    }

    // Jacobian w.r.t. translation (3 parameters)
    for k in 0..3 {
        let mut trans_pert = extrinsics.translation;
        trans_pert[k] += eps;

        let p_cam_pert = extrinsics.rotation * p_obj.coords + trans_pert;
        let x_pert = p_cam_pert[0] / p_cam_pert[2];
        let y_pert = p_cam_pert[1] / p_cam_pert[2];
        let (xd_pert, yd_pert) = distortion.apply(x_pert, y_pert);
        let u_pert = intrinsics.fx * xd_pert + intrinsics.cx;
        let v_pert = intrinsics.fy * yd_pert + intrinsics.cy;

        let u = intrinsics.fx * xd + intrinsics.cx;
        let v = intrinsics.fy * yd + intrinsics.cy;

        jac_trans[(row, k)] = (u_pert - u) / eps;
        jac_trans[(row + 1, k)] = (v_pert - v) / eps;
    }

    // Jacobian w.r.t. intrinsics [fx, fy, cx, cy]
    let u_base = intrinsics.fx * xd + intrinsics.cx;
    let v_base = intrinsics.fy * yd + intrinsics.cy;

    jac_intr[(row, 0)] = xd; // ∂u/∂fx
    jac_intr[(row, 1)] = 0.0; // ∂u/∂fy
    jac_intr[(row, 2)] = 1.0; // ∂u/∂cx
    jac_intr[(row, 3)] = 0.0; // ∂u/∂cy

    jac_intr[(row + 1, 0)] = 0.0; // ∂v/∂fx
    jac_intr[(row + 1, 1)] = yd; // ∂v/∂fy
    jac_intr[(row + 1, 2)] = 0.0; // ∂v/∂cx
    jac_intr[(row + 1, 3)] = 1.0; // ∂v/∂cy

    // Jacobian w.r.t. distortion [k1, k2, p1, p2, k3]
    for k in 0..5 {
        let mut dist_pert = *distortion;
        match k {
            0 => dist_pert.k1 += eps,
            1 => dist_pert.k2 += eps,
            2 => dist_pert.p1 += eps,
            3 => dist_pert.p2 += eps,
            4 => dist_pert.k3 += eps,
            _ => unreachable!(),
        }

        let (xd_pert, yd_pert) = dist_pert.apply(x, y);
        let u_pert = intrinsics.fx * xd_pert + intrinsics.cx;
        let v_pert = intrinsics.fy * yd_pert + intrinsics.cy;

        jac_dist[(row, k)] = (u_pert - u_base) / eps;
        jac_dist[(row + 1, k)] = (v_pert - v_base) / eps;
    }
}
