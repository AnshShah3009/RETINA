//! Camera calibration module
//!
//! Provides functionality for camera calibration using planar patterns (chessboards)
//! and refinement of calibration results through iterative optimization.

use crate::Result;
use cv_core::{CameraIntrinsics, Distortion, Pose};
use image::GrayImage;
use nalgebra::{DMatrix, Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::path::Path;

use crate::pattern::find_chessboard_corners;

#[derive(Debug, Clone)]
pub struct CameraCalibrationResult {
    pub intrinsics: CameraIntrinsics,
    pub extrinsics: Vec<Pose>,
    pub distortion: Distortion,
    pub rms_reprojection_error: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraCalibrationOptions {
    /// Enforce fx/fy to match this ratio (fx = ratio * fy).
    pub fix_aspect_ratio: Option<f64>,
    /// Enforce principal point to these pixel coordinates.
    pub fix_principal_point: Option<(f64, f64)>,
    /// Use provided intrinsics as initial guess (requires external initialization)
    pub use_intrinsic_guess: bool,
    /// Fix tangential distortion coefficients (p1, p2) to zero
    pub zero_tangent_dist: bool,
    /// Fix focal length (fx, fy) during optimization
    pub fix_focal_length: bool,
    /// Fix radial distortion coefficient k1
    pub fix_k1: bool,
    /// Fix radial distortion coefficient k2
    pub fix_k2: bool,
    /// Fix radial distortion coefficient k3
    pub fix_k3: bool,
}

#[derive(Debug, Clone)]
pub struct CalibrationFileReport {
    pub total_images: usize,
    pub used_images: usize,
    pub rejected_images: Vec<usize>,
}

/// Generate 3D object points for a planar chessboard pattern
///
/// The object points are generated in the plane z=0, with x ranging from 0 to
/// (cols-1)*square_size and y ranging from 0 to (rows-1)*square_size.
pub fn generate_chessboard_object_points(
    pattern_size: (usize, usize),
    square_size: f64,
) -> Vec<Point3<f64>> {
    let (cols, rows) = pattern_size;
    let mut points = Vec::with_capacity(cols * rows);
    for y in 0..rows {
        for x in 0..cols {
            points.push(Point3::new(
                x as f64 * square_size,
                y as f64 * square_size,
                0.0,
            ));
        }
    }
    points
}

/// Calibrate camera using planar homographies with default options
///
/// Requires at least 3 views of a planar pattern with 4 or more correspondences
/// per view.
pub fn calibrate_camera_planar(
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
) -> Result<CameraCalibrationResult> {
    calibrate_camera_planar_with_options(
        object_points,
        image_points,
        image_size,
        CameraCalibrationOptions::default(),
    )
}

/// Calibrate camera using planar homographies with options
///
/// This performs closed-form calibration using homography decomposition.
/// The method requires at least 3 views of a planar pattern with 4 or more
/// correspondences per view. Object points must have z=0 (planar).
pub fn calibrate_camera_planar_with_options(
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
    options: CameraCalibrationOptions,
) -> Result<CameraCalibrationResult> {
    if object_points.len() != image_points.len() || object_points.len() < 3 {
        return Err(cv_core::Error::AlgorithmError(
            "calibrate_camera_planar needs >=3 views with matching point sets".to_string(),
        ));
    }

    let mut homographies = Vec::with_capacity(object_points.len());
    for (obj, img) in object_points.iter().zip(image_points.iter()) {
        if obj.len() != img.len() || obj.len() < 4 {
            return Err(cv_core::Error::AlgorithmError(
                "each calibration view needs >=4 correspondences".to_string(),
            ));
        }
        if obj.iter().any(|p| p.z.abs() > 1e-9) {
            return Err(cv_core::Error::AlgorithmError(
                "calibrate_camera_planar expects planar object points (z=0)".to_string(),
            ));
        }
        let obj2d: Vec<Point2<f64>> = obj.iter().map(|p| Point2::new(p.x, p.y)).collect();
        homographies.push(estimate_homography_dlt(&obj2d, img)?);
    }

    let k = intrinsics_from_planar_homographies(&homographies)?;
    let mut fx = k[(0, 0)];
    let mut fy = k[(1, 1)];
    let mut cx = k[(0, 2)];
    let mut cy = k[(1, 2)];
    if let Some(ratio) = options.fix_aspect_ratio {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(cv_core::Error::AlgorithmError(
                "fix_aspect_ratio must be finite and > 0".to_string(),
            ));
        }
        // Closest constrained fit to unconstrained (fx, fy) under fx = ratio * fy.
        fy = (ratio * fx + fy) / (ratio * ratio + 1.0);
        fx = ratio * fy;
    }
    if let Some((fixed_cx, fixed_cy)) = options.fix_principal_point {
        if !fixed_cx.is_finite() || !fixed_cy.is_finite() {
            return Err(cv_core::Error::AlgorithmError(
                "fix_principal_point must be finite".to_string(),
            ));
        }
        cx = fixed_cx;
        cy = fixed_cy;
    }

    // fix_focal_length: freeze fx and fy at their current values during
    // optimization. At this point we simply skip any mutation -- the values
    // produced by the closed-form solver (or constrained by fix_aspect_ratio)
    // are kept as-is. The flag is primarily consumed in the iterative
    // refinement loop below to prevent focal-length updates.

    let intrinsics = CameraIntrinsics::new(fx, fy, cx, cy, image_size.0, image_size.1);
    let k_inv = intrinsics.inverse_matrix();
    let mut extrinsics = Vec::with_capacity(homographies.len());
    for h in &homographies {
        extrinsics.push(extrinsics_from_homography(&k_inv, h)?);
    }

    let rms = compute_rms_reprojection(
        &intrinsics,
        &extrinsics,
        &Distortion::none(),
        object_points,
        image_points,
    )?;
    let distortion = Distortion::none();

    let mut result = CameraCalibrationResult {
        intrinsics,
        extrinsics,
        distortion,
        rms_reprojection_error: rms,
    };
    if !is_valid_camera_calibration(&result) {
        return Err(cv_core::Error::AlgorithmError(
            "calibrate_camera_planar produced non-finite or degenerate calibration".to_string(),
        ));
    }

    // Iteratively refine the closed-form solution. This estimates distortion
    // coefficients (which the closed-form solver leaves at zero) and respects
    // the fix_* flags in `options`. The refined result is only adopted when it
    // does not degrade the reprojection error.
    if let Ok(refined) = refine_camera_calibration_iterative_with_options(
        &result,
        object_points,
        image_points,
        80,
        options,
    ) {
        if is_valid_camera_calibration(&refined)
            && refined.rms_reprojection_error <= result.rms_reprojection_error + 1e-9
        {
            result = refined;
        }
    }
    Ok(result)
}

/// Calibrate camera from chessboard images with default options
pub fn calibrate_camera_from_chessboard_images(
    images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<CameraCalibrationResult> {
    calibrate_camera_from_chessboard_images_with_options(
        images,
        pattern_size,
        square_size,
        CameraCalibrationOptions::default(),
    )
}

/// Calibrate camera from chessboard images with options
pub fn calibrate_camera_from_chessboard_images_with_options(
    images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
    options: CameraCalibrationOptions,
) -> Result<CameraCalibrationResult> {
    if images.is_empty() {
        return Err(cv_core::Error::AlgorithmError(
            "calibrate_camera_from_chessboard_images: images cannot be empty".to_string(),
        ));
    }
    let (w, h) = images[0].dimensions();
    if images.iter().any(|img| img.dimensions() != (w, h)) {
        return Err(cv_core::Error::AlgorithmError(
            "all calibration images must have the same dimensions".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut image_points = Vec::new();
    for img in images {
        if let Ok(corners) = find_chessboard_corners(img, pattern_size) {
            object_points.push(board.clone());
            image_points.push(corners);
        }
    }

    if object_points.len() < 3 {
        return Err(cv_core::Error::AlgorithmError(format!(
            "need at least 3 valid chessboard frames, found {}",
            object_points.len()
        )));
    }

    calibrate_camera_planar_with_options(&object_points, &image_points, (w, h), options)
}

/// Calibrate camera from chessboard image files with default options
pub fn calibrate_camera_from_chessboard_files<P: AsRef<Path>>(
    image_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<(CameraCalibrationResult, CalibrationFileReport)> {
    calibrate_camera_from_chessboard_files_with_options(
        image_paths,
        pattern_size,
        square_size,
        CameraCalibrationOptions::default(),
    )
}

/// Calibrate camera from chessboard image files with options
pub fn calibrate_camera_from_chessboard_files_with_options<P: AsRef<Path>>(
    image_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
    options: CameraCalibrationOptions,
) -> Result<(CameraCalibrationResult, CalibrationFileReport)> {
    if image_paths.is_empty() {
        return Err(cv_core::Error::AlgorithmError(
            "calibration file list cannot be empty".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut image_points = Vec::new();
    let mut rejected = Vec::new();
    let mut expected_dims = None;

    for (idx, path) in image_paths.iter().enumerate() {
        let img = match image::open(path) {
            Ok(i) => i.to_luma8(),
            Err(_) => {
                rejected.push(idx);
                continue;
            }
        };

        if let Some((w, h)) = expected_dims {
            if img.dimensions() != (w, h) {
                rejected.push(idx);
                continue;
            }
        } else {
            expected_dims = Some(img.dimensions());
        }

        match find_chessboard_corners(&img, pattern_size) {
            Ok(corners) => {
                object_points.push(board.clone());
                image_points.push(corners);
            }
            Err(_) => rejected.push(idx),
        }
    }

    if object_points.len() < 3 {
        return Err(cv_core::Error::AlgorithmError(format!(
            "need at least 3 valid chessboard images, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        cv_core::Error::AlgorithmError("no readable images in provided file list".to_string())
    })?;

    let calib = calibrate_camera_planar_with_options(&object_points, &image_points, dims, options)
        .map_err(|e| {
            cv_core::Error::AlgorithmError(format!(
                "camera calibration failed for file subset (used {} / {} images): {}",
                object_points.len(),
                image_paths.len(),
                e
            ))
        })?;
    let report = CalibrationFileReport {
        total_images: image_paths.len(),
        used_images: object_points.len(),
        rejected_images: rejected,
    };
    Ok((calib, report))
}

/// Refine camera calibration iteratively
///
/// Iteratively refines intrinsics, extrinsics, and distortion coefficients.
pub fn refine_camera_calibration_iterative(
    initial: &CameraCalibrationResult,
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    max_iters: usize,
) -> Result<CameraCalibrationResult> {
    refine_camera_calibration_iterative_with_options(
        initial,
        object_points,
        image_points,
        max_iters,
        CameraCalibrationOptions::default(),
    )
}

/// Refine camera calibration iteratively, respecting the given `CameraCalibrationOptions`.
///
/// Iteratively refines intrinsics, extrinsics, and distortion coefficients with a
/// joint Levenberg-Marquardt bundle adjustment. The `fix_*` flags in `options`
/// freeze the corresponding parameters so they are never modified during optimization.
pub fn refine_camera_calibration_iterative_with_options(
    initial: &CameraCalibrationResult,
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    max_iters: usize,
    options: CameraCalibrationOptions,
) -> Result<CameraCalibrationResult> {
    if object_points.len() != image_points.len() || object_points.len() != initial.extrinsics.len()
    {
        return Err(cv_core::Error::AlgorithmError(
            "refine_camera_calibration_iterative: inconsistent input sizes".to_string(),
        ));
    }

    let n_views = initial.extrinsics.len();
    let total_pts: usize = object_points.iter().map(|v| v.len()).sum();
    if total_pts == 0 {
        return Err(cv_core::Error::AlgorithmError(
            "refine_camera_calibration_iterative: no points to refine".to_string(),
        ));
    }

    // Parameter layout:
    // [0..4]   fx, fy, cx, cy
    // [4..9]   k1, k2, p1, p2, k3
    // [9 + 6i .. 9 + 6i + 3]  rotation vector (axis-angle) for view i
    // [9 + 6i + 3 .. 9 + 6i + 6] translation for view i
    let n_params = 9 + 6 * n_views;
    let mut params = Vec::with_capacity(n_params);
    params.push(initial.intrinsics.fx);
    params.push(initial.intrinsics.fy);
    params.push(initial.intrinsics.cx);
    params.push(initial.intrinsics.cy);
    params.push(initial.distortion.k1);
    params.push(initial.distortion.k2);
    params.push(initial.distortion.p1);
    params.push(initial.distortion.p2);
    params.push(initial.distortion.k3);
    for ext in &initial.extrinsics {
        let rv = ext.rotation.scaled_axis();
        params.push(rv.x);
        params.push(rv.y);
        params.push(rv.z);
        params.push(ext.translation.x);
        params.push(ext.translation.y);
        params.push(ext.translation.z);
    }

    // Fixed-parameter mask derived from options.
    let mut fixed = vec![false; n_params];
    if options.fix_focal_length {
        fixed[0] = true;
        fixed[1] = true;
    }
    if let Some(ratio) = options.fix_aspect_ratio {
        if ratio.is_finite() && ratio > 0.0 {
            // fx is derived as ratio * fy, so fx is not an independent parameter.
            fixed[0] = true;
        }
    }
    if let Some((pcx, pcy)) = options.fix_principal_point {
        if pcx.is_finite() && pcy.is_finite() {
            fixed[2] = true;
            fixed[3] = true;
        }
    }
    if options.fix_k1 {
        fixed[4] = true;
    }
    if options.fix_k2 {
        fixed[5] = true;
    }
    if options.zero_tangent_dist {
        fixed[6] = true;
        fixed[7] = true;
    }
    if options.fix_k3 {
        fixed[8] = true;
    }
    let free_indices: Vec<usize> = (0..n_params).filter(|&i| !fixed[i]).collect();
    if free_indices.is_empty() {
        return Ok(initial.clone());
    }

    // Rebuild intrinsics / distortion / poses from the parameter vector.
    let rebuild = |params: &[f64], n_views: usize| -> (CameraIntrinsics, Distortion, Vec<Pose>) {
        let aspect_ratio = options
            .fix_aspect_ratio
            .filter(|r| *r > 0.0 && r.is_finite());
        let fx = if let Some(ratio) = aspect_ratio {
            ratio * params[1]
        } else {
            params[0]
        };
        let intrinsics = CameraIntrinsics::new(
            fx,
            params[1],
            params[2],
            params[3],
            initial.intrinsics.width,
            initial.intrinsics.height,
        );
        let distortion = Distortion {
            k1: params[4],
            k2: params[5],
            p1: if options.zero_tangent_dist {
                0.0
            } else {
                params[6]
            },
            p2: if options.zero_tangent_dist {
                0.0
            } else {
                params[7]
            },
            k3: params[8],
        };
        let mut poses = Vec::with_capacity(n_views);
        for i in 0..n_views {
            let base = 9 + 6 * i;
            let rv = nalgebra::Vector3::new(params[base], params[base + 1], params[base + 2]);
            let t = nalgebra::Vector3::new(params[base + 3], params[base + 4], params[base + 5]);
            poses.push(Pose::from_quat_translation(
                nalgebra::UnitQuaternion::from_scaled_axis(rv),
                t,
            ));
        }
        (intrinsics, distortion, poses)
    };

    // Residuals: 2 per point (u, v reprojection error).
    let residuals = |params: &[f64], n_views: usize| -> Vec<f64> {
        let (intrinsics, distortion, poses) = rebuild(params, n_views);
        let mut out = Vec::with_capacity(2 * total_pts);
        for (i, ext) in poses.iter().enumerate() {
            for (p3, p2) in object_points[i].iter().zip(image_points[i].iter()) {
                let pc = ext.rotation * p3.coords + ext.translation;
                if pc[2].abs() <= 1e-12 {
                    out.push(0.0);
                    out.push(0.0);
                    continue;
                }
                let xn = pc[0] / pc[2];
                let yn = pc[1] / pc[2];
                let (xd, yd) = distortion.apply(xn, yn);
                out.push(intrinsics.fx * xd + intrinsics.cx - p2.x);
                out.push(intrinsics.fy * yd + intrinsics.cy - p2.y);
            }
        }
        out
    };

    let mut lambda = 1e-3;
    let mut best_params = params.clone();
    let mut best_cost = residuals(&params, n_views)
        .iter()
        .map(|v| v * v)
        .sum::<f64>();

    for _ in 0..max_iters {
        let r = residuals(&params, n_views);
        let mut j = DMatrix::<f64>::zeros(r.len(), free_indices.len());
        for (col, &pi) in free_indices.iter().enumerate() {
            // Relative finite-difference step avoids poor conditioning from
            // mixing units (pixel focal lengths vs. dimensionless rotations).
            let scale = 1.0 + params[pi].abs();
            let eps = 1e-7 * scale;
            let mut p_pert = params.clone();
            p_pert[pi] += eps;
            let r_pert = residuals(&p_pert, n_views);
            for (row, (rp, rb)) in r_pert.iter().zip(r.iter()).enumerate() {
                j[(row, col)] = (rp - rb) / eps;
            }
        }

        let jt = j.transpose();
        let h = &jt * &j;
        let g = &jt * nalgebra::DVector::from(r.clone());

        // Column scaling improves conditioning (pixel focal lengths vs. small
        // rotations/distortion). Solve in scaled coordinates then rescale back.
        let col_norm: Vec<f64> = (0..h.ncols())
            .map(|c| (h[(c, c)].abs()).sqrt().max(1e-12))
            .collect();
        let mut hs = h.clone();
        let mut gs = g.clone();
        for (c, &cn) in col_norm.iter().enumerate() {
            for r in 0..hs.nrows() {
                hs[(r, c)] /= cn;
                hs[(c, r)] /= cn;
            }
            gs[c] /= cn;
        }

        // Damped normal equations (Marquardt).
        let mut hd = hs.clone();
        for i in 0..hd.nrows() {
            hd[(i, i)] += lambda * hd[(i, i)].max(1e-12);
        }
        let delta_scaled = match hd.lu().solve(&gs) {
            Some(d) => d,
            None => {
                lambda *= 2.0;
                continue;
            }
        };
        let mut delta = delta_scaled.clone();
        for (c, &cn) in col_norm.iter().enumerate() {
            delta[c] /= cn;
        }

        let mut p_new = params.clone();
        for (col, &pi) in free_indices.iter().enumerate() {
            p_new[pi] -= delta[col];
        }
        // Snap fixed parameters to their constrained values.
        if let Some((pcx, pcy)) = options.fix_principal_point {
            p_new[2] = pcx;
            p_new[3] = pcy;
        }

        let new_cost = residuals(&p_new, n_views)
            .iter()
            .map(|v| v * v)
            .sum::<f64>();
        // Marquardt gain ratio computed in scaled coordinates.
        let predicted = 0.5
            * delta_scaled
                .dot(&(delta_scaled.clone().component_mul(&hs.diagonal()) * lambda + &gs));
        let rho = (best_cost - new_cost) / (predicted.abs().max(1e-300));
        if new_cost < best_cost {
            best_cost = new_cost;
            best_params = p_new.clone();
            params = p_new;
            lambda = (lambda * (1.0 / 3.0)).max(1e-9);
        } else {
            lambda = (lambda * 2.0).min(1e12);
        }
        if lambda > 1e12 || best_cost < 1e-24 {
            break;
        }
        // Stop when the gain ratio indicates convergence to a stationary point.
        if rho < 1e-4 && delta.norm() < 1e-8 {
            break;
        }
    }

    let (intrinsics, distortion, extrinsics) = rebuild(&best_params, n_views);
    let rms = compute_rms_reprojection(
        &intrinsics,
        &extrinsics,
        &distortion,
        object_points,
        image_points,
    )?;
    Ok(CameraCalibrationResult {
        intrinsics,
        extrinsics,
        distortion,
        rms_reprojection_error: rms,
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Estimate homography using Direct Linear Transform (DLT)
fn estimate_homography_dlt(src: &[Point2<f64>], dst: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if src.len() != dst.len() || src.len() < 4 {
        return Err(cv_core::Error::AlgorithmError(
            "estimate_homography_dlt needs >=4 paired points".to_string(),
        ));
    }

    let (src_n, ts) = normalize_points_hartley(src)?;
    let (dst_n, td) = normalize_points_hartley(dst)?;
    let n = src.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let x = src_n[i].x;
        let y = src_n[i].y;
        let u = dst_n[i].x;
        let v = dst_n[i].y;
        let r0 = 2 * i;
        let r1 = r0 + 1;
        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;

        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }

    let svd = a.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD failed in estimate_homography_dlt".to_string())
    })?;
    let h = vt.row(vt.nrows() - 1);
    let hn = Matrix3::new(
        h[(0, 0)],
        h[(0, 1)],
        h[(0, 2)],
        h[(0, 3)],
        h[(0, 4)],
        h[(0, 5)],
        h[(0, 6)],
        h[(0, 7)],
        h[(0, 8)],
    );
    let mut hdenorm = td.try_inverse().unwrap_or(Matrix3::identity()) * hn * ts;
    if hdenorm[(2, 2)].abs() > 1e-12 {
        hdenorm /= hdenorm[(2, 2)];
    }
    Ok(hdenorm)
}

/// Compute intrinsic matrix from planar homographies
fn intrinsics_from_planar_homographies(homographies: &[Matrix3<f64>]) -> Result<Matrix3<f64>> {
    if homographies.len() < 3 {
        return Err(cv_core::Error::AlgorithmError(
            "need at least 3 homographies for planar calibration".to_string(),
        ));
    }

    let mut v = DMatrix::<f64>::zeros(2 * homographies.len(), 6);
    for (i, h) in homographies.iter().enumerate() {
        let v12 = v_ij(h, 0, 1);
        let v11 = v_ij(h, 0, 0);
        let v22 = v_ij(h, 1, 1);
        for j in 0..6 {
            v[(2 * i, j)] = v12[j];
            v[(2 * i + 1, j)] = v11[j] - v22[j];
        }
    }

    let svd = v.svd(true, true);
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError(
            "SVD failed in intrinsics_from_planar_homographies".to_string(),
        )
    })?;
    let b = vt.row(vt.nrows() - 1);
    let mut b11 = b[(0, 0)];
    let mut b12 = b[(0, 1)];
    let mut b22 = b[(0, 2)];
    let mut b13 = b[(0, 3)];
    let mut b23 = b[(0, 4)];
    let mut b33 = b[(0, 5)];

    let mut denom = b11 * b22 - b12 * b12;
    if denom.abs() < 1e-18 || b11.abs() < 1e-18 {
        return Err(cv_core::Error::AlgorithmError(
            "degenerate calibration system".to_string(),
        ));
    }

    let mut v0 = (b12 * b13 - b11 * b23) / denom;
    let mut lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;

    // Nullspace sign is arbitrary; flip once if needed.
    if lambda <= 0.0 {
        b11 = -b11;
        b12 = -b12;
        b22 = -b22;
        b13 = -b13;
        b23 = -b23;
        b33 = -b33;
        denom = b11 * b22 - b12 * b12;
        if denom.abs() < 1e-18 || b11.abs() < 1e-18 {
            return Err(cv_core::Error::AlgorithmError(
                "degenerate calibration system after sign flip".to_string(),
            ));
        }
        v0 = (b12 * b13 - b11 * b23) / denom;
        lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    }
    if lambda <= 0.0 {
        return Err(cv_core::Error::AlgorithmError(
            "invalid lambda in planar calibration".to_string(),
        ));
    }
    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / denom).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    Ok(Matrix3::new(alpha, gamma, u0, 0.0, beta, v0, 0.0, 0.0, 1.0))
}

/// Compute camera extrinsics from homography
fn extrinsics_from_homography(k_inv: &Matrix3<f64>, h: &Matrix3<f64>) -> Result<Pose> {
    let h1 = h.column(0).into_owned();
    let h2 = h.column(1).into_owned();
    let h3 = h.column(2).into_owned();

    let r1_raw = k_inv * h1;
    let r2_raw = k_inv * h2;
    let t_raw = k_inv * h3;
    let scale = 1.0 / r1_raw.norm().max(1e-18);

    let r1 = r1_raw * scale;
    let r2 = r2_raw * scale;
    let r3 = r1.cross(&r2);
    let mut r = Matrix3::from_columns(&[r1, r2, r3]);

    let svd = r.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD U missing in extrinsics_from_homography".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD V^T missing in extrinsics_from_homography".to_string())
    })?;
    r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
    }

    let t = t_raw * scale;
    Ok(Pose::new(r, t))
}

/// Compute RMS reprojection error, accounting for lens distortion when present.
#[allow(clippy::type_complexity)]
fn compute_rms_reprojection(
    intrinsics: &CameraIntrinsics,
    extrinsics: &[Pose],
    distortion: &Distortion,
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
) -> Result<f64> {
    if extrinsics.len() != object_points.len() || object_points.len() != image_points.len() {
        return Err(cv_core::Error::AlgorithmError(
            "compute_rms_reprojection: mismatched batch sizes".to_string(),
        ));
    }

    let (sq_sum, count) = extrinsics
        .par_iter()
        .zip(object_points.par_iter())
        .zip(image_points.par_iter())
        .map(
            |((ext, obj), img): ((&Pose, &Vec<Point3<f64>>), &Vec<Point2<f64>>)| {
                let mut local_sq_sum = 0.0f64;
                let mut local_count = 0usize;
                for (p3, p2) in obj.iter().zip(img.iter()) {
                    let pc = ext.rotation * p3.coords + ext.translation;
                    if pc[2].abs() <= 1e-18 {
                        continue;
                    }
                    let xn = pc[0] / pc[2];
                    let yn = pc[1] / pc[2];
                    let (xd, yd) = distortion.apply(xn, yn);
                    let u = intrinsics.fx * xd + intrinsics.cx;
                    let v = intrinsics.fy * yd + intrinsics.cy;
                    let du = u - p2.x;
                    let dv = v - p2.y;
                    local_sq_sum += du * du + dv * dv;
                    local_count += 1;
                }
                (local_sq_sum, local_count)
            },
        )
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    if count == 0 {
        return Err(cv_core::Error::AlgorithmError(
            "compute_rms_reprojection: no valid points".to_string(),
        ));
    }
    Ok((sq_sum / count as f64).sqrt())
}

/// Check if camera calibration result is valid
fn is_valid_camera_calibration(result: &CameraCalibrationResult) -> bool {
    let k = &result.intrinsics;
    let intrinsics_valid = k.fx.is_finite()
        && k.fy.is_finite()
        && k.cx.is_finite()
        && k.cy.is_finite()
        && k.fx.abs() > 1e-12
        && k.fy.abs() > 1e-12;
    if !intrinsics_valid || !result.rms_reprojection_error.is_finite() {
        return false;
    }

    result.extrinsics.iter().all(|ext| {
        ext.rotation_matrix().iter().all(|v: &f64| v.is_finite())
            && ext.translation.iter().all(|v: &f64| v.is_finite())
    })
}

/// Helper function to compute v_ij for intrinsic calibration
fn v_ij(h: &Matrix3<f64>, i: usize, j: usize) -> [f64; 6] {
    [
        h[(0, i)] * h[(0, j)],
        h[(0, i)] * h[(1, j)] + h[(1, i)] * h[(0, j)],
        h[(1, i)] * h[(1, j)],
        h[(2, i)] * h[(0, j)] + h[(0, i)] * h[(2, j)],
        h[(2, i)] * h[(1, j)] + h[(1, i)] * h[(2, j)],
        h[(2, i)] * h[(2, j)],
    ]
}

/// Normalize points using Hartley normalization
fn normalize_points_hartley(points: &[Point2<f64>]) -> Result<(Vec<Point2<f64>>, Matrix3<f64>)> {
    if points.is_empty() {
        return Err(cv_core::Error::AlgorithmError(
            "normalize_points_hartley: empty points array".to_string(),
        ));
    }

    let mean_x = points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64;

    let mean_dist = points
        .iter()
        .map(|p| ((p.x - mean_x).powi(2) + (p.y - mean_y).powi(2)).sqrt())
        .sum::<f64>()
        / points.len() as f64;

    let scale = if mean_dist.abs() > 1e-18 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let normalized = points
        .iter()
        .map(|p| Point2::new((p.x - mean_x) * scale, (p.y - mean_y) * scale))
        .collect();

    let t = Matrix3::new(
        scale,
        0.0,
        -mean_x * scale,
        0.0,
        scale,
        -mean_y * scale,
        0.0,
        0.0,
        1.0,
    );

    Ok((normalized, t))
}
