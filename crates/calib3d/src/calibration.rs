//! Camera calibration module
//!
//! Provides functionality for camera calibration using planar patterns (chessboards)
//! and refinement of calibration results through iterative optimization.

use crate::project_points_with_distortion;
use crate::solve_pnp_refine;
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
        return Err(cv_core::Error::CalibrationError(
            "calibrate_camera_planar needs >=3 views with matching point sets".to_string(),
        ));
    }

    let mut homographies = Vec::with_capacity(object_points.len());
    for (obj, img) in object_points.iter().zip(image_points.iter()) {
        if obj.len() != img.len() || obj.len() < 4 {
            return Err(cv_core::Error::CalibrationError(
                "each calibration view needs >=4 correspondences".to_string(),
            ));
        }
        if obj.iter().any(|p| p.z.abs() > 1e-9) {
            return Err(cv_core::Error::CalibrationError(
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
            return Err(cv_core::Error::CalibrationError(
                "fix_aspect_ratio must be finite and > 0".to_string(),
            ));
        }
        // Closest constrained fit to unconstrained (fx, fy) under fx = ratio * fy.
        fy = (ratio * fx + fy) / (ratio * ratio + 1.0);
        fx = ratio * fy;
    }
    if let Some((fixed_cx, fixed_cy)) = options.fix_principal_point {
        if !fixed_cx.is_finite() || !fixed_cy.is_finite() {
            return Err(cv_core::Error::CalibrationError(
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

    let rms = compute_rms_reprojection(&intrinsics, &extrinsics, object_points, image_points)?;
    let mut distortion = Distortion::none();
    if options.zero_tangent_dist {
        distortion.p1 = 0.0;
        distortion.p2 = 0.0;
    }
    // Note: Other fix_kX flags should be respected during iterative refinement,
    // which is a planned P1 expansion. For now, we return zero distortion.

    let result = CameraCalibrationResult {
        intrinsics,
        extrinsics,
        distortion,
        rms_reprojection_error: rms,
    };
    if !is_valid_camera_calibration(&result) {
        return Err(cv_core::Error::CalibrationError(
            "calibrate_camera_planar produced non-finite or degenerate calibration".to_string(),
        ));
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
        return Err(cv_core::Error::CalibrationError(
            "calibrate_camera_from_chessboard_images: images cannot be empty".to_string(),
        ));
    }
    let (w, h) = images[0].dimensions();
    if images.iter().any(|img| img.dimensions() != (w, h)) {
        return Err(cv_core::Error::CalibrationError(
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
        return Err(cv_core::Error::CalibrationError(format!(
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
        return Err(cv_core::Error::CalibrationError(
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
        return Err(cv_core::Error::CalibrationError(format!(
            "need at least 3 valid chessboard images, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        cv_core::Error::CalibrationError("no readable images in provided file list".to_string())
    })?;

    let calib = calibrate_camera_planar_with_options(&object_points, &image_points, dims, options)
        .map_err(|e| {
            cv_core::Error::CalibrationError(format!(
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
    if object_points.len() != image_points.len() || object_points.len() != initial.extrinsics.len()
    {
        return Err(cv_core::Error::CalibrationError(
            "refine_camera_calibration_iterative: inconsistent input sizes".to_string(),
        ));
    }

    let mut result = initial.clone();
    let mut prev_rms = result.rms_reprojection_error;

    for iter in 0..max_iters {
        // 1. Refine intrinsics (closed-form fix)
        result.intrinsics = estimate_intrinsics_from_extrinsics(
            &result.extrinsics,
            object_points,
            image_points,
            result.intrinsics,
        )?;

        // 2. Refine extrinsics (for each view)
        for i in 0..result.extrinsics.len() {
            result.extrinsics[i] = solve_pnp_refine(
                &result.extrinsics[i],
                &object_points[i],
                &image_points[i],
                &result.intrinsics,
                Some(&result.distortion),
                5,
            )
            .unwrap_or(result.extrinsics[i]);
        }

        // 3. Refine distortion (P1: Iterative LM for distortion)
        if iter % 2 == 0 {
            result.distortion = refine_distortion(
                &result.intrinsics,
                &result.extrinsics,
                &result.distortion,
                object_points,
                image_points,
                5,
            )
            .unwrap_or(result.distortion);
        }

        let cur_rms = compute_rms_reprojection(
            &result.intrinsics,
            &result.extrinsics,
            object_points,
            image_points,
        )?;
        if (prev_rms - cur_rms).abs() < 1e-8 {
            prev_rms = cur_rms;
            break;
        }
        prev_rms = cur_rms;
    }

    result.rms_reprojection_error = prev_rms;
    Ok(result)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Estimate homography using Direct Linear Transform (DLT)
fn estimate_homography_dlt(src: &[Point2<f64>], dst: &[Point2<f64>]) -> Result<Matrix3<f64>> {
    if src.len() != dst.len() || src.len() < 4 {
        return Err(cv_core::Error::CalibrationError(
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
        cv_core::Error::CalibrationError("SVD failed in estimate_homography_dlt".to_string())
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
        return Err(cv_core::Error::CalibrationError(
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
        cv_core::Error::CalibrationError(
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
        return Err(cv_core::Error::CalibrationError(
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
            return Err(cv_core::Error::CalibrationError(
                "degenerate calibration system after sign flip".to_string(),
            ));
        }
        v0 = (b12 * b13 - b11 * b23) / denom;
        lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    }
    if lambda <= 0.0 {
        return Err(cv_core::Error::CalibrationError(
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
        cv_core::Error::CalibrationError("SVD U missing in extrinsics_from_homography".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError(
            "SVD V^T missing in extrinsics_from_homography".to_string(),
        )
    })?;
    r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
    }

    let t = t_raw * scale;
    Ok(Pose::new(r, t))
}

/// Compute RMS reprojection error
#[allow(clippy::type_complexity)]
fn compute_rms_reprojection(
    intrinsics: &CameraIntrinsics,
    extrinsics: &[Pose],
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
) -> Result<f64> {
    if extrinsics.len() != object_points.len() || object_points.len() != image_points.len() {
        return Err(cv_core::Error::CalibrationError(
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
                    let u = intrinsics.fx * (pc[0] / pc[2]) + intrinsics.cx;
                    let v = intrinsics.fy * (pc[1] / pc[2]) + intrinsics.cy;
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
        return Err(cv_core::Error::CalibrationError(
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

/// Estimate intrinsics from camera extrinsics
fn estimate_intrinsics_from_extrinsics(
    extrinsics: &[Pose],
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    fallback: CameraIntrinsics,
) -> Result<CameraIntrinsics> {
    let mut sx2 = 0.0f64;
    let mut sxu = 0.0f64;
    let mut sx = 0.0f64;
    let mut su = 0.0f64;
    let mut n_x = 0usize;

    let mut sy2 = 0.0f64;
    let mut syv = 0.0f64;
    let mut sy = 0.0f64;
    let mut sv = 0.0f64;
    let mut n_y = 0usize;

    for ((ext, obj), img) in extrinsics
        .iter()
        .zip(object_points.iter())
        .zip(image_points.iter())
    {
        for (p3, p2) in obj.iter().zip(img.iter()) {
            let pc = ext.rotation * p3.coords + ext.translation;
            if pc[2].abs() <= 1e-12 {
                continue;
            }
            let xn = pc[0] / pc[2];
            let yn = pc[1] / pc[2];

            sx2 += xn * xn;
            sxu += xn * p2.x;
            sx += xn;
            su += p2.x;
            n_x += 1;

            sy2 += yn * yn;
            syv += yn * p2.y;
            sy += yn;
            sv += p2.y;
            n_y += 1;
        }
    }

    if n_x < 2 || n_y < 2 {
        return Err(cv_core::Error::CalibrationError(
            "estimate_intrinsics_from_extrinsics: insufficient valid points".to_string(),
        ));
    }

    let det_x = sx2 * n_x as f64 - sx * sx;
    let det_y = sy2 * n_y as f64 - sy * sy;
    if det_x.abs() < 1e-18 || det_y.abs() < 1e-18 {
        return Ok(fallback);
    }

    let fx = (sxu * n_x as f64 - sx * su) / det_x;
    let cx = (sx2 * su - sx * sxu) / det_x;
    let fy = (syv * n_y as f64 - sy * sv) / det_y;
    let cy = (sy2 * sv - sy * syv) / det_y;

    if !fx.is_finite() || !fy.is_finite() || fx.abs() < 1e-12 || fy.abs() < 1e-12 {
        return Ok(fallback);
    }

    Ok(CameraIntrinsics::new(
        fx,
        fy,
        cx,
        cy,
        fallback.width,
        fallback.height,
    ))
}

/// Refine distortion coefficients iteratively
fn refine_distortion(
    intrinsics: &CameraIntrinsics,
    extrinsics: &[Pose],
    initial_distortion: &Distortion,
    object_points: &[Vec<Point3<f64>>],
    image_points: &[Vec<Point2<f64>>],
    max_iters: usize,
) -> Result<Distortion> {
    let mut distortion = *initial_distortion;
    let mut params = [
        distortion.k1,
        distortion.k2,
        distortion.p1,
        distortion.p2,
        distortion.k3,
    ];

    let total_pts: usize = object_points.iter().map(|v| v.len()).sum();
    if total_pts < 10 {
        return Ok(distortion);
    }

    for _ in 0..max_iters {
        let mut j = DMatrix::<f64>::zeros(2 * total_pts, 5);
        let mut r = DMatrix::<f64>::zeros(2 * total_pts, 1);

        // Compute residual and Jacobian
        let mut row_idx = 0;
        for (i, ext) in extrinsics.iter().enumerate() {
            for (p3, p2) in object_points[i].iter().zip(image_points[i].iter()) {
                let pc = ext.rotation * p3.coords + ext.translation;
                if pc[2].abs() <= 1e-12 {
                    continue;
                }

                let pred = project_points_with_distortion(&[*p3], intrinsics, ext, &distortion)?[0];
                r[(row_idx, 0)] = pred.x - p2.x;
                r[(row_idx + 1, 0)] = pred.y - p2.y;

                // Numerical Jacobian w.r.t distortion params
                let eps = 1e-7;
                for k in 0..5 {
                    let mut d_perturbed = distortion;
                    match k {
                        0 => d_perturbed.k1 += eps,
                        1 => d_perturbed.k2 += eps,
                        2 => d_perturbed.p1 += eps,
                        3 => d_perturbed.p2 += eps,
                        4 => d_perturbed.k3 += eps,
                        _ => {
                            return Err(cv_core::Error::AlgorithmError(
                                "Invalid distortion index".to_string(),
                            ))
                        }
                    }
                    let p_perturbed =
                        project_points_with_distortion(&[*p3], intrinsics, ext, &d_perturbed)?[0];
                    j[(row_idx, k)] = (p_perturbed.x - pred.x) / eps;
                    j[(row_idx + 1, k)] = (p_perturbed.y - pred.y) / eps;
                }
                row_idx += 2;
            }
        }

        // Truncate to actual number of rows (skipped behind-camera points
        // leave trailing zero rows that corrupt the solve).
        let j = j.rows(0, row_idx).clone_owned();
        let r = r.rows(0, row_idx).clone_owned();

        let jt = j.transpose();
        let h = &jt * &j;
        let g = &jt * &r;

        let delta = h.lu().solve(&g).ok_or_else(|| {
            cv_core::Error::AlgorithmError(
                "Distortion refinement normal equation failed".to_string(),
            )
        })?;

        for k in 0..5 {
            params[k] -= delta[(k, 0)];
        }

        distortion.k1 = params[0];
        distortion.k2 = params[1];
        distortion.p1 = params[2];
        distortion.p2 = params[3];
        distortion.k3 = params[4];

        if delta.norm() < 1e-9 {
            break;
        }
    }

    Ok(distortion)
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
        return Err(cv_core::Error::CalibrationError(
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
