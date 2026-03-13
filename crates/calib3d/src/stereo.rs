use crate::calibration::*;
use crate::essential_fundamental::*;
use crate::pattern::find_chessboard_corners;
use cv_core::{CameraIntrinsics, Pose};
use image::GrayImage;
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3};
use std::path::Path;

use crate::Result;

#[derive(Debug, Clone)]
pub struct StereoRectifyMatrices {
    pub r1: Matrix3<f64>,
    pub r2: Matrix3<f64>,
    pub p1: Matrix3x4<f64>,
    pub p2: Matrix3x4<f64>,
    pub q: Matrix4<f64>,
}

#[derive(Debug, Clone)]
pub struct StereoCalibrationResult {
    pub left: CameraCalibrationResult,
    pub right: CameraCalibrationResult,
    pub relative_extrinsics: Pose,
    pub essential_matrix: Matrix3<f64>,
    pub fundamental_matrix: Matrix3<f64>,
}

#[derive(Debug, Clone)]
pub struct StereoCalibrationFileReport {
    pub total_pairs: usize,
    pub used_pairs: usize,
    pub rejected_pairs: Vec<usize>,
}

pub fn stereo_calibrate_planar(
    object_points: &[Vec<Point3<f64>>],
    left_image_points: &[Vec<Point2<f64>>],
    right_image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
) -> Result<StereoCalibrationResult> {
    stereo_calibrate_planar_with_options(
        object_points,
        left_image_points,
        right_image_points,
        image_size,
        CameraCalibrationOptions::default(),
    )
}

pub fn stereo_calibrate_planar_with_options(
    object_points: &[Vec<Point3<f64>>],
    left_image_points: &[Vec<Point2<f64>>],
    right_image_points: &[Vec<Point2<f64>>],
    image_size: (u32, u32),
    options: CameraCalibrationOptions,
) -> Result<StereoCalibrationResult> {
    if object_points.len() != left_image_points.len()
        || object_points.len() != right_image_points.len()
    {
        return Err(cv_core::Error::CalibrationError(
            "stereo_calibrate_planar expects matching batch sizes".to_string(),
        ));
    }
    if object_points.len() < 3 {
        return Err(cv_core::Error::CalibrationError(
            "stereo_calibrate_planar needs at least 3 views".to_string(),
        ));
    }

    let left = calibrate_camera_planar_with_options(
        object_points,
        left_image_points,
        image_size,
        options,
    )?;
    let right = calibrate_camera_planar_with_options(
        object_points,
        right_image_points,
        image_size,
        options,
    )?;

    let n = left.extrinsics.len().min(right.extrinsics.len());
    if n == 0 {
        return Err(cv_core::Error::CalibrationError(
            "stereo_calibrate_planar: no usable extrinsics".to_string(),
        ));
    }

    let mut t_sum = Vector3::zeros();
    let mut r_sum = Matrix3::<f64>::zeros();
    for i in 0..n {
        let r_l = left.extrinsics[i].rotation;
        let t_l = left.extrinsics[i].translation;
        let r_r = right.extrinsics[i].rotation;
        let t_r = right.extrinsics[i].translation;

        // Convert quaternion rotation to matrix for accumulation
        let r_rel = (r_r * r_l.inverse()).to_rotation_matrix().into_inner();
        let t_rel = t_r - r_rel * t_l;
        r_sum += r_rel;
        t_sum += t_rel;
    }
    t_sum /= n as f64;

    let svd = r_sum.svd(true, true);
    let u = svd.u.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD U missing in stereo_calibrate_planar".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD V^T missing in stereo_calibrate_planar".to_string())
    })?;
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
    }

    let relative_extrinsics = Pose::new(r, t_sum);
    let essential_matrix = essential_from_extrinsics(&relative_extrinsics);
    let fundamental_matrix =
        fundamental_from_essential(&essential_matrix, &left.intrinsics, &right.intrinsics);

    Ok(StereoCalibrationResult {
        left,
        right,
        relative_extrinsics,
        essential_matrix,
        fundamental_matrix,
    })
}

pub fn stereo_calibrate_from_chessboard_images(
    left_images: &[GrayImage],
    right_images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<StereoCalibrationResult> {
    stereo_calibrate_from_chessboard_images_with_options(
        left_images,
        right_images,
        pattern_size,
        square_size,
        CameraCalibrationOptions::default(),
    )
}

pub fn stereo_calibrate_from_chessboard_images_with_options(
    left_images: &[GrayImage],
    right_images: &[GrayImage],
    pattern_size: (usize, usize),
    square_size: f64,
    options: CameraCalibrationOptions,
) -> Result<StereoCalibrationResult> {
    if left_images.len() != right_images.len() || left_images.is_empty() {
        return Err(cv_core::Error::CalibrationError(
            "left/right image lists must be non-empty and equal-sized".to_string(),
        ));
    }

    let (w, h) = left_images[0].dimensions();
    if left_images.iter().any(|i| i.dimensions() != (w, h))
        || right_images.iter().any(|i| i.dimensions() != (w, h))
    {
        return Err(cv_core::Error::CalibrationError(
            "all stereo calibration images must share the same dimensions".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();

    for (l, r) in left_images.iter().zip(right_images.iter()) {
        let cl = find_chessboard_corners(l, pattern_size);
        let cr = find_chessboard_corners(r, pattern_size);
        if let (Ok(pl), Ok(pr)) = (cl, cr) {
            object_points.push(board.clone());
            left_points.push(pl);
            right_points.push(pr);
        }
    }

    if object_points.len() < 3 {
        return Err(cv_core::Error::CalibrationError(format!(
            "need at least 3 valid stereo chessboard pairs, found {}",
            object_points.len()
        )));
    }

    stereo_calibrate_planar_with_options(
        &object_points,
        &left_points,
        &right_points,
        (w, h),
        options,
    )
}

pub fn stereo_calibrate_from_chessboard_files<P: AsRef<Path>>(
    left_paths: &[P],
    right_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
) -> Result<(StereoCalibrationResult, StereoCalibrationFileReport)> {
    stereo_calibrate_from_chessboard_files_with_options(
        left_paths,
        right_paths,
        pattern_size,
        square_size,
        CameraCalibrationOptions::default(),
    )
}

pub fn stereo_calibrate_from_chessboard_files_with_options<P: AsRef<Path>>(
    left_paths: &[P],
    right_paths: &[P],
    pattern_size: (usize, usize),
    square_size: f64,
    options: CameraCalibrationOptions,
) -> Result<(StereoCalibrationResult, StereoCalibrationFileReport)> {
    if left_paths.len() != right_paths.len() || left_paths.is_empty() {
        return Err(cv_core::Error::CalibrationError(
            "left/right file lists must be non-empty and equal-sized".to_string(),
        ));
    }

    let board = generate_chessboard_object_points(pattern_size, square_size);
    let mut object_points = Vec::new();
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();
    let mut rejected = Vec::new();
    let mut expected_dims = None;

    for i in 0..left_paths.len() {
        let left = image::open(&left_paths[i]).map(|v| v.to_luma8());
        let right = image::open(&right_paths[i]).map(|v| v.to_luma8());
        let (left, right) = match (left, right) {
            (Ok(l), Ok(r)) => (l, r),
            _ => {
                rejected.push(i);
                continue;
            }
        };

        if let Some((w, h)) = expected_dims {
            if left.dimensions() != (w, h) || right.dimensions() != (w, h) {
                rejected.push(i);
                continue;
            }
        } else {
            expected_dims = Some(left.dimensions());
        }

        let cl = find_chessboard_corners(&left, pattern_size);
        let cr = find_chessboard_corners(&right, pattern_size);
        if let (Ok(pl), Ok(pr)) = (cl, cr) {
            object_points.push(board.clone());
            left_points.push(pl);
            right_points.push(pr);
        } else {
            rejected.push(i);
        }
    }

    if object_points.len() < 3 {
        return Err(cv_core::Error::CalibrationError(format!(
            "need at least 3 valid stereo pairs, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        cv_core::Error::CalibrationError(
            "no readable stereo pairs in provided file lists".to_string(),
        )
    })?;

    let calib = stereo_calibrate_planar_with_options(
        &object_points,
        &left_points,
        &right_points,
        dims,
        options,
    )
    .map_err(|e| {
        cv_core::Error::CalibrationError(format!(
            "stereo calibration failed for file subset (used {} / {} pairs): {}",
            object_points.len(),
            left_paths.len(),
            e
        ))
    })?;
    let report = StereoCalibrationFileReport {
        total_pairs: left_paths.len(),
        used_pairs: object_points.len(),
        rejected_pairs: rejected,
    };
    Ok((calib, report))
}

pub fn stereo_rectify_matrices(
    left_intrinsics: &CameraIntrinsics,
    right_intrinsics: &CameraIntrinsics,
    left_extrinsics: &Pose,
    right_extrinsics: &Pose,
) -> Result<StereoRectifyMatrices> {
    let left_rot_mat = left_extrinsics.rotation_matrix();
    let rel_r = left_rot_mat.transpose() * right_extrinsics.rotation_matrix();
    let rel_t =
        left_rot_mat.transpose() * (right_extrinsics.translation - left_extrinsics.translation);
    let baseline = rel_t.norm();
    if baseline <= 1e-12 {
        return Err(cv_core::Error::CalibrationError(
            "stereo_rectify_matrices requires non-zero baseline".to_string(),
        ));
    }

    let ex = rel_t / baseline;
    let helper = if ex[2].abs() < 0.9 {
        Vector3::<f64>::new(0.0, 0.0, 1.0)
    } else {
        Vector3::<f64>::new(0.0, 1.0, 0.0)
    };
    let ey = helper.cross(&ex).normalize();
    let ez = ex.cross(&ey).normalize();
    let basis = Matrix3::from_columns(&[ex, ey, ez]);
    let r_rect = basis.transpose();

    let r1 = r_rect;
    let r2 = r_rect * rel_r;

    let fx = 0.5 * (left_intrinsics.fx + right_intrinsics.fx);
    let fy = 0.5 * (left_intrinsics.fy + right_intrinsics.fy);
    let cx1 = 0.5 * (left_intrinsics.cx + right_intrinsics.cx);
    let cx2 = cx1;
    let cy = 0.5 * (left_intrinsics.cy + right_intrinsics.cy);
    let tx = -fx * baseline;

    let p1 = Matrix3x4::new(
        fx, 0.0, cx1, 0.0, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );
    let p2 = Matrix3x4::new(
        fx, 0.0, cx2, tx, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    let mut q = Matrix4::<f64>::zeros();
    q[(0, 0)] = 1.0;
    q[(0, 3)] = -cx1;
    q[(1, 1)] = 1.0;
    q[(1, 3)] = -cy;
    q[(2, 3)] = fx;
    q[(3, 2)] = -1.0 / tx;
    q[(3, 3)] = (cx1 - cx2) / tx;

    Ok(StereoRectifyMatrices { r1, r2, p1, p2, q })
}
