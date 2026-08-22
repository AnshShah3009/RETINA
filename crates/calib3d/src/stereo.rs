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
        return Err(cv_core::Error::AlgorithmError(
            "stereo_calibrate_planar expects matching batch sizes".to_string(),
        ));
    }
    if object_points.len() < 3 {
        return Err(cv_core::Error::AlgorithmError(
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
        return Err(cv_core::Error::AlgorithmError(
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
        cv_core::Error::AlgorithmError("SVD U missing in stereo_calibrate_planar".to_string())
    })?;
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD V^T missing in stereo_calibrate_planar".to_string())
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
        return Err(cv_core::Error::AlgorithmError(
            "left/right image lists must be non-empty and equal-sized".to_string(),
        ));
    }

    let (w, h) = left_images[0].dimensions();
    if left_images.iter().any(|i| i.dimensions() != (w, h))
        || right_images.iter().any(|i| i.dimensions() != (w, h))
    {
        return Err(cv_core::Error::AlgorithmError(
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
        return Err(cv_core::Error::AlgorithmError(format!(
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
        return Err(cv_core::Error::AlgorithmError(
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
        return Err(cv_core::Error::AlgorithmError(format!(
            "need at least 3 valid stereo pairs, found {}",
            object_points.len()
        )));
    }
    let dims = expected_dims.ok_or_else(|| {
        cv_core::Error::AlgorithmError(
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
        cv_core::Error::AlgorithmError(format!(
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
    let r_left = left_extrinsics.rotation_matrix();
    let r_right = right_extrinsics.rotation_matrix();

    // Relative rotation mapping left-camera coordinates to right-camera
    // coordinates: X_right = M · X_left.
    let m = r_right * r_left.transpose();

    // Camera centers under the world→camera convention (X_c = R·X_w + t ⇒ C = −Rᵀ·t).
    let c_left = -r_left.transpose() * &left_extrinsics.translation;
    let c_right = -r_right.transpose() * &right_extrinsics.translation;

    // Baseline direction (left → right) expressed in each camera's frame.
    let b_left = r_left * (c_right - c_left);
    let baseline = b_left.norm();
    if baseline <= 1e-12 {
        return Err(cv_core::Error::AlgorithmError(
            "stereo_rectify_matrices requires non-zero baseline".to_string(),
        ));
    }

    // Rectifying rotation for the left camera: its ROWS form the new basis so
    // that R₁·b̂ = x̂ (epipole at infinity on the x-axis) while keeping the old
    // optical direction as the new +z (scene stays in front).
    let ex_l = b_left / baseline;
    let k = Vector3::<f64>::new(0.0, 0.0, 1.0);
    let mut ey_l = k.cross(&ex_l);
    if ey_l.norm() < 1e-6 {
        // Baseline nearly parallel to the optical axis; fall back to +y.
        ey_l = Vector3::<f64>::new(0.0, 1.0, 0.0).cross(&ex_l);
    }
    let ey_l = ey_l.normalize();
    let ez_l = ex_l.cross(&ey_l).normalize();
    let r1 = Matrix3::from_columns(&[ex_l, ey_l, ez_l]).transpose();

    // Right camera shares the same world-frame axes expressed through M:
    // O_right = R₂·R_right = R₁·Mᵀ·R_right = R₁·R_left = O_left ⇒ parallel axes.
    let r2 = r1 * m.transpose();

    let fx = 0.5 * (left_intrinsics.fx + right_intrinsics.fx);
    let fy = 0.5 * (left_intrinsics.fy + right_intrinsics.fy);
    let cx1 = 0.5 * (left_intrinsics.cx + right_intrinsics.cx);
    let cx2 = cx1;
    let cy = 0.5 * (left_intrinsics.cy + right_intrinsics.cy);

    let p1 = Matrix3x4::new(
        fx, 0.0, cx1, 0.0, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );
    let p2 = Matrix3x4::new(
        fx, 0.0, cx2, -fx * baseline, //
        0.0, fy, cy, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    // Disparity-to-depth: Z = fx·baseline / d. Following OpenCV,
    //   Tx = P2[0][3]/P2[0][0] (physical baseline, negative),
    //   Q(3,2) = −1/Tx,  Q(3,3) = (cx1−cx2)/Tx.
    let tx_physical = -fx * baseline / fx;
    let mut q = Matrix4::<f64>::zeros();
    q[(0, 0)] = 1.0;
    q[(0, 3)] = -cx1;
    q[(1, 1)] = 1.0;
    q[(1, 3)] = -cy;
    q[(2, 3)] = fx;
    q[(3, 2)] = -1.0 / tx_physical;
    q[(3, 3)] = (cx1 - cx2) / tx_physical;

    Ok(StereoRectifyMatrices { r1, r2, p1, p2, q })
}

/// Fisheye stereo rectification.
///
/// Computes rectification matrices for a stereo pair using fisheye camera models.
/// Uses the same Bouguet-style rotation splitting as `stereo_rectify_matrices` but
/// with fisheye-appropriate new camera matrices (reduced FOV to avoid excessive distortion).
///
/// # Arguments
/// * `left_intrinsics` / `right_intrinsics` — Pinhole intrinsics (fx, fy, cx, cy)
/// * `left_fisheye` / `right_fisheye` — Fisheye distortion coefficients (k1-k4)
/// * `left_extrinsics` / `right_extrinsics` — Camera poses
/// * `fov_scale` — Field-of-view scaling factor (0.0-1.0). Lower = less distortion, more cropping.
pub fn fisheye_stereo_rectify(
    left_intrinsics: &CameraIntrinsics,
    right_intrinsics: &CameraIntrinsics,
    _left_fisheye: &cv_core::FisheyeDistortion,
    _right_fisheye: &cv_core::FisheyeDistortion,
    left_extrinsics: &Pose,
    right_extrinsics: &Pose,
    fov_scale: f64,
) -> Result<StereoRectifyMatrices> {
    // Compute standard stereo rectification (rotation + projection)
    let result = stereo_rectify_matrices(
        left_intrinsics,
        right_intrinsics,
        left_extrinsics,
        right_extrinsics,
    )?;

    // Scale the focal length in the new projection matrices to reduce FOV
    // This avoids extreme warping at the edges of fisheye images
    let scale = fov_scale.clamp(0.1, 1.0);

    let mut p1 = result.p1;
    let mut p2 = result.p2;
    p1[(0, 0)] *= scale; // fx
    p1[(1, 1)] *= scale; // fy
    p2[(0, 0)] *= scale;
    p2[(1, 1)] *= scale;
    // Adjust tx proportionally
    p2[(0, 3)] *= scale;

    Ok(StereoRectifyMatrices {
        r1: result.r1,
        r2: result.r2,
        p1,
        p2,
        q: result.q,
    })
}

#[cfg(test)]
mod rectify_tests {
    use super::*;
    use nalgebra::{Rotation3, Unit};

    #[test]
    fn test_rectify_scanline_alignment_and_q_depth() {
        // Non-trivial poses: both cameras rotated and translated arbitrarily.
        let intr_l = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let intr_r = CameraIntrinsics::new(505.0, 502.0, 315.0, 242.0, 640, 480);

        let r_l = Rotation3::from_axis_angle(
            &Unit::new_normalize(Vector3::new(0.1, 0.05, 1.0)),
            0.15,
        )
        .into_inner();
        let r_r = Rotation3::from_axis_angle(
            &Unit::new_normalize(Vector3::new(-0.07, 0.12, 1.0)),
            -0.22,
        )
        .into_inner();
        let left = Pose::new(r_l, Vector3::new(0.02, -0.01, 0.05));
        let right = Pose::new(r_r, Vector3::new(-0.18, 0.03, 0.06));

        let m = stereo_rectify_matrices(&intr_l, &intr_r, &left, &right)
            .expect("rectification must succeed");

        // World-frame orientations of the rectified cameras must be identical
        // (parallel axes ⇒ epipolar lines horizontal on common scanlines).
        let o_left = m.r1 * left.rotation_matrix();
        let o_right = m.r2 * right.rotation_matrix();
        let axis_err = rotation_vector_from_matrix(&(o_left.transpose() * o_right)).norm();
        assert!(axis_err < 1e-9, "rectified axes differ by {}", axis_err);

        // Baseline must map to +x in both rectified frames.
        let c_l = -(left.rotation_matrix().transpose() * left.translation);
        let c_r = -(right.rotation_matrix().transpose() * right.translation);
        let b_w = c_r - c_l;
        let b_l = left.rotation_matrix() * b_w;
        let b_r = right.rotation_matrix() * b_w;
        assert!((m.r1 * b_l)[1].abs() < 1e-9 && (m.r1 * b_l)[2].abs() < 1e-9);
        assert!((m.r2 * b_r)[1].abs() < 1e-9 && (m.r2 * b_r)[2].abs() < 1e-9);
        assert!((m.r1 * b_l)[0] > 0.0);

        // Q must reproject a synthetic 3D point through its disparity exactly.
        let p_world = Point3::new(0.4, 0.25, 3.0);
        // Rectified frame coordinates. Both P1 and P2 take coordinates in
        // the COMMON rectified frame (anchored at the left camera); P2's tx
        // encodes the baseline.
        let pr_l = m.r1 * left.rotation_matrix() * (p_world - c_l);

        let pl = m.p1 * nalgebra::Vector4::new(pr_l.x, pr_l.y, pr_l.z, 1.0);
        let pr = m.p2 * nalgebra::Vector4::new(pr_l.x, pr_l.y, pr_l.z, 1.0);
        let ul = pl.x / pl.z;
        let ur = pr.x / pr.z;
        let vl = pl.y / pl.z;
        let vr = pr.y / pr.z;
        // Scanline alignment after rectification:
        assert!(
            (vl - vr).abs() < 1e-6,
            "scanline misalignment: {} vs {}",
            vl,
            vr
        );

        let disparity = ul - ur;
        assert!(disparity > 0.0, "left disparity must be positive");

        let hq = m.q * nalgebra::Vector4::new(ul, vl, disparity, 1.0);
        let rec = Point3::new(hq.x / hq.w, hq.y / hq.w, hq.z / hq.w);
        // Depth via Q must equal true Z (fx·B/d), not fx²·B/d.
        let baseline = b_w.norm();
        let expected_z = {
            let f = 0.5 * (intr_l.fx + intr_r.fx);
            f * baseline / disparity
        };
        assert!(
            (rec.z - expected_z).abs() / expected_z < 1e-6,
            "Q depth {} != expected {}",
            rec.z,
            expected_z
        );
    }

    fn rotation_vector_from_matrix(r: &nalgebra::Matrix3<f64>) -> nalgebra::Vector3<f64> {
        let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
        let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let denom = 2.0 * angle.sin();
        if denom.abs() < 1e-10 {
            return nalgebra::Vector3::zeros();
        }
        nalgebra::Vector3::new(
            (r[(2, 1)] - r[(1, 2)]) / denom * angle,
            (r[(0, 2)] - r[(2, 0)]) / denom * angle,
            (r[(1, 0)] - r[(0, 1)]) / denom * angle,
        )
    }
}

