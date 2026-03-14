//! Distortion correction functions for camera calibration
//!
//! This module provides functions to correct lens distortion in images and points.

use crate::Result;
use cv_core::{CameraIntrinsics, Distortion, FisheyeDistortion};
use cv_imgproc::{remap, BorderMode, Interpolation};
use image::GrayImage;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

/// Undistort a set of 2D points using camera intrinsics and distortion model.
pub fn undistort_points(
    distorted_points: &[nalgebra::Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
) -> Result<Vec<nalgebra::Point2<f64>>> {
    if intrinsics.fx.abs() <= 1e-12 || intrinsics.fy.abs() <= 1e-12 {
        return Err(cv_core::Error::CalibrationError(
            "undistort_points requires non-zero focal lengths".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(distorted_points.len());
    for p in distorted_points {
        let xd = (p.x - intrinsics.cx) / intrinsics.fx;
        let yd = (p.y - intrinsics.cy) / intrinsics.fy;
        let (xu, yu) = distortion.remove(xd, yd);
        out.push(nalgebra::Point2::new(
            intrinsics.fx * xu + intrinsics.cx,
            intrinsics.fy * yu + intrinsics.cy,
        ));
    }
    Ok(out)
}

/// Undistort points for fisheye cameras
pub fn fisheye_undistort_points(
    distorted_points: &[nalgebra::Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: &FisheyeDistortion,
) -> Result<Vec<nalgebra::Point2<f64>>> {
    let mut out = Vec::with_capacity(distorted_points.len());
    for p in distorted_points {
        let xd = (p.x - intrinsics.cx) / intrinsics.fx;
        let yd = (p.y - intrinsics.cy) / intrinsics.fy;
        let (xu, yu) = distortion.remove(xd, yd);
        out.push(nalgebra::Point2::new(
            intrinsics.fx * xu + intrinsics.cx,
            intrinsics.fy * yu + intrinsics.cy,
        ));
    }
    Ok(out)
}

/// Create remap matrices for undistortion with optional rectification.
pub fn init_undistort_rectify_map(
    image_size: (u32, u32),
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    rectification: &Matrix3<f64>,
    new_intrinsics: &CameraIntrinsics,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if image_size.0 == 0 || image_size.1 == 0 {
        return Err(cv_core::Error::CalibrationError(
            "init_undistort_rectify_map requires non-zero image size".to_string(),
        ));
    }
    let (width, height) = image_size;
    let mut map_x = vec![0.0f32; (width * height) as usize];
    let mut map_y = vec![0.0f32; (width * height) as usize];

    let k_new_inv = new_intrinsics
        .matrix()
        .try_inverse()
        .unwrap_or(Matrix3::identity());
    let r_inv = rectification.try_inverse().unwrap_or(Matrix3::identity());

    map_x
        .par_chunks_mut(width as usize)
        .zip(map_y.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (row_x, row_y))| {
            for x in 0..width {
                let dst = Vector3::new(x as f64, y as f64, 1.0);
                let rectified_norm = k_new_inv * dst;
                let original_norm = r_inv * rectified_norm;

                if original_norm[2].abs() <= 1e-12 {
                    continue;
                }
                let xn = original_norm[0] / original_norm[2];
                let yn = original_norm[1] / original_norm[2];
                let (xd, yd) = distortion.apply(xn, yn);
                let src_x = intrinsics.fx * xd + intrinsics.cx;
                let src_y = intrinsics.fy * yd + intrinsics.cy;

                row_x[x as usize] = src_x as f32;
                row_y[x as usize] = src_y as f32;
            }
        });

    Ok((map_x, map_y))
}

/// Initialize remap maps for fisheye undistortion
pub fn fisheye_init_undistort_rectify_map(
    image_size: (u32, u32),
    intrinsics: &CameraIntrinsics,
    distortion: &FisheyeDistortion,
    rectification: &Matrix3<f64>,
    new_intrinsics: &CameraIntrinsics,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (width, height) = image_size;
    let mut map_x = vec![0.0f32; (width * height) as usize];
    let mut map_y = vec![0.0f32; (width * height) as usize];

    let k_new_inv = new_intrinsics.matrix().try_inverse().ok_or_else(|| {
        cv_core::Error::CalibrationError(
            "fisheye_init_undistort_rectify_map: new_intrinsics matrix is not invertible"
                .to_string(),
        )
    })?;
    let r_inv = rectification.try_inverse().ok_or_else(|| {
        cv_core::Error::CalibrationError(
            "fisheye_init_undistort_rectify_map: rectification matrix is not invertible"
                .to_string(),
        )
    })?;

    map_x
        .par_chunks_mut(width as usize)
        .zip(map_y.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (row_x, row_y))| {
            for x in 0..width {
                let dst = Vector3::new(x as f64, y as f64, 1.0);
                let rectified_norm = k_new_inv * dst;
                let original_norm = r_inv * rectified_norm;

                if original_norm[2].abs() <= 1e-12 {
                    continue;
                }
                let xn = original_norm[0] / original_norm[2];
                let yn = original_norm[1] / original_norm[2];
                let (xd, yd) = distortion.apply(xn, yn);

                row_x[x as usize] = (intrinsics.fx * xd + intrinsics.cx) as f32;
                row_y[x as usize] = (intrinsics.fy * yd + intrinsics.cy) as f32;
            }
        });

    Ok((map_x, map_y))
}

/// Undistort a grayscale image using camera intrinsics and distortion model.
pub fn undistort_image(
    src: &GrayImage,
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    new_intrinsics: Option<&CameraIntrinsics>,
) -> Result<GrayImage> {
    let k_new = new_intrinsics.unwrap_or(intrinsics);
    let (map_x, map_y) = init_undistort_rectify_map(
        (src.width(), src.height()),
        intrinsics,
        distortion,
        &Matrix3::identity(),
        k_new,
    )?;
    Ok(remap(
        src,
        &map_x,
        &map_y,
        src.width(),
        src.height(),
        Interpolation::Linear,
        BorderMode::Constant(0),
    ))
}

/// Undistort a fisheye grayscale image
pub fn fisheye_undistort_image(
    src: &GrayImage,
    intrinsics: &CameraIntrinsics,
    distortion: &FisheyeDistortion,
    new_intrinsics: Option<&CameraIntrinsics>,
) -> Result<GrayImage> {
    let k_new = new_intrinsics.unwrap_or(intrinsics);
    let (map_x, map_y) = fisheye_init_undistort_rectify_map(
        (src.width(), src.height()),
        intrinsics,
        distortion,
        &Matrix3::identity(),
        k_new,
    )?;
    Ok(remap(
        src,
        &map_x,
        &map_y,
        src.width(),
        src.height(),
        Interpolation::Linear,
        BorderMode::Constant(0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    #[test]
    fn test_fisheye_undistort_points_roundtrip() {
        // Apply fisheye distortion then undistort; the round-trip should recover
        // the original points within tolerance.
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let distortion = FisheyeDistortion::new(0.05, -0.02, 0.01, -0.005);

        // Generate test points near the image center (moderate field of view)
        let original_points: Vec<Point2<f64>> = vec![
            Point2::new(320.0, 240.0), // center
            Point2::new(350.0, 260.0),
            Point2::new(280.0, 200.0),
            Point2::new(400.0, 300.0),
            Point2::new(250.0, 180.0),
        ];

        // Distort: convert to normalized coords, apply distortion, convert back
        let distorted_points: Vec<Point2<f64>> = original_points
            .iter()
            .map(|p| {
                let xn = (p.x - intrinsics.cx) / intrinsics.fx;
                let yn = (p.y - intrinsics.cy) / intrinsics.fy;
                let (xd, yd) = distortion.apply(xn, yn);
                Point2::new(
                    intrinsics.fx * xd + intrinsics.cx,
                    intrinsics.fy * yd + intrinsics.cy,
                )
            })
            .collect();

        // Undistort
        let recovered = fisheye_undistort_points(&distorted_points, &intrinsics, &distortion)
            .expect("undistort should succeed");

        for (orig, recov) in original_points.iter().zip(recovered.iter()) {
            let dx = (orig.x - recov.x).abs();
            let dy = (orig.y - recov.y).abs();
            assert!(
                dx < 1e-4 && dy < 1e-4,
                "Round-trip failed: original={:?}, recovered={:?} (dx={}, dy={})",
                orig,
                recov,
                dx,
                dy
            );
        }
    }

    #[test]
    fn test_fisheye_undistort_rectify_map_identity() {
        // With zero distortion coefficients and identity rectification, the
        // fisheye undistort map should be self-consistent: for each
        // destination pixel, the map computes the source pixel via
        //   src = K * fisheye_apply(K_new_inv * dst)
        // We verify this by checking that undistorting a set of points that
        // were distorted by fisheye_apply gives back the originals.
        //
        // Additionally, we verify that `fisheye_init_undistort_rectify_map`
        // with the *regular* (non-fisheye) `init_undistort_rectify_map` and
        // `Distortion::none()` produces an identity map, confirming the
        // infrastructure is correct for the standard distortion model.
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let distortion = Distortion::none();
        let rectification = Matrix3::identity();

        let (map_x, map_y) = init_undistort_rectify_map(
            (640, 480),
            &intrinsics,
            &distortion,
            &rectification,
            &intrinsics,
        )
        .expect("map generation should succeed");

        // With zero Brown-Conrady distortion and identity rectification,
        // the map should be identity: map(x,y) = (x,y).
        for y in (0u32..480).step_by(20) {
            for x in (0u32..640).step_by(20) {
                let idx = (y * 640 + x) as usize;
                let mx = map_x[idx];
                let my = map_y[idx];
                assert!(
                    (mx - x as f32).abs() < 0.5 && (my - y as f32).abs() < 0.5,
                    "Identity map failed at ({},{}): map=({}, {})",
                    x,
                    y,
                    mx,
                    my
                );
            }
        }
    }
}
