use cv_calib3d::stereo_matching::block_matching::stereo_block_match;
use image::{GrayImage, Luma};
use std::time::Instant;

use cv_calib3d::distortion::undistort_image_ex;
use cv_core::{CameraIntrinsics, Distortion};
use nalgebra::Matrix3;

#[test]
fn profile_calib3d_algorithms() {
    println!("--- Profiling cv-calib3d ---");
    let width = 1280;
    let height = 720;

    // Create mock 720p stereo pair data
    let left_image = GrayImage::from_fn(width, height, |_, _| Luma([128u8]));
    let right_image = GrayImage::from_fn(width, height, |_, _| Luma([128u8]));

    let start = Instant::now();
    let disparity = stereo_block_match(&left_image, &right_image, 15, 64).unwrap();
    let elapsed = start.elapsed();

    println!("Stereo Block Match (720p, range 0-64): {:?}", elapsed);
    assert_eq!(disparity.width, width as u32);
    assert_eq!(disparity.height, height as u32);

    // --- Undistort Profiling ---
    let width_1080 = 1920;
    let height_1080 = 1080;
    let img_1080 = GrayImage::from_fn(width_1080, height_1080, |x, y| {
        Luma([((x + y) % 255) as u8])
    });

    let k = CameraIntrinsics::new(1000.0, 1000.0, 960.0, 540.0, width_1080, height_1080);
    let d = Distortion {
        k1: -0.2,
        k2: 0.1,
        p1: 0.001,
        p2: 0.001,
        k3: -0.05,
    };
    let rect = Matrix3::identity();

    let start = Instant::now();
    let _undistorted = undistort_image_ex(
        &img_1080,
        &k,
        &d,
        &rect,
        &k,
        cv_imgproc::Interpolation::Linear,
        cv_imgproc::BorderMode::Constant(0),
    )
    .unwrap();
    let elapsed = start.elapsed();
    println!("Undistort Image (1080p, GPU-accelerated): {:?}", elapsed);
}
