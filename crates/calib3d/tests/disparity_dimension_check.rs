#![forbid(unsafe_code)]
//! Regression tests for defect #1: `disparity_to_pointcloud` indexed
//! `left_image` with coordinates derived from the *disparity map* and never
//! checked that the image actually has those dimensions, so a mismatch between
//! the two (an ordinary caller mistake: a matcher and an image from different
//! captures) panicked inside `image::GrayImage::get_pixel`.

use cv_calib3d::stereo_matching::{disparity_to_pointcloud, DisparityMap, StereoParams};
use image::{GrayImage, Luma};

/// Unique temp path per test: the harness runs tests concurrently inside a
/// single process, so a shared name races. Process id + per-test counter.
fn unique_tmp(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!(
        "cv_calib3d_disp_dim_{}_{}_{}.svg",
        std::process::id(),
        n,
        tag
    ))
}

/// Mismatched sizes must produce a clear `Err`, not a panic.
#[test]
fn disparity_to_pointcloud_rejects_mismatched_image_size() {
    let params = StereoParams::new(500.0, 0.1, 16.0, 8.0);
    let mut disparity = DisparityMap::new(32, 16, 0, 64);
    disparity.set(31, 15, 16.0); // valid disparity -> would index (31, 15)

    // Image is far smaller than the disparity map.
    let left = GrayImage::new(8, 4);

    let err = disparity_to_pointcloud(&disparity, &left, &params)
        .expect_err("mismatched image size must be rejected, not panicked on");

    let msg = format!("{}", err);
    assert!(
        msg.contains("32x16") && msg.contains("8x4"),
        "error should name both the disparity and image dimensions, got: {}",
        msg
    );
}

/// CONTROL: a correctly sized image must still produce points and colours.
#[test]
fn disparity_to_pointcloud_accepts_matching_image_size() {
    let params = StereoParams::new(500.0, 0.1, 16.0, 8.0);
    let mut disparity = DisparityMap::new(32, 16, 0, 64);
    for y in 0..16 {
        for x in 0..32 {
            disparity.set(x, y, 16.0);
        }
    }

    let mut left = GrayImage::new(32, 16);
    for y in 0..16 {
        for x in 0..32 {
            left.put_pixel(x, y, Luma([(x * 8) as u8]));
        }
    }

    let cloud = disparity_to_pointcloud(&disparity, &left, &params)
        .expect("matching dimensions must still succeed");

    assert_eq!(cloud.len(), 32 * 16);
    assert!(cloud.colors.is_some(), "colors must be populated");
    // sanity: colour at (4, 4) is Luma 32 -> 32/255
    let idx = 4 * 32 + 4;
    let c = cloud.colors.as_ref().unwrap();
    assert!((c[idx].x - 32.0 / 255.0).abs() < 1e-6);

    let _ = unique_tmp("matching");
}
