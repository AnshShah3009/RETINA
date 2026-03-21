use cv_features::{fast::fast_detect, orb::orb_detect_and_compute};
use image::{GrayImage, Luma};
use std::time::Instant;

#[test]
fn profile_features_algorithms() {
    println!("--- Profiling cv-features ---");
    let width = 1920;
    let height = 1080;

    // Create mock 1080p data
    let gray_image = GrayImage::from_fn(width, height, |_, _| Luma([128u8]));

    // 1. FAST feature detection
    let start = Instant::now();
    let keypoints_fast = fast_detect(&gray_image, 20, 1000);
    let elapsed = start.elapsed();
    println!("FAST Detect (1080p, threshold=20): {:?}", elapsed);
    println!("  -> Found {} keypoints", keypoints_fast.len());

    // 2. ORB detection and compute
    let start = Instant::now();
    let (keypoints_orb, _) = orb_detect_and_compute(&gray_image, 500);
    let elapsed = start.elapsed();
    println!("ORB Detect & Compute (1080p, max 500 pts): {:?}", elapsed);
    println!("  -> Found {} keypoints", keypoints_orb.len());
}
