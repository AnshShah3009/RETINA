//! Example: Feature Detection and Matching
//!
//! Run with: cargo run --example features_demo

fn main() {
    println!("=== cv-features Example ===\n");

    println!("1. Feature Detection:");
    println!("   - ORB: cv_features::orb_detect_and_compute");
    println!("   - AKAZE: cv_features::akaze_detect_and_compute");
    println!("   - SIFT: cv_features::sift_detect_and_compute");

    println!("\n2. Feature Matching:");
    println!("   - Brute Force: cv_features::Matcher::new(MatchType::BruteForce)");
    println!("   - FLANN: cv_features::Matcher::new(MatchType::Flann)");

    println!("\nNote: Full example requires valid test images");
    println!("      See Python bindings for complete usage examples.");

    println!("\n=== Example Complete ===");
}
