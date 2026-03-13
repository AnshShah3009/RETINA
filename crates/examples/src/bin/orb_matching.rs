//! Demo for visualizing ORB feature matches
//!
//! This example detects ORB features in two images, matches them,
//! and draws the matches side-by-side.

use cv_core::{storage::CpuStorage, Tensor, TensorShape};
use cv_features::{detect_and_compute_ctx, MatchType, Matcher, Orb};
use cv_runtime::orchestrator::scheduler;
use std::env;

#[path = "../visualization_utils.rs"]
mod visualization_utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <image1> <image2> [output]", args[0]);
        return Ok(());
    }

    let img1_path = &args[1];
    let img2_path = &args[2];
    let out_path = if args.len() > 3 {
        &args[3]
    } else {
        "matches.png"
    };

    println!("Loading images...");
    let img1 = image::open(img1_path)?.to_luma8();
    let img2 = image::open(img2_path)?.to_luma8();

    println!("Initializing Runtime...");
    let s = scheduler().unwrap();
    let group = s.get_default_group().unwrap();
    let device = group.device().unwrap();

    println!("Detecting ORB features...");
    let orb = Orb::default();

    // Detect in Image 1
    let t1: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
        img1.to_vec(),
        TensorShape::new(1, img1.height() as usize, img1.width() as usize),
    )?;
    let (kps1, desc1) = detect_and_compute_ctx(&orb, &device, &group, &t1);
    println!("Image 1: {} keypoints", kps1.len());

    // Detect in Image 2
    let t2: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
        img2.to_vec(),
        TensorShape::new(1, img2.height() as usize, img2.width() as usize),
    )?;
    let (kps2, desc2) = detect_and_compute_ctx(&orb, &device, &group, &t2);
    println!("Image 2: {} keypoints", kps2.len());

    println!("Matching features...");
    let matcher = Matcher::new(MatchType::BruteForceHamming);

    // Pass Descriptors directly
    let mut matches = matcher.match_descriptors(&desc1, &desc2);
    println!("Found {} matches", matches.len());

    // Filter matches (simple distance check for demo)
    matches.filter_by_distance(50.0);
    println!("Kept {} good matches", matches.len());

    // Create a dummy mask for visualization (all valid for now)
    // Real app would use RANSAC here.
    matches.mask = Some(vec![1; matches.len()]);

    println!("Drawing matches...");
    let result = visualization_utils::draw_matches(&img1, &kps1, &img2, &kps2, &matches);

    println!("Saving to {}", out_path);
    result.save(out_path)?;

    Ok(())
}
