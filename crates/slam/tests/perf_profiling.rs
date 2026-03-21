use cv_core::geometry::CameraIntrinsics;
use cv_runtime::orchestrator::{scheduler, WorkloadHint};
use cv_slam::Slam;
use image::{GrayImage, Luma};
use std::time::Instant;

#[test]
fn profile_slam_algorithms() {
    println!("--- Profiling cv-slam ---");
    let width = 640;
    let height = 480;

    // Create mock VGA data
    let image = GrayImage::from_fn(width, height, |_, _| Luma([128u8]));

    let s = scheduler().unwrap();
    let group = s
        .get_best_group(cv_hal::BackendType::Cpu, WorkloadHint::Latency)
        .unwrap()
        .unwrap();

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, width as u32, height as u32);

    let mut slam = Slam::new(group, intrinsics);

    let start = Instant::now();

    // Process 10 frames to see tracking/feature extraction cost
    for _ in 0..10 {
        let _ = slam.process_image(&image);
    }

    let elapsed = start.elapsed();
    println!(
        "SLAM Pipeline (10 VGA frames, tracking only): {:?}",
        elapsed
    );
}
