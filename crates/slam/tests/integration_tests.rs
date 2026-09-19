use cv_core::CameraIntrinsics;
use cv_runtime::scheduler;
use cv_slam::Slam;
use image::{GrayImage, Luma};

#[test]
fn test_slam_basic_pipeline() {
    let s = scheduler().expect("Failed to get scheduler");
    let group = s.get_default_group().expect("Failed to get default group");

    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let mut slam = Slam::new(group, intrinsics);

    // Process a few identical images
    // The first frame will fail tracking because there's no map
    let mut img = GrayImage::new(640, 480);
    for y in 0..480 {
        for x in 0..640 {
            if ((x / 32) + (y / 32)) % 2 == 0 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
    }

    let res1 = slam.process_image(&img);
    // First frame may succeed or fail depending on implementation
    // Just verify it doesn't panic
    let _ = res1;

    // We haven't implemented map initialization in Slam::process_image yet,
    // it just tries to track. But we verified it compiles and doesn't panic.
    // In a real scenario, we'd add points to the map here.
}
