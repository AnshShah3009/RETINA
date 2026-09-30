#![forbid(unsafe_code)]
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

/// A single monocular frame must not produce a pose.
///
/// The tracker used to fabricate a map by back-projecting every keypoint onto a
/// hard-coded plane at z = 1.0, report success, and return the identity. Every
/// later frame then matched the real scene against that plane and ran PnP on it
/// - the most degenerate configuration PnP has, a fronto-parallel planar target -
/// and returned a confident, meaningless pose with no error surfaced.
///
/// Monocular SLAM genuinely cannot initialise from one frame: there is no
/// parallax, so nothing can be triangulated. Failing is the correct answer, and
/// a caller with depth has `process_frame_with_depth`.
#[test]
fn single_frame_tracking_reports_failure() {
    use cv_core::{CpuTensor, Tensor, TensorShape};
    use cv_slam::tracking::Tracker;

    let group = std::sync::Arc::new(
        cv_runtime::orchestrator::ResourceGroup::new(
            "slam-test",
            cv_hal::backend::DeviceId(0),
            1,
            None,
            cv_runtime::orchestrator::GroupPolicy::default(),
        )
        .expect("resource group"),
    );
    let intrinsics = cv_core::CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let mut tracker = Tracker::new(group, intrinsics);

    // A textured frame so detection has something to find.
    let (w, h) = (64usize, 48usize);
    let data: Vec<u8> = (0..h * w)
        .map(|i| (((i * 7 + i / w * 13) % 256) as u8))
        .collect();
    let image: CpuTensor<u8> = Tensor::from_vec(data, TensorShape::new(1, h, w)).unwrap();
    let mut map = cv_slam::types::WorldMap::new();

    let result = tracker.process_frame(&image, &mut map);
    assert!(
        result.is_err(),
        "one monocular frame must not yield a pose; got {:?} and a map with {} points",
        result.as_ref().map(|(p, _)| p.clone()).ok(),
        map.points.len()
    );
    assert!(
        map.points.is_empty(),
        "no map points can be created from a single frame, but {} were",
        map.points.len()
    );
}
