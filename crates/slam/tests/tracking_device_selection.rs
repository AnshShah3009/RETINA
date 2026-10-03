//! The tracker must choose its compute device from where the *data* is, not from
//! what the resource group happens to be bound to.
//!
//! `process_frame` used to take its device from `ResourceGroup::device()`, which
//! resolves through `cv_hal::compute::get_device_by_id` -> `GpuContext::global()`
//! - a **process-wide** `OnceLock`. Anything else in the process that initialises
//! it (`cv_runtime::registry()` does; so does almost every GPU test in the
//! workspace) makes `global()` succeed for everyone.
//!
//! So on a machine with a GPU, a `CpuStorage` image - which is what
//! `Tensor::from_vec` produces, what `Slam::process_image` hands over, and what
//! any image decoder produces - reached `ComputeDevice::Gpu`, and
//! `GpuContext::match_descriptors` rejects `CpuStorage` outright:
//!
//! ```text
//! Err("Invalid input: GpuContext requires GpuStorage tensors")
//! ```
//!
//! The order dependence is the point. Because `GLOBAL_CONTEXT` is global, whether
//! the tracker saw a GPU depended on which tests ran first in the process: the
//! same test failed on the first `cargo test` run and passed on later runs, and
//! passed under `cargo nextest`, which gives each test binary its own process.
//!
//! The fix selects the device from the input tensor's storage, so a CPU tensor
//! takes the CPU path regardless of what the group's device is.

#![forbid(unsafe_code)]

use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::compute::ComputeDevice;
use cv_slam::mapping::MapExt;
use cv_slam::tracking::{Tracker, TrackingFrame};
use cv_slam::{MapPoint, WorldMap};
use nalgebra::{Point3, Vector3};

const W: u32 = 128;
const H: u32 = 96;

/// A map the query frame cannot match, so the tracker cannot succeed.
///
/// All ten descriptors are identical on purpose: a tie in the ratio test is never
/// accepted, so the match count is 0 regardless of what the detector finds. That
/// makes the *only* thing under test the device selection - the frame's outcome
/// is a deterministic tracking failure, not a property of one texture.
fn unmatchable_map() -> WorldMap {
    let mut map = WorldMap::new();
    for i in 0..10u64 {
        map.add_point(MapPoint::new(
            i,
            Point3::new(i as f32, 0.0, 3.0),
            vec![0xA5; 32],
        ));
    }
    map
}

fn query_image() -> CpuTensor<u8> {
    let data: Vec<u8> = (0..(H * W))
        .map(|i| ((i * 37 + i / W * 91) % 256) as u8)
        .collect();
    Tensor::from_vec(data, TensorShape::new(1, H as usize, W as usize)).expect("query tensor")
}

fn group_on(
    device_id: cv_hal::backend::DeviceId,
) -> std::sync::Arc<cv_runtime::orchestrator::ResourceGroup> {
    std::sync::Arc::new(
        cv_runtime::orchestrator::ResourceGroup::new(
            "tracking-device-selection",
            device_id,
            1,
            None,
            cv_runtime::orchestrator::GroupPolicy::default(),
        )
        .expect("resource group"),
    )
}

fn intrinsics() -> cv_core::CameraIntrinsics {
    cv_core::CameraIntrinsics::new(500.0, 500.0, W as f64 / 2.0, H as f64 / 2.0, W, H)
}

/// The GPU context, or `None` on a machine without an adapter.
///
/// Initialising it is what poisons the tracker's device selection pre-fix, so
/// the tests below call this explicitly rather than hoping some other test in
/// the process did it first. That is the difference between a test that
/// reproduces the order dependence on demand and one that only catches it on
/// someone's first `cargo test` run.
fn global_gpu_context() -> Option<()> {
    match futures::executor::block_on(cv_hal::gpu::GpuContext::init_global()) {
        Ok(_) => Some(()),
        Err(e) => {
            println!("skipping: no GPU adapter available ({e})");
            None
        }
    }
}

/// The defect.
///
/// With a global GPU context published, `get_device()` returns a GPU device, and
/// every device-bound resource group is therefore a GPU group. A CPU tensor fed
/// to the tracker then hit `GpuContext::match_descriptors`, which refuses
/// `CpuStorage`, and the frame failed with a HAL storage error that had nothing
/// to do with tracking.
#[test]
fn a_cpu_tensor_is_tracked_even_while_a_global_gpu_context_exists() {
    if global_gpu_context().is_none() {
        return;
    }

    // CONTROL / guard. The defect only exists while `get_device()` is a GPU, so
    // assert that precondition rather than trusting it: if it silently becomes
    // CPU-only this test would pass vacuously.
    let device = cv_hal::compute::get_device().expect("a compute device");
    assert!(
        matches!(device, ComputeDevice::Gpu(_)),
        "this test only means anything when a global GPU context is published, \
         but get_device() resolved to {device:?}"
    );

    // The group is bound to that GPU device, exactly as a caller who asked for
    // GPU compute would construct it.
    let mut tracker = Tracker::new(group_on(device.device_id()), intrinsics());

    let mut map = unmatchable_map();
    let image = query_image();
    let result = tracker.process_frame(&image, &mut map);

    match &result {
        Ok(_) => {}
        Err(e) => {
            // The point of the test: the error must be about tracking, never
            // about where the tensor lives.
            assert!(
                !e.contains("GpuStorage") && !e.contains("GPU"),
                "a CPU-storage image reached a GPU kernel: {e}. The tracker's \
                 device must follow the input tensor's storage."
            );
            println!("tracking failed as expected for an unmatchable map: {e}");
        }
    }
}

/// The same defect reached through `Slam`, which is the entry point every real
/// caller uses and which always hands the tracker a `CpuStorage` tensor.
///
/// Pre-fix this returned the HAL storage error for every frame on a GPU machine,
/// i.e. SLAM could not run at all.
#[test]
fn slam_process_image_survives_a_global_gpu_context() {
    if global_gpu_context().is_none() {
        return;
    }
    let device = cv_hal::compute::get_device().expect("a compute device");
    let mut slam = cv_slam::Slam::new(group_on(device.device_id()), intrinsics());

    // CONTROL. An empty map short-circuits inside `process_frame` before any
    // matching happens, so it never reaches the device-dependent kernel and
    // would pass pre-fix for the wrong reason. Seed the map so the frame does
    // the descriptor matching that selects the device.
    slam.map.points = unmatchable_map().points;

    let mut img = image::GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            img.put_pixel(x, y, image::Luma([((x * 7 + y * 13) % 256) as u8]));
        }
    }
    if let Err(e) = slam.process_image(&img) {
        assert!(
            !e.contains("GpuStorage") && !e.contains("GPU"),
            "Slam handed a CPU image to a GPU kernel: {e}"
        );
        println!("unmatchable map refused as expected: {e}");
    }
}

/// A frame that fails to track must fall back to the previous pose, not error -
/// and it must do so on a GPU machine too.
///
/// This is the branch the reported symptom sat on ("a failure with a pose to
/// fall back on is not an error"), so it is the cheapest end-to-end way to see
/// whether the tracker got far enough to reach its own bookkeeping at all.
#[test]
fn a_failed_frame_still_falls_back_to_the_last_pose_on_a_gpu_machine() {
    if global_gpu_context().is_none() {
        return;
    }
    let device = cv_hal::compute::get_device().expect("a compute device");
    let mut tracker = Tracker::new(group_on(device.device_id()), intrinsics());

    let previous = cv_core::Pose::from_quat_translation(
        nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.7),
        Vector3::new(2.0, 0.5, -0.3),
    );
    tracker.last_frame = Some(TrackingFrame {
        image: Tensor::from_vec(
            vec![0u8; (W * H) as usize],
            TensorShape::new(1, H as usize, W as usize),
        )
        .expect("image tensor"),
        keypoints: cv_core::KeyPoints::default(),
        descriptors: cv_core::Descriptors::default(),
        pose: previous,
    });

    let mut map = unmatchable_map();
    let (pose, inliers) = tracker
        .process_frame(&query_image(), &mut map)
        .expect("a failure with a pose to fall back on is not an error");

    assert!(inliers.is_empty(), "no correspondence can be reported");
    let err = (pose.matrix() - previous.matrix())
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        err < 1e-12,
        "expected the previous pose, got {pose:?} (max element error {err:.3e})"
    );
}

/// Order independence, stated directly.
///
/// Both trackers are built in the same process, the second one after the global
/// GPU context has been published. Pre-fix the first would run on CPU and the
/// second on GPU, so the same input produced a storage error on the second
/// depending on nothing but test ordering. Post-fix both run the CPU path
/// because both inputs are CPU tensors, and neither outcome depends on the
/// global context at all.
#[test]
fn the_outcome_does_not_depend_on_the_global_device_context() {
    let image = query_image();

    // Phase 1: before anything has initialised the GPU context in this process.
    let cpu_device = cv_hal::compute::get_device().expect("a compute device");
    let mut before = Tracker::new(group_on(cpu_device.device_id()), intrinsics());
    let r1 = before.process_frame(&image, &mut unmatchable_map());

    // Phase 2: same input, same group construction, GPU context now published.
    let _ = global_gpu_context();
    let now_device = cv_hal::compute::get_device().expect("a compute device");
    let mut after = Tracker::new(group_on(now_device.device_id()), intrinsics());
    let r2 = after.process_frame(&image, &mut unmatchable_map());

    let shape = |r: &Result<(cv_core::Pose, Vec<usize>), String>| match r {
        Ok((p, idx)) => format!("Ok(translation {:?}, {} inliers)", p.translation, idx.len()),
        Err(e) => format!("Err({e})"),
    };
    println!("before global init: {}", shape(&r1));
    println!("after  global init: {}", shape(&r2));

    // The storage error is a device-selection failure, so it must appear in
    // neither phase.
    for (label, r) in [("before", &r1), ("after", &r2)] {
        if let Err(e) = r {
            assert!(
                !e.contains("GpuStorage"),
                "{label} phase hit the GPU-storage error: {e}"
            );
        }
    }
}
