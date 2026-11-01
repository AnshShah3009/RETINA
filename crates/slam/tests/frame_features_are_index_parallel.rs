//! The frame cv-slam builds from the detector must have index-parallel
//! keypoints and descriptors.
//!
//! `Tracker::track` does
//!
//! ```text
//! let (keypoints, descriptors) =
//!     detect_and_compute_ctx(&self.detector, &device, &self.group, image);
//! let mut frame = TrackingFrame { keypoints, descriptors, .. };
//! ```
//!
//! so `TrackingFrame::keypoints` and `TrackingFrame::descriptors` are exactly
//! what cv-features returned, and the frame is kept verbatim in
//! `Tracker::init_reference` / `current_frame` / `last_frame` - all public.
//! Every consumer of a frame therefore pairs feature `i` in one array with
//! feature `i` in the other.
//!
//! That pairing is only meaningful if the two arrays are index-parallel, which
//! cv-features states as a contract: `orb_detect_and_compute` documents "the
//! returned keypoints and descriptors are **parallel and equal in length**" and
//! derives the keypoint list from the descriptors for exactly this reason, and
//! `detect_and_compute_ctx`'s GPU branch pairs them by construction (`paired`
//! is zipped, sorted and truncated together).
//!
//! Its CPU branch did not. It returned the detector's raw keypoint list beside
//! the shorter descriptor list, so `keypoints[i]` is not the keypoint of
//! `descriptors[i]`. Measured on the 320x240 texture below: **500 keypoints
//! against 407 descriptors, and 407 of 407 keypoint positions disagree** - the
//! first one by 133 px. On the CPU path that is the *only* backend cv-slam uses
//! (`device_for_u8_tensor` selects CPU for a `CpuStorage` image, which is what
//! `Slam::process_image` and every decoder produce), so cv-slam was handed a
//! frame whose two feature arrays describe different pixels, and stored it.
//!
//! The pairing is not academic here: `add_keyframe` had to rebuild its keypoint
//! list from the descriptors to stay correct ("Keypoints taken from the
//! descriptors so that keyframe indices are descriptor indices"), and the
//! correspondence-building code that used to read `frame.keypoints[i]` fed PnP
//! the pixel of a different feature (see `tracking_descriptor_alignment.rs`).
//! Any caller reading the frame - debug tooling, a visualiser, a future
//! refinement stage - pairs a descriptor with the wrong pixel, silently.

use cv_core::{CameraIntrinsics, CpuTensor, Tensor, TensorShape};
use cv_features::detect_and_compute_ctx;
use cv_slam::tracking::Tracker;
use image::{GrayImage, Luma};

const W: u32 = 320;
const H: u32 = 240;

/// A deterministic pseudo-random texture, so the detector has plenty to
/// describe near - and past - every image border (the border is where
/// `Orb::extract` drops a keypoint, which is what desynchronises the arrays).
fn noise_image() -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        let v = (x.wrapping_mul(7919) ^ y.wrapping_mul(104729) ^ (x * x + y * y)) % 256;
        Luma([v as u8])
    })
}

fn tracker() -> Tracker {
    let dev = cv_hal::compute::get_device().expect("a compute device");
    let group = std::sync::Arc::new(
        cv_runtime::orchestrator::ResourceGroup::new(
            "slam-frame-feature-parity",
            dev.device_id(),
            1,
            None,
            cv_runtime::orchestrator::GroupPolicy::default(),
        )
        .expect("resource group"),
    );
    group.device().expect("the group's device must resolve");
    Tracker::new(
        group,
        CameraIntrinsics::new(500.0, 500.0, 160.0, 120.0, W, H),
    )
}

/// Assert one frame's two feature arrays are the same features in the same
/// order, and report how far apart they are when they are not.
fn assert_parallel(
    label: &str,
    frame: &cv_slam::tracking::TrackingFrame<cv_core::storage::CpuStorage<u8>>,
) {
    println!(
        "{label}: {} keypoints, {} descriptors",
        frame.keypoints.len(),
        frame.descriptors.len()
    );
    assert_eq!(
        frame.keypoints.len(),
        frame.descriptors.len(),
        "{label}: {} keypoints but {} descriptors. `Orb::extract` drops a \
         keypoint whose 31px patch leaves the image, so the detection list is \
         longer than the descriptor list; returning both unpaired leaves every \
         index past the first drop naming a different feature",
        frame.keypoints.len(),
        frame.descriptors.len()
    );

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    let mut mismatched = 0usize;
    for (i, d) in frame.descriptors.descriptors.iter().enumerate() {
        let kp = &frame.keypoints.keypoints[i];
        let err = ((d.keypoint.x - kp.x).powi(2) + (d.keypoint.y - kp.y).powi(2)).sqrt();
        if err > 1e-9 {
            mismatched += 1;
            if err > worst {
                worst = err;
                worst_at = i;
            }
        }
    }
    println!(
        "{label}: {mismatched} of {} keypoints are not the descriptor's keypoint; \
         worst {worst:.1} px at index {worst_at}",
        frame.descriptors.len()
    );
    assert_eq!(
        mismatched,
        0,
        "{label}: {mismatched} of {} keypoints are not the keypoint their \
         descriptor was computed from - worst {worst:.1} px at index {worst_at} \
         (keypoints[{worst_at}] = ({:.1}, {:.1}), descriptors[{worst_at}].keypoint \
         = ({:.1}, {:.1})). The frame cv-slam stores pairs a descriptor with the \
         pixel of a different feature",
        frame.descriptors.len(),
        frame.keypoints.keypoints[worst_at].x,
        frame.keypoints.keypoints[worst_at].y,
        frame.descriptors.descriptors[worst_at].keypoint.x,
        frame.descriptors.descriptors[worst_at].keypoint.y,
    );
}

/// The CPU compute device, which is the one cv-slam selects for a CPU image.
///
/// `device_for_u8_tensor` chooses the device from the input tensor's storage: a
/// `CpuStorage` image takes `ComputeDevice::Cpu`, never the resource group's
/// device. So the CPU branch of `detect_and_compute_ctx` is the one that runs for
/// every caller of `Slam::process_image` and every decoder-produced frame - the
/// GPU branch below is not on that path at all.
fn cpu_device() -> cv_hal::compute::ComputeDevice<'static> {
    use std::sync::OnceLock;
    static CPU: OnceLock<cv_hal::cpu::CpuBackend> = OnceLock::new();
    let backend = CPU.get_or_init(|| cv_hal::cpu::CpuBackend::new().expect("CPU backend"));
    cv_hal::compute::ComputeDevice::Cpu(backend)
}

/// The detector output cv-slam asks for, checked as cv-slam pairs it.
#[test]
fn the_detector_output_cv_slam_pairs_is_index_parallel() {
    let img = noise_image();
    let tensor: CpuTensor<u8> =
        Tensor::from_vec(img.to_vec(), TensorShape::new(1, H as usize, W as usize))
            .expect("image tensor");

    let tracker = tracker();
    // The device cv-slam picks for this tensor, not the group's.
    let device = cpu_device();
    assert!(
        matches!(device, cv_hal::compute::ComputeDevice::Cpu(_)),
        "this test is about the CPU path cv-slam takes for a CPU image"
    );
    let (keypoints, descriptors) =
        detect_and_compute_ctx(&tracker.detector, &device, &tracker.group, &tensor);
    println!(
        "detector: {} keypoints, {} descriptors",
        keypoints.len(),
        descriptors.len()
    );
    assert!(
        descriptors.len() >= 100,
        "need a texture the detector can describe, got {}",
        descriptors.len()
    );

    // Exactly the pairing `TrackingFrame` exposes to its callers.
    let frame = cv_slam::tracking::TrackingFrame {
        image: tensor.clone(),
        keypoints,
        descriptors,
        pose: cv_core::Pose::default(),
    };
    assert_parallel("detector output", &frame);
}

/// The frame cv-slam actually keeps: the monocular initialisation reference.
///
/// `process_frame` holds the first frame in `Tracker::init_reference` (a public
/// field) and never modifies its features, so this is byte for byte what the
/// detector handed the tracker.
#[test]
fn the_frame_the_tracker_stores_is_index_parallel() {
    let img = noise_image();
    let tensor: CpuTensor<u8> =
        Tensor::from_vec(img.to_vec(), TensorShape::new(1, H as usize, W as usize))
            .expect("image tensor");

    let mut tracker = tracker();
    let mut map = cv_slam::WorldMap::new();
    let err = tracker
        .process_frame(&tensor, &mut map)
        .expect_err("one monocular frame cannot yield a pose");
    println!("first frame held as the initialisation reference: {err}");

    let frame = tracker
        .init_reference
        .as_ref()
        .expect("the first frame must be held as the initialisation reference");
    assert_parallel("stored reference frame", frame);
}
