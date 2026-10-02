//! A frame that fails to track must not report a pose of its own.
//!
//! `Tracker::process_frame` falls back to the previous pose when matching or PnP
//! fails:
//!
//! ```text
//! if !tracking_success {
//!     if let Some(ref last) = self.last_frame { frame.pose = last.pose; }
//!     else if self.current_frame.is_none() { return Err(...); }
//! }
//! ```
//!
//! `last_frame` is filled from `current_frame` only at the **end** of a call, so
//! after the first successful frame `current_frame` holds it and `last_frame` is
//! still empty. On that second frame a tracking failure satisfies neither
//! condition, and the frame keeps the `Pose::default()` it was constructed with:
//! the tracker returns `Ok((identity, vec![]))` for a camera whose last known
//! pose was somewhere else entirely. Both the fabricated pose and a real one are
//! well-formed matrices, so nothing downstream can tell them apart.
//!
//! The test drives exactly that state through the public fields (which are what
//! the tracker itself writes), so it does not depend on the ORB/PnP wobble of a
//! particular frame. The map is non-empty but unmatchable - a map from somewhere
//! else, i.e. a lost track - which is the real situation that reaches this code.

use cv_core::storage::CpuStorage;
use cv_core::{CameraIntrinsics, CpuTensor, Descriptors, KeyPoints, Pose, Tensor, TensorShape};
use cv_slam::mapping::MapExt;
use cv_slam::tracking::{Tracker, TrackingFrame};
use cv_slam::{MapPoint, WorldMap};
use nalgebra::{Point3, UnitQuaternion, Vector3};

const W: u32 = 128;
const H: u32 = 96;

/// A pose that is plainly not the identity, so a fabricated `Pose::default()` is
/// distinguishable from a real estimate.
fn known_pose() -> Pose {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.7);
    let t = Vector3::new(2.0, 0.5, -0.3);
    Pose::from_quat_translation(rotation, t)
}

fn frame_with_pose(pose: Pose) -> TrackingFrame<CpuStorage<u8>> {
    TrackingFrame {
        image: Tensor::from_vec(vec![0u8; (W * H) as usize], TensorShape::new(1, 96, 128))
            .expect("image tensor"),
        keypoints: KeyPoints::default(),
        descriptors: Descriptors::default(),
        pose,
    }
}

fn tracker() -> Tracker {
    let dev = cv_hal::compute::get_device().expect("a compute device");
    let group = std::sync::Arc::new(
        cv_runtime::orchestrator::ResourceGroup::new(
            "slam-failure-pose",
            dev.device_id(),
            1,
            None,
            cv_runtime::orchestrator::GroupPolicy::default(),
        )
        .expect("resource group"),
    );
    group.device().expect("the group's device must resolve");
    Tracker::new(group, CameraIntrinsics::new(500.0, 500.0, 64.0, 48.0, W, H))
}

/// A map the query frame cannot match, so matching yields fewer than the ten
/// landmarks the tracker requires.
///
/// All ten descriptors are identical on purpose: a tie in the ratio test is
/// never accepted, so the match count is 0 (or at most 1) regardless of what the
/// detector finds. That makes the failure path deterministic rather than a
/// property of one particular texture.
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

fn matrix_error(a: &Pose, b: &Pose) -> f64 {
    (a.matrix() - b.matrix())
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Control: when a previous pose *is* available, the failed frame reuses it. The
/// invariant below is not "always fail" - it is "never invent".
#[test]
fn a_failed_frame_reuses_the_last_pose_when_there_is_one() {
    let mut tracker = tracker();
    let previous = known_pose();
    tracker.last_frame = Some(frame_with_pose(previous));

    let mut map = unmatchable_map();
    let (pose, inliers) = tracker
        .process_frame(&query_image(), &mut map)
        .expect("a failure with a pose to fall back on is not an error");

    assert!(inliers.is_empty(), "no correspondence can be reported");
    assert!(
        matrix_error(&pose, &previous) < 1e-12,
        "expected the previous pose {:?}, got {:?}",
        previous.matrix(),
        pose.matrix()
    );
}

/// The defect: after one successful frame the tracker holds that frame in
/// `current_frame` and nothing in `last_frame`. A failure on the next frame has
/// no fallback branch and returns the *identity* instead of the last pose - or an
/// error.
#[test]
fn a_failed_frame_after_a_successful_one_does_not_invent_a_pose() {
    let mut tracker = tracker();
    let previous = known_pose();
    tracker.current_frame = Some(frame_with_pose(previous));

    // The state the tracker's own bookkeeping produces: the successful frame is
    // in `current_frame`, `last_frame` is still empty.
    assert!(tracker.last_frame.is_none());

    let mut map = unmatchable_map();
    match tracker.process_frame(&query_image(), &mut map) {
        // Failing is an honest answer when there is no pose to fall back on.
        Err(e) => println!("reported an error: {e}"),
        Ok((pose, inliers)) => {
            assert!(inliers.is_empty(), "no correspondence can be reported");
            let err = matrix_error(&pose, &previous);
            println!(
                "failed frame reported translation {:?} (last successful pose was {:?}), \
                 error vs the previous pose {err:.6}, distance from the identity {:.6}",
                pose.translation,
                previous.translation,
                pose.translation.norm()
            );
            assert!(
                err < 1e-12,
                "a frame that failed to track reported a pose of its own: {:?} instead of \
                 the last successful {:?} (or an error). It is {:.6} from the identity, \
                 which is exactly what the fabricated Pose::default() looks like.",
                pose.translation,
                previous.translation,
                pose.translation.norm()
            );
        }
    }
}
