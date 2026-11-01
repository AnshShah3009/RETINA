//! A frame that fails to track must not report a pose of its own.
//!
//! Current contract: a frame that cannot be tracked against a non-empty map
//! returns `Err("Tracking lost: ...")`, sets `Tracker::state` to `Lost`, and
//! leaves the last good frame in place. Returning the previous pose as `Ok` (the
//! fix described below) still made a lost frame look like a measurement.
//!
//! History: `Tracker::process_frame` fell back to the previous pose when
//! matching or PnP failed:
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
use cv_slam::tracking::{Tracker, TrackingFrame, TrackingState};
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
            vec![0xa5; 32],
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

/// Assert the failure contract: `Err`, state `Lost`, and the last good frame
/// left in place untouched.
fn assert_reported_loss(tracker: &mut Tracker, previous: &Pose) {
    let mut map = unmatchable_map();
    match tracker.process_frame(&query_image(), &mut map) {
        Ok((pose, inliers)) => panic!(
            "a frame that failed to track returned Ok(({:?}, {} inliers)); a lost frame \
             must be reported, not given a pose as if it had been measured",
            pose.translation,
            inliers.len()
        ),
        Err(e) => {
            println!("reported: {e}");
            assert!(e.contains("Tracking lost"), "unexpected error: {e}");
        }
    }
    assert_eq!(tracker.state, TrackingState::Lost);
    let kept = tracker
        .current_frame
        .as_ref()
        .or(tracker.last_frame.as_ref())
        .expect("the last good frame must be kept");
    assert!(
        matrix_error(&kept.pose, previous) < 1e-12,
        "the last good pose was overwritten: {:?}",
        kept.pose.translation
    );
}

/// A failure with an older frame in `last_frame` is reported as lost. This
/// used to return `Ok(previous pose)`: indistinguishable from a frame that had
/// actually been tracked to the same place.
#[test]
fn a_failed_frame_is_reported_when_there_is_a_previous_pose() {
    let mut tracker = tracker();
    let previous = known_pose();
    tracker.last_frame = Some(frame_with_pose(previous));
    assert_reported_loss(&mut tracker, &previous);
}

/// After one successful frame the tracker holds that frame in `current_frame`
/// and nothing in `last_frame`. A failure on the next frame once returned the
/// *identity* (the fabricated `Pose::default()`); it must be reported as lost
/// and leave the successful frame in place.
#[test]
fn a_failed_frame_after_a_successful_one_does_not_invent_a_pose() {
    let mut tracker = tracker();
    let previous = known_pose();
    tracker.current_frame = Some(frame_with_pose(previous));
    assert!(tracker.last_frame.is_none());
    assert_reported_loss(&mut tracker, &previous);
}
