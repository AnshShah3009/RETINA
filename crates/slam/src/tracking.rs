use crate::mapping::MapExt;
use crate::types::{KeyFrame, MapPoint, WorldMap};
use crate::{local_ba, local_map};
use cv_calib3d::{recover_pose_from_essential, solve_pnp_ransac, triangulate_points};
use cv_core::storage::CpuStorage;
use cv_core::{storage::Storage, CameraIntrinsics, KeyPoints, Pose, Tensor};
use cv_features::{detect_and_compute_ctx, Descriptors, Orb};
use cv_hal::compute::ComputeDevice;
use cv_hal::cpu::CpuBackend;
use cv_runtime::orchestrator::ResourceGroup;
use nalgebra::{Matrix3x4, Point2, Point3, Vector3};
use std::sync::Arc;

pub struct TrackingFrame<S: Storage<u8> + 'static> {
    pub image: Tensor<u8, S>,
    pub keypoints: KeyPoints,
    pub descriptors: Descriptors,
    pub pose: Pose,
}

/// Where the tracker is in its life cycle.
///
/// `process_frame` / `process_frame_with_depth` return `Err` for every frame
/// that does not yield a measured pose; this tells the caller *why* without
/// parsing the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    /// No map and no initialisation reference yet.
    NotInitialized,
    /// Monocular only: a reference frame is held, waiting for a frame with
    /// enough parallax to triangulate the initial map.
    Initializing,
    /// The last frame was tracked against the map.
    Tracking,
    /// The last frame could not be tracked against a non-empty map. The map is
    /// kept, and every following frame is matched against all of it again, so
    /// returning to a mapped view relocalises.
    Lost,
}

/// Minimum PnP inliers for a frame to count as tracked.
const MIN_TRACKED_INLIERS: usize = 15;
/// Minimum keypoints with valid depth to bootstrap an RGB-D map.
const MIN_BOOTSTRAP_POINTS: usize = 30;
/// Minimum descriptor matches against the monocular reference frame. Fewer
/// means the reference no longer views the same scene, so it is replaced.
const MIN_INIT_MATCHES: usize = 50;
/// Minimum triangulated points for a monocular initialisation.
const MIN_INIT_POINTS: usize = 40;
/// Minimum median triangulation angle for a monocular initialisation.
const MIN_INIT_PARALLAX_DEG: f64 = 1.0;
/// An RGB-D keyframe is inserted when a tracked frame keeps fewer than this
/// fraction of the last keyframe's established map points (see
/// `KeyframeDistance::reference`)...
const KEYFRAME_TRACKED_RATIO: f64 = 0.6;
/// ...or has moved more than this fraction of the last keyframe's median scene
/// depth from it...
const KEYFRAME_BASELINE_RATIO: f64 = 0.15;
/// ...or has turned more than this many degrees away from it.
const KEYFRAME_ROTATION_DEG: f64 = 15.0;
/// Monocular keyframes need parallax to triangulate from, so none is inserted
/// closer than this fraction of the median depth to the last one...
const MONO_KEYFRAME_MIN_BASELINE_RATIO: f64 = 0.02;
/// ...and one is inserted when the tracked set falls below this fraction of
/// the last keyframe's map points, or the baseline exceeds
/// [`MONO_KEYFRAME_BASELINE_RATIO`], whichever comes first.
const MONO_KEYFRAME_TRACKED_RATIO: f64 = 0.8;
const MONO_KEYFRAME_BASELINE_RATIO: f64 = 0.08;
/// Keyframes before the newest one that a monocular keyframe is triangulated
/// against.
const MONO_TRIANGULATION_NEIGHBOURS: usize = 3;
/// Search radius (pixels) around a map point projected with the
/// constant-velocity prediction.
const PREDICTED_SEARCH_RADIUS: f64 = 20.0;
/// Search radius (pixels) around a map point projected with the coarse pose.
const REFINE_SEARCH_RADIUS: f64 = 5.0;
/// Default [`Tracker::local_ba_window`].
const LOCAL_BA_WINDOW: usize = 10;
/// Levenberg-Marquardt iterations per local bundle adjustment.
const LOCAL_BA_ITERATIONS: usize = 15;
/// Reprojection error (pixels) of a tracked map point in the final pose.
const TRACK_INLIER_PX: f64 = 2.0;

pub struct Tracker {
    pub group: Arc<ResourceGroup>,
    pub current_frame: Option<TrackingFrame<CpuStorage<u8>>>,
    pub last_frame: Option<TrackingFrame<CpuStorage<u8>>>,
    pub intrinsics: CameraIntrinsics,
    pub detector: Orb,
    pub last_depth: Option<Tensor<f32, CpuStorage<f32>>>,
    pub state: TrackingState,
    /// Monocular initialisation reference (world origin once initialised).
    pub init_reference: Option<TrackingFrame<CpuStorage<u8>>>,
    /// Motion between the last two tracked frames (`T_cur · T_prev⁻¹`), the
    /// constant-velocity prediction for the next one. `None` until two
    /// consecutive frames are tracked, and after tracking is lost.
    pub velocity: Option<Pose>,
    /// Keyframes optimised by the local bundle adjustment that runs after each
    /// new keyframe; 0 disables it.
    ///
    /// Monocular adjustment minimises reprojection error, then removes the
    /// scale gauge. RGB-D adjustment adds the creating keyframe's depth as a
    /// residual (see [`MapPoint::depth_m`]), which is what keeps the metric
    /// scale: reprojection alone moved keyframes 0.4–0.6 m off their depth
    /// points on the 20-frame sweep.
    pub local_ba_window: usize,
}

/// Result of matching a frame against the map.
struct MapTrack {
    pose: Pose,
    /// Map-point index of every PnP inlier.
    map_indices: Vec<usize>,
    /// Query descriptor index of every PnP inlier, parallel to `map_indices`.
    query_indices: Vec<usize>,
}

impl Tracker {
    pub fn new(group: Arc<ResourceGroup>, intrinsics: CameraIntrinsics) -> Self {
        Self {
            group,
            current_frame: None,
            last_frame: None,
            intrinsics,
            detector: Orb::default().with_n_features(500),
            last_depth: None,
            state: TrackingState::NotInitialized,
            init_reference: None,
            velocity: None,
            local_ba_window: LOCAL_BA_WINDOW,
        }
    }

    /// Track one monocular frame.
    ///
    /// * Empty map: two-view initialisation. The first frame is stored as the
    ///   reference (and becomes the world origin) and `Err` is returned; a later
    ///   frame with enough parallax initialises the map by essential matrix,
    ///   pose recovery and triangulation, and returns its pose. Monocular scale
    ///   is arbitrary: the initialisation baseline has length 1.
    /// * Non-empty map: tracked against the map (see `track_against_map`).
    ///   When the frame has moved far enough from the last keyframe, or tracks
    ///   too few of its points, it becomes a keyframe: new map points are
    ///   triangulated against the previous keyframes from features not yet in
    ///   the map, then a local bundle adjustment refines the newest keyframes
    ///   and their points (see [`Tracker::local_ba_window`]), and the frame's
    ///   adjusted pose is returned. Two map points that project onto the same
    ///   feature are merged, and a point that still has only its creating
    ///   observations two keyframes later is removed. RGB-D keeps every depth
    ///   point: one measurement already places it, and merging neighbours
    ///   there made the trajectory drift.
    ///
    /// Returns `Err` for every frame without a measured pose (initialising,
    /// tracking lost); [`Tracker::state`] says which.
    pub fn process_frame<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &mut self,
        image: &Tensor<u8, S>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        self.track(image, None, map)
    }

    fn track<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &mut self,
        image: &Tensor<u8, S>,
        depth: Option<&Tensor<f32, CpuStorage<f32>>>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        // The device must be chosen to match where the *data* is, not just what
        // the resource group happens to be bound to. See
        // `device_for_u8_tensor` for why.
        let device = device_for_u8_tensor::<S>(&image, &self.group)?;

        let (keypoints, descriptors) =
            detect_and_compute_ctx(&self.detector, &device, &self.group, image);

        // Convert to CPU for long-term storage and frame-to-frame matching if needed
        let cpu_img_tensor = convert_to_cpu(&device, image)
            .map_err(|e| format!("Failed to convert image to CPU: {}", e))?;

        let mut frame = TrackingFrame {
            image: cpu_img_tensor,
            keypoints,
            descriptors,
            pose: Pose::default(),
        };

        if map.points.is_empty() {
            return match depth {
                Some(d) => self.bootstrap_rgbd(frame, d, map),
                None => self.initialize_monocular(&device, frame, map),
            };
        }

        let tracked = match self.track_against_map(&device, &frame, map) {
            Ok(t) => t,
            Err(reason) => {
                // Tracking loss is reported, not papered over. This used to
                // return `Ok(last pose)`, indistinguishable from a measurement.
                // The previous good frame stays in `current_frame`.
                self.state = TrackingState::Lost;
                self.velocity = None;
                return Err(format!(
                    "Tracking lost: {reason}. The map is kept and the next frame is \
                     matched against all of it again."
                ));
            }
        };
        frame.pose = tracked.pose;
        let was_tracking = self.state == TrackingState::Tracking;
        self.state = TrackingState::Tracking;
        self.velocity = match (&self.current_frame, was_tracking) {
            (Some(prev), true) => Some(frame.pose.compose(&prev.pose.inverse())),
            _ => None,
        };

        if needs_keyframe(&frame.pose, tracked.map_indices.len(), map, depth.is_none()) {
            match depth {
                Some(d) => {
                    let fresh = depth_points(&self.intrinsics, &frame, d, &tracked.query_indices)?;
                    add_keyframe(&frame, &tracked, fresh, true, map);
                }
                None => {
                    // Monocular: the keyframe links what it tracked, then new
                    // points are triangulated against its predecessors from
                    // features neither has mapped yet.
                    add_keyframe(&frame, &tracked, Vec::new(), false, map);
                    local_map::triangulate_new_points(
                        &self.intrinsics,
                        map,
                        MONO_TRIANGULATION_NEIGHBOURS,
                    );
                }
            }
            if self.local_ba_window > 0 {
                if local_ba::local_bundle_adjust(
                    &self.intrinsics,
                    map,
                    self.local_ba_window,
                    1,
                    LOCAL_BA_ITERATIONS,
                )
                .is_some()
                {
                    // The new keyframe is this frame: report its adjusted pose.
                    if let Some(kf) = map.keyframes.last() {
                        let kf = kf.read().unwrap_or_else(|e| e.into_inner());
                        frame.pose = local_map::keyframe_world_to_camera(&kf.pose);
                    }
                    self.velocity = match (&self.current_frame, was_tracking) {
                        (Some(prev), true) => Some(frame.pose.compose(&prev.pose.inverse())),
                        _ => None,
                    };
                }
            }
            // After the pose is settled: fusion uses the refined points, and
            // culling removes the observations the adjustment just rejected.
            // Compaction renumbers every point, so the inlier list (indices
            // into the map as it was) is translated by birth id.
            let births = birth_ids(map, &tracked.map_indices);
            maintain_map(&self.intrinsics, map, depth.is_none());
            let result = (frame.pose, live_indices(map, &births));
            self.push_frame(frame);
            return Ok(result);
        }

        let result = (frame.pose, tracked.map_indices);
        self.push_frame(frame);
        Ok(result)
    }

    /// Constant-velocity prediction of the next frame's pose, while tracking.
    fn predicted_pose(&self) -> Option<Pose> {
        if self.state != TrackingState::Tracking {
            return None;
        }
        let last = self.current_frame.as_ref()?.pose;
        Some(match &self.velocity {
            Some(v) => v.compose(&last),
            None => last,
        })
    }

    fn push_frame(&mut self, frame: TrackingFrame<CpuStorage<u8>>) {
        self.last_frame = self.current_frame.take();
        self.current_frame = Some(frame);
    }

    /// Locate the frame against the map.
    ///
    /// 1. Coarse pose: while tracking, map points are projected with the
    ///    constant-velocity prediction and matched within
    ///    [`PREDICTED_SEARCH_RADIUS`], then PnP-RANSAC. Without a prediction
    ///    (after initialisation or loss), or if that fails, every map point is
    ///    matched by descriptor alone (relocalisation).
    /// 2. Fine pose: every map point is projected again with the coarse pose
    ///    and matched within [`REFINE_SEARCH_RADIUS`]; the pose is refined on
    ///    those matches, keeping those within [`TRACK_INLIER_PX`].
    ///
    /// Step 2 is what lets the tracked set follow a growing map: a global
    /// descriptor match with a ratio test drops every point that has a
    /// near-duplicate in the map, and keyframes that overlap create many.
    fn track_against_map(
        &self,
        device: &ComputeDevice,
        frame: &TrackingFrame<CpuStorage<u8>>,
        map: &mut WorldMap,
    ) -> Result<MapTrack, String> {
        if frame.descriptors.is_empty() {
            return Err("no features detected in the frame".to_string());
        }
        let k = &self.intrinsics;
        let coarse = match self
            .predicted_pose()
            .and_then(|p| self.coarse_by_projection(frame, map, &p))
        {
            Some(pose) => pose,
            None => self.coarse_by_descriptors(device, frame, map)?,
        };

        let pairs = local_map::match_by_projection(
            k,
            &frame.descriptors,
            map,
            &coarse,
            REFINE_SEARCH_RADIUS,
        );
        let (object, image) = correspondences(frame, map, &pairs);
        let (pose, inliers) = if object.len() >= MIN_TRACKED_INLIERS {
            local_map::refine_pose(k, &coarse, &object, &image, TRACK_INLIER_PX)
        } else {
            (coarse, vec![false; object.len()])
        };
        let (mut map_indices, mut query_indices) = (Vec::new(), Vec::new());
        for (&(mi, qi), &ok) in pairs.iter().zip(&inliers) {
            if ok {
                map_indices.push(mi);
                query_indices.push(qi);
            }
        }
        if map_indices.len() < MIN_TRACKED_INLIERS {
            return Err(format!(
                "{} map points within {TRACK_INLIER_PX} px of the refined pose, among {} \
                 projection matches (need {MIN_TRACKED_INLIERS})",
                map_indices.len(),
                pairs.len(),
            ));
        }
        Ok(MapTrack {
            pose,
            map_indices,
            query_indices,
        })
    }

    /// Coarse pose from map points matched around their predicted projection.
    fn coarse_by_projection(
        &self,
        frame: &TrackingFrame<CpuStorage<u8>>,
        map: &WorldMap,
        predicted: &Pose,
    ) -> Option<Pose> {
        let pairs = local_map::match_by_projection(
            &self.intrinsics,
            &frame.descriptors,
            map,
            predicted,
            PREDICTED_SEARCH_RADIUS,
        );
        let (object, image) = correspondences(frame, map, &pairs);
        if object.len() < 2 * MIN_TRACKED_INLIERS {
            return None;
        }
        let (pose, inliers) =
            solve_pnp_ransac(&object, &image, &self.intrinsics, None, 3.0, 100).ok()?;
        (inliers.iter().filter(|&&b| b).count() >= MIN_TRACKED_INLIERS).then_some(pose)
    }

    /// Coarse pose from every map point matched by descriptor alone (no pose
    /// prior): relocalisation, and the first frame after initialisation.
    fn coarse_by_descriptors(
        &self,
        device: &ComputeDevice,
        frame: &TrackingFrame<CpuStorage<u8>>,
        map: &mut WorldMap,
    ) -> Result<Pose, String> {
        // Cached on the map. This used to rebuild the whole map's descriptor
        // tensor on every frame - a `flat_map` clone of every descriptor in
        // the map, O(total map points) per frame - and the map only ever
        // grows, so the cost was paid in full on every subsequent frame.
        let n_points = map.points.len();
        let map_desc_bytes = map.descriptor_bytes();
        let map_desc_tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
            (*map_desc_bytes).clone(),
            cv_core::TensorShape::new(1, n_points, 32),
        )
        .map_err(|e| e.to_string())?;
        let query_desc_tensor = descriptor_tensor(&frame.descriptors)?;

        // Use accelerated matching via the device
        let matches = device
            .match_descriptors(&query_desc_tensor, &map_desc_tensor, 0.7)
            .map_err(|e| e.to_string())?;

        if matches.len() < 10 {
            return Err(format!(
                "{} descriptor matches against {} map points (need 10)",
                matches.len(),
                n_points
            ));
        }

        let mut object_pts = Vec::new();
        let mut image_pts = Vec::new();
        for m in &matches.matches {
            if let Ok(p) = map.points[m.train_idx as usize].read() {
                object_pts.push(Point3::new(
                    p.world_pos.x as f64,
                    p.world_pos.y as f64,
                    p.world_pos.z as f64,
                ));
                // The descriptor's own keypoint: the detector's keypoint and
                // descriptor arrays are not parallel (descriptors whose patch
                // leaves the image are dropped).
                let kp = &frame.descriptors.descriptors[m.query_idx as usize].keypoint;
                image_pts.push(Point2::new(kp.x, kp.y));
            }
        }

        let (pose, inliers) =
            solve_pnp_ransac(&object_pts, &image_pts, &self.intrinsics, None, 2.0, 100)
                .map_err(|e| format!("PnP failed: {e}"))?;

        // `inliers` is parallel to `object_pts` (one flag per successfully
        // read point), NOT to `matches`.
        let n_inliers = inliers.iter().filter(|&&ok| ok).count();
        if n_inliers < MIN_TRACKED_INLIERS {
            return Err(format!(
                "PnP found {} inliers among {} matches (need {})",
                n_inliers,
                object_pts.len(),
                MIN_TRACKED_INLIERS
            ));
        }
        Ok(pose)
    }

    /// RGB-D bootstrap: the first frame defines the world (identity pose), and
    /// every keypoint with valid depth is back-projected into a map point.
    fn bootstrap_rgbd(
        &mut self,
        mut frame: TrackingFrame<CpuStorage<u8>>,
        depth: &Tensor<f32, CpuStorage<f32>>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        frame.pose = Pose::identity();
        let empty = MapTrack {
            pose: frame.pose,
            map_indices: Vec::new(),
            query_indices: Vec::new(),
        };
        let candidates = depth_points(&self.intrinsics, &frame, depth, &empty.query_indices)?;
        if candidates.len() < MIN_BOOTSTRAP_POINTS {
            self.state = TrackingState::NotInitialized;
            return Err(format!(
                "RGB-D bootstrap failed: only {} keypoints (of {} descriptors) have valid \
                 depth, need {}. The map is still empty.",
                candidates.len(),
                frame.descriptors.len(),
                MIN_BOOTSTRAP_POINTS
            ));
        }
        let created = add_keyframe(&frame, &empty, candidates, true, map);
        self.state = TrackingState::Tracking;
        self.init_reference = None;
        let pose = frame.pose;
        self.push_frame(frame);
        Ok((pose, created))
    }

    /// Monocular two-view initialisation.
    fn initialize_monocular(
        &mut self,
        device: &ComputeDevice,
        mut frame: TrackingFrame<CpuStorage<u8>>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        let Some(reference) = self.init_reference.take() else {
            let n = frame.descriptors.len();
            self.init_reference = Some(frame);
            self.state = TrackingState::Initializing;
            return Err(format!(
                "Monocular initialisation: no pose yet. This frame ({n} features) is held \
                 as the reference; the map is built from the first later frame that sees \
                 the same scene from a translated viewpoint (rotation alone gives no \
                 parallax). Frames with depth can use process_frame_with_depth instead."
            ));
        };

        match two_view_init(device, &self.intrinsics, &reference, &frame) {
            Ok(init) => {
                frame.pose = init.pose;
                let mut created = insert_two_view_map(&reference, &frame, &init, map);
                // The init matched features by descriptor alone; with both
                // poses known, an epipolar-guided search finds more.
                created.extend(local_map::triangulate_new_points(&self.intrinsics, map, 1));
                if self.local_ba_window > 0 {
                    local_ba::local_bundle_adjust(&self.intrinsics, map, 2, 1, LOCAL_BA_ITERATIONS);
                    let kf = map.keyframes[1].read().unwrap_or_else(|e| e.into_inner());
                    frame.pose = local_map::keyframe_world_to_camera(&kf.pose);
                }
                // Two keyframes is not old enough to cull a point for lack of
                // reobservation; this removes points fusion emptied.
                let births = birth_ids(map, &created);
                maintain_map(&self.intrinsics, map, true);
                created = live_indices(map, &births);
                self.state = TrackingState::Tracking;
                let pose = frame.pose;
                // The reference is the world origin, so the motion from it is
                // the pose itself.
                self.velocity = Some(pose);
                self.current_frame = Some(reference);
                self.push_frame(frame);
                Ok((pose, created))
            }
            Err(InitFailure { reason, replace }) => {
                self.state = TrackingState::Initializing;
                let kept = if replace {
                    self.init_reference = Some(frame);
                    "this frame replaces the reference"
                } else {
                    self.init_reference = Some(reference);
                    "the reference is kept so parallax can accumulate"
                };
                Err(format!(
                    "Monocular initialisation: no pose yet ({reason}); {kept}."
                ))
            }
        }
    }

    /// Track one RGB-D frame. `depth` is metric depth (camera z) with the same
    /// height and width as `image`; non-finite or non-positive values are
    /// treated as missing.
    ///
    /// * Empty map: the frame is the world origin; keypoints with valid depth
    ///   are back-projected into map points and the frame becomes a keyframe.
    /// * Non-empty map: tracked against the map (see `track_against_map`); a
    ///   keyframe with new depth points is inserted when too few of the last
    ///   keyframe's established points remain tracked or the camera has moved
    ///   or turned far enough from it (see `needs_keyframe`). The same local
    ///   bundle adjustment as the monocular path then runs, with each new
    ///   point's depth as an extra residual so the keyframe cannot slide off
    ///   the measurement that placed it.
    pub fn process_frame_with_depth<
        S: Storage<u8> + cv_core::StorageFactory<u8> + 'static,
        D: Storage<f32> + cv_core::StorageFactory<f32> + 'static,
    >(
        &mut self,
        image: &Tensor<u8, S>,
        depth: &Tensor<f32, D>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        use cv_hal::storage::GpuStorage;
        use cv_hal::tensor_ext::TensorToCpu;

        let depth_cpu = if let Some(gpu_storage) =
            depth.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        {
            let input_gpu = Tensor {
                storage: gpu_storage.clone(),
                shape: depth.shape,
                dtype: depth.dtype,
                _phantom: std::marker::PhantomData,
            };
            let device = self
                .group
                .device()
                .map_err(|e| format!("Failed to get compute device: {}", e))?;
            let gpu_ctx = match device {
                ComputeDevice::Gpu(g) => g,
                _ => return Err("GPU depth tensor but the group has no GPU device".into()),
            };
            input_gpu
                .to_cpu_ctx(gpu_ctx)
                .map_err(|e| format!("Depth GPU download failed: {}", e))?
        } else if let Some(cpu_storage) = depth.storage.as_any().downcast_ref::<CpuStorage<f32>>() {
            Tensor {
                storage: cpu_storage.clone(),
                shape: depth.shape,
                dtype: depth.dtype,
                _phantom: std::marker::PhantomData,
            }
        } else {
            return Err(format!(
                "Unsupported depth storage type: {}",
                depth.storage.data_type_name()
            ));
        };
        if depth_cpu.shape.hw() != image.shape.hw() {
            return Err(format!(
                "depth is {:?} but the image is {:?}; they must be registered at the same \
                 resolution",
                depth_cpu.shape.hw(),
                image.shape.hw()
            ));
        }

        let (mut pose, tracked_indices) = self.track(image, Some(&depth_cpu), map)?;

        // Attempt ICP refinement if we have GPU support and previous depth data
        if let (Some(ref last_depth), Some(ref last_frame)) = (&self.last_depth, &self.last_frame) {
            let prev_pose = last_frame.pose;
            // Try to refine pose using dense ICP
            if let Err(e) = self.refine_pose_with_icp(&mut pose, &prev_pose, last_depth, &depth_cpu)
            {
                // If ICP fails, just continue with sparse tracking result
                eprintln!("ICP refinement failed: {}", e);
            }
        }

        // Store current depth for next frame
        self.last_depth = Some(depth_cpu);

        Ok((pose, tracked_indices))
    }

    fn refine_pose_with_icp(
        &self,
        pose: &mut Pose,
        prev_pose: &Pose,
        prev_depth: &Tensor<f32, cv_core::storage::CpuStorage<f32>>,
        curr_depth: &Tensor<f32, cv_core::storage::CpuStorage<f32>>,
    ) -> Result<(), String> {
        use cv_hal::tensor_ext::TensorToGpu;

        let device = self
            .group
            .device()
            .map_err(|e| format!("Failed to get compute device: {}", e))?;

        let gpu_ctx = match device {
            ComputeDevice::Gpu(g) => g,
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => return Ok(()),
        };

        // Convert depth tensors to GPU
        let prev_depth_gpu = prev_depth
            .to_gpu_ctx(gpu_ctx)
            .map_err(|e| format!("Previous depth GPU upload failed: {}", e))?;
        let curr_depth_gpu = curr_depth
            .to_gpu_ctx(gpu_ctx)
            .map_err(|e| format!("Current depth GPU upload failed: {}", e))?;

        // Build intrinsics array [fx, fy, cx, cy]
        let intrinsics = [
            self.intrinsics.fx as f32,
            self.intrinsics.fy as f32,
            self.intrinsics.cx as f32,
            self.intrinsics.cy as f32,
        ];

        // The shader aligns current-frame points into the previous frame, so
        // the estimate it needs is the RELATIVE transform
        //   T_rel = T_prev · T_curr⁻¹   (curr-cam coords -> prev-cam coords),
        // not the absolute world->camera pose.
        let t_rel = prev_pose.matrix() * pose.inverse().matrix();
        let pose_matrix_f32 = nalgebra::Matrix4::from_iterator(t_rel.iter().map(|&x| x as f32));

        // Call dense ICP
        let (ata, atb) = cv_hal::gpu_kernels::icp::dense_step(
            gpu_ctx,
            &prev_depth_gpu,
            &curr_depth_gpu,
            &intrinsics,
            &pose_matrix_f32,
            0.05,                  // max_dist: 5cm
            30.0_f32.to_radians(), // max_angle: 30 degrees
        )
        .map_err(|e| format!("ICP step failed: {}", e))?;

        // Solve the linear system for pose update: atA * delta = -atb
        // The ICP energy is E = sum ||r_i||^2 where r_i are residuals.
        // The normal equations give atA * delta = -atb (gradient descent direction).
        let delta = -(ata
            .try_inverse()
            .ok_or_else(|| "ICP AtA matrix not invertible".to_string())?
            * atb);

        // Update pose with delta. The GPU kernel packs the ICP Jacobian as
        //   J = [ n (translation) , cross(p, n) (rotation) ],
        // so the solved increment is delta = [translation(3), omega(3)].
        let translation = nalgebra::Vector3::new(delta[0] as f64, delta[1] as f64, delta[2] as f64);
        // Small angle approximation for rotation from omega
        let omega = nalgebra::Vector3::new(delta[3] as f64, delta[4] as f64, delta[5] as f64);

        *pose = apply_icp_increment(&t_rel, prev_pose, &translation, &omega);

        Ok(())
    }
}

/// A frame's descriptors as the `(1, n, 32)` tensor the matcher takes.
fn descriptor_tensor(descriptors: &Descriptors) -> Result<Tensor<u8, CpuStorage<u8>>, String> {
    let mut flat = Vec::with_capacity(descriptors.len() * 32);
    for d in &descriptors.descriptors {
        let len = d.data.len().min(32);
        flat.extend_from_slice(&d.data[..len]);
        flat.resize(flat.len() + (32 - len), 0);
    }
    Tensor::from_vec(flat, cv_core::TensorShape::new(1, descriptors.len(), 32))
        .map_err(|e| e.to_string())
}

/// Keyframe pose as stored on [`KeyFrame`]: camera -> world (`position` is the
/// camera centre), from the tracker's world -> camera `pose`.
fn keyframe_pose(pose: &Pose) -> cv_core::TypedPose {
    let c2w = pose.inverse();
    cv_core::TypedPose::new(
        Point3::from(c2w.translation.cast::<f32>()),
        c2w.rotation_matrix().cast::<f32>(),
        cv_core::FrameConvention::opencv(),
    )
}

/// World points and pixel observations of `(map index, descriptor index)`
/// pairs, in order.
fn correspondences(
    frame: &TrackingFrame<CpuStorage<u8>>,
    map: &WorldMap,
    pairs: &[(usize, usize)],
) -> (Vec<Point3<f64>>, Vec<Point2<f64>>) {
    pairs
        .iter()
        .map(|&(mi, qi)| {
            let p = map.points[mi].read().unwrap_or_else(|e| e.into_inner());
            let kp = frame.descriptors.descriptors[qi].keypoint;
            (p.world_pos.cast::<f64>(), Point2::new(kp.x, kp.y))
        })
        .unzip()
}

/// Birth ids of `indices`, so the list can be translated after culling
/// compacts the map. A point culling removes simply disappears from the result.
fn birth_ids(map: &WorldMap, indices: &[usize]) -> Vec<u64> {
    indices
        .iter()
        .filter_map(|&i| {
            map.points
                .get(i)
                .map(|p| p.read().unwrap_or_else(|e| e.into_inner()).created_at)
        })
        .collect()
}

fn live_indices(map: &WorldMap, births: &[u64]) -> Vec<usize> {
    let mut at = std::collections::HashMap::with_capacity(map.points.len());
    for (i, p) in map.points.iter().enumerate() {
        let born = p.read().unwrap_or_else(|e| e.into_inner()).created_at;
        at.insert(born, i);
    }
    births.iter().filter_map(|b| at.get(b).copied()).collect()
}

/// Fuse duplicates into the newest keyframe and drop points that were never
/// seen again.
///
/// Monocular map maintenance after a new keyframe.
///
/// Points that project onto the same feature are merged, and a point that
/// still has only its creating observations (a triangulated pair; it must be
/// seen from one more keyframe) two keyframes later is removed. RGB-D does
/// neither. A depth point is already metric, and on the 20-frame sweep
/// merging neighbours moved the camera from 2.9 cm / 0.39° to 4.8 cm / 0.65°.
/// See [`local_map::fuse_keyframe`] and [`local_map::cull_under_observed`].
fn maintain_map(k: &CameraIntrinsics, map: &mut WorldMap, monocular: bool) {
    if !monocular {
        return;
    }
    if let Some(idx) = map.keyframes.len().checked_sub(1) {
        local_map::fuse_keyframe(k, map, idx);
    }
    local_map::cull_under_observed(map, 3);
}

/// Whether a tracked frame should become a keyframe.
///
/// RGB-D: when it keeps fewer than [`KEYFRAME_TRACKED_RATIO`] of the
/// established map points the last keyframe observes, or has moved [`KEYFRAME_BASELINE_RATIO`] of the
/// scene depth or turned [`KEYFRAME_ROTATION_DEG`] away from it. This used to
/// be the tracked-ratio test alone, against a tracked set that a global
/// descriptor match kept small (~20-25% of a keyframe's points), so every
/// frame became a keyframe (20 of 20 on the 20-frame sweep; now 5).
///
/// Monocular: the same idea with a stricter ratio and baseline (a keyframe is
/// also where the map grows, so it must come before the tracked set thins
/// out), but never closer than [`MONO_KEYFRAME_MIN_BASELINE_RATIO`]: without
/// parallax nothing can be triangulated.
fn needs_keyframe(pose: &Pose, n_tracked: usize, map: &WorldMap, monocular: bool) -> bool {
    let Some(d) = local_map::distance_to_last_keyframe(map, pose) else {
        return !monocular;
    };
    let tracked = n_tracked as f64;
    let reference = d.reference as f64;
    if monocular {
        d.baseline_ratio >= MONO_KEYFRAME_MIN_BASELINE_RATIO
            && (tracked < MONO_KEYFRAME_TRACKED_RATIO * reference
                || d.baseline_ratio > MONO_KEYFRAME_BASELINE_RATIO)
    } else {
        tracked < KEYFRAME_TRACKED_RATIO * reference
            || d.baseline_ratio > KEYFRAME_BASELINE_RATIO
            || d.rotation_deg > KEYFRAME_ROTATION_DEG
    }
}

/// Back-project every descriptor's keypoint that has valid depth, skipping the
/// descriptor indices in `skip`. Returns `(descriptor index, camera point)`.
///
/// Depth is read at the nearest pixel. A keypoint whose 3x3 neighbourhood spans
/// a depth discontinuity (range above 10% of its depth) is rejected: corners on
/// an occlusion boundary would otherwise take the depth of either surface.
fn depth_points(
    k: &CameraIntrinsics,
    frame: &TrackingFrame<CpuStorage<u8>>,
    depth: &Tensor<f32, CpuStorage<f32>>,
    skip: &[usize],
) -> Result<Vec<(usize, Point3<f64>)>, String> {
    let (h, w) = depth.shape.hw();
    let data = depth
        .storage
        .as_slice()
        .ok_or_else(|| "depth tensor is not host-readable".to_string())?;
    let valid = |z: f32| z.is_finite() && z > 0.0;
    let mut out = Vec::new();
    for (i, d) in frame.descriptors.descriptors.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        let (u, v) = (d.keypoint.x, d.keypoint.y);
        let (ui, vi) = (u.round() as i64, v.round() as i64);
        if ui < 1 || vi < 1 || ui >= w as i64 - 1 || vi >= h as i64 - 1 {
            continue;
        }
        let z = data[vi as usize * w + ui as usize];
        if !valid(z) {
            continue;
        }
        let (mut lo, mut hi) = (z, z);
        let mut holes = false;
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let zn = data[(vi + dy) as usize * w + (ui + dx) as usize];
                holes |= !valid(zn);
                lo = lo.min(zn);
                hi = hi.max(zn);
            }
        }
        if holes || hi - lo > 0.1 * z {
            continue;
        }
        let z = z as f64;
        out.push((
            i,
            Point3::new((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z),
        ));
    }
    Ok(out)
}

/// Make `frame` a keyframe: link the tracked map points and add `new_points`
/// (camera coordinates, keyed by descriptor index) to the map. Returns the map
/// indices of the created points.
///
/// `from_depth` is set only when `new_points` were back-projected from a depth
/// image. Monocular initialisation also passes points here, but those are
/// triangulated in the reference camera and their z is not a measurement: the
/// scale is arbitrary, and stamping it made local bundle adjustment pin that
/// gauge.
fn add_keyframe(
    frame: &TrackingFrame<CpuStorage<u8>>,
    tracked: &MapTrack,
    new_points: Vec<(usize, Point3<f64>)>,
    from_depth: bool,
    map: &mut WorldMap,
) -> Vec<usize> {
    let kf_id = map.keyframes.len() as u64;
    // Keypoints taken from the descriptors so that keyframe indices are
    // descriptor indices.
    let keypoints = KeyPoints {
        keypoints: frame
            .descriptors
            .descriptors
            .iter()
            .map(|d| d.keypoint)
            .collect(),
    };
    let mut kf = KeyFrame::new(
        kf_id,
        keyframe_pose(&frame.pose),
        keypoints,
        frame.descriptors.clone(),
    );

    for (&mi, &qi) in tracked.map_indices.iter().zip(&tracked.query_indices) {
        let mut p = map.points[mi].write().unwrap_or_else(|p| p.into_inner());
        p.observations.push((kf_id, qi));
        kf.map_points[qi] = Some(p.id);
    }

    let c2w = frame.pose.inverse();
    let mut created = Vec::with_capacity(new_points.len());
    for (qi, p_cam) in new_points {
        let idx = map.points.len();
        let world = c2w.transform_point(&p_cam);
        let mut mp = MapPoint::new(
            idx as u64,
            world.cast::<f32>(),
            frame.descriptors.descriptors[qi].data.clone(),
        );
        mp.observations.push((kf_id, qi));
        if from_depth && p_cam.z.is_finite() && p_cam.z > 0.0 {
            mp.depth_m = Some(p_cam.z as f32);
        }
        map.add_point(mp);
        kf.map_points[qi] = Some(idx as u64);
        created.push(idx);
    }
    map.keyframes.push(Arc::new(std::sync::RwLock::new(kf)));
    created
}

/// Why a monocular initialisation attempt failed, and whether the reference
/// frame should be replaced (it no longer sees the scene) or kept (the scene
/// is shared but parallax is still insufficient).
struct InitFailure {
    reason: String,
    replace: bool,
}

struct TwoViewInit {
    /// World (= reference camera) -> current camera. Baseline length 1.
    pose: Pose,
    /// (reference descriptor index, current descriptor index, world point).
    points: Vec<(usize, usize, Point3<f64>)>,
}

/// Two-view initialisation: match, essential matrix by RANSAC, pose recovery
/// by cheirality, triangulation, and per-point checks (positive depth in both
/// views, reprojection error, triangulation angle).
fn two_view_init(
    device: &ComputeDevice,
    k: &CameraIntrinsics,
    reference: &TrackingFrame<CpuStorage<u8>>,
    frame: &TrackingFrame<CpuStorage<u8>>,
) -> Result<TwoViewInit, InitFailure> {
    let fail = |reason: String, replace: bool| InitFailure { reason, replace };
    if reference.descriptors.is_empty() || frame.descriptors.is_empty() {
        return Err(fail("no features to match".into(), true));
    }
    let query = descriptor_tensor(&frame.descriptors).map_err(|e| fail(e, true))?;
    let train = descriptor_tensor(&reference.descriptors).map_err(|e| fail(e, true))?;
    let matches = device
        .match_descriptors(&query, &train, 0.7)
        .map_err(|e| fail(e.to_string(), true))?;
    if matches.len() < MIN_INIT_MATCHES {
        return Err(fail(
            format!(
                "{} matches with the reference, need {}",
                matches.len(),
                MIN_INIT_MATCHES
            ),
            true,
        ));
    }

    let kp = |f: &TrackingFrame<CpuStorage<u8>>, i: i32| {
        let p = f.descriptors.descriptors[i as usize].keypoint;
        Point2::new(p.x, p.y)
    };
    let pts_ref: Vec<Point2<f64>> = matches
        .matches
        .iter()
        .map(|m| kp(reference, m.train_idx))
        .collect();
    let pts_cur: Vec<Point2<f64>> = matches
        .matches
        .iter()
        .map(|m| kp(frame, m.query_idx))
        .collect();

    let (e, inliers) = essential_ransac_5point(&pts_ref, &pts_cur, k, 1.0, 500)
        .ok_or_else(|| fail("essential matrix: RANSAC found no model".into(), false))?;
    let idx: Vec<usize> = (0..inliers.len()).filter(|&i| inliers[i]).collect();
    if idx.len() < MIN_INIT_POINTS {
        return Err(fail(
            format!(
                "{} essential-matrix inliers, need {}",
                idx.len(),
                MIN_INIT_POINTS
            ),
            false,
        ));
    }
    let in_ref: Vec<Point2<f64>> = idx.iter().map(|&i| pts_ref[i]).collect();
    let in_cur: Vec<Point2<f64>> = idx.iter().map(|&i| pts_cur[i]).collect();
    let pose = recover_pose_from_essential(&e, &in_ref, &in_cur, k)
        .map_err(|e| fail(format!("pose recovery: {e}"), false))?;

    let normalize = |p: &Point2<f64>| Point2::new((p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy);
    let n_ref: Vec<Point2<f64>> = in_ref.iter().map(normalize).collect();
    let n_cur: Vec<Point2<f64>> = in_cur.iter().map(normalize).collect();
    let r = pose.rotation_matrix();
    let p1 = Matrix3x4::identity();
    let mut p2 = Matrix3x4::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    p2.fixed_view_mut::<3, 1>(0, 3).copy_from(&pose.translation);
    let tri = triangulate_points(&p1, &p2, &n_ref, &n_cur)
        .map_err(|e| fail(format!("triangulation: {e}"), false))?;

    let c2: Vector3<f64> = -(r.transpose() * pose.translation);
    let reproj = |pc: &Vector3<f64>, obs: &Point2<f64>| {
        let u = k.fx * pc.x / pc.z + k.cx;
        let v = k.fy * pc.y / pc.z + k.cy;
        ((u - obs.x).powi(2) + (v - obs.y).powi(2)).sqrt()
    };
    let mut points = Vec::new();
    let mut angles = Vec::new();
    for (j, x) in tri.iter().enumerate() {
        if !(x.x.is_finite() && x.y.is_finite() && x.z.is_finite()) {
            continue;
        }
        let x1 = x.coords;
        let x2 = r * x1 + pose.translation;
        if x1.z <= 0.0 || x2.z <= 0.0 {
            continue;
        }
        if reproj(&x1, &in_ref[j]) > 2.0 || reproj(&x2, &in_cur[j]) > 2.0 {
            continue;
        }
        let a = x1;
        let b = x1 - c2;
        let angle = (a.dot(&b) / (a.norm() * b.norm()))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < 0.2 {
            continue;
        }
        let m = &matches.matches[idx[j]];
        points.push((m.train_idx as usize, m.query_idx as usize, *x));
        angles.push(angle);
    }
    if points.len() < MIN_INIT_POINTS {
        return Err(fail(
            format!(
                "{} points triangulated cleanly, need {}",
                points.len(),
                MIN_INIT_POINTS
            ),
            false,
        ));
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = angles[angles.len() / 2];
    if median < MIN_INIT_PARALLAX_DEG {
        return Err(fail(
            format!(
                "insufficient parallax: median triangulation angle {median:.2} deg, need \
                 {MIN_INIT_PARALLAX_DEG} deg"
            ),
            false,
        ));
    }
    Ok(TwoViewInit { pose, points })
}

/// Essential matrix (`x2ᵀ E x1 = 0`, pixel inputs) by RANSAC over Nistér's
/// 5-point solver, scored by Sampson distance.
///
/// Not `cv_calib3d::find_essential_mat_ransac`: that drives the 8-point
/// solver, which is degenerate when the correspondences are (near-)planar. On
/// the synthetic wall-and-boxes scene, where most features sit on the back
/// wall, it returned 70/82 "inliers" for a baseline ~55 degrees away from the
/// true one, and only 30 points survived triangulation. The 5-point solver has
/// no planar degeneracy: 81/82 inliers, baseline within ~5 degrees, all 81
/// points triangulated.
fn essential_ransac_5point(
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    k: &CameraIntrinsics,
    threshold_px: f64,
    iterations: usize,
) -> Option<(nalgebra::Matrix3<f64>, Vec<bool>)> {
    let n = pts1.len();
    if n < 5 || pts2.len() != n {
        return None;
    }
    let norm = |p: &Point2<f64>| Vector3::new((p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy, 1.0);
    let n1: Vec<Vector3<f64>> = pts1.iter().map(norm).collect();
    let n2: Vec<Vector3<f64>> = pts2.iter().map(norm).collect();
    let f = 0.5 * (k.fx + k.fy);
    let thr = (threshold_px / f).powi(2);
    let sampson = |e: &nalgebra::Matrix3<f64>, a: &Vector3<f64>, b: &Vector3<f64>| {
        let ea = e * a;
        let etb = e.transpose() * b;
        let num = b.dot(&ea).powi(2);
        let den = ea.x * ea.x + ea.y * ea.y + etb.x * etb.x + etb.y * etb.y;
        if den > 0.0 {
            num / den
        } else {
            f64::INFINITY
        }
    };

    // Deterministic sampling, so a sequence tracks the same way every run.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ n as u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut best: Option<(nalgebra::Matrix3<f64>, Vec<bool>, usize, f64)> = None;
    for _ in 0..iterations {
        let mut sample = [0usize; 5];
        let mut filled = 0;
        while filled < 5 {
            let c = next() % n;
            if !sample[..filled].contains(&c) {
                sample[filled] = c;
                filled += 1;
            }
        }
        let s1: Vec<[f64; 2]> = sample.iter().map(|&i| [n1[i].x, n1[i].y]).collect();
        let s2: Vec<[f64; 2]> = sample.iter().map(|&i| [n2[i].x, n2[i].y]).collect();
        let Ok(models) = cv_calib3d::EssentialSolver::estimate_5point(&s1, &s2) else {
            continue;
        };
        for e in models {
            let mut mask = vec![false; n];
            let mut count = 0;
            let mut cost = 0.0;
            for i in 0..n {
                let d = sampson(&e, &n1[i], &n2[i]);
                if d < thr {
                    mask[i] = true;
                    count += 1;
                    cost += d;
                }
            }
            let better = match &best {
                None => count > 0,
                Some((_, _, c, s)) => count > *c || (count == *c && cost < *s),
            };
            if better {
                best = Some((e, mask, count, cost));
            }
        }
    }
    best.map(|(e, mask, _, _)| (e, mask))
}

/// Insert the two initialisation keyframes and their triangulated points.
fn insert_two_view_map(
    reference: &TrackingFrame<CpuStorage<u8>>,
    frame: &TrackingFrame<CpuStorage<u8>>,
    init: &TwoViewInit,
    map: &mut WorldMap,
) -> Vec<usize> {
    let none = MapTrack {
        pose: Pose::identity(),
        map_indices: Vec::new(),
        query_indices: Vec::new(),
    };
    // Reference keyframe at the origin: points are in its camera frame.
    //
    // The descriptor matcher works query by query, so two current-frame
    // features can both name the same reference feature. A feature is one 3D
    // point and the reference keyframe has one slot per feature, so a second
    // landmark there could never be linked: its observation would point at a
    // slot naming the first. Keep the first correspondence for each reference
    // feature and drop the rest; the query feature of a dropped pair is then
    // simply not mapped by the initialisation.
    let mut taken = vec![false; reference.descriptors.descriptors.len()];
    let mut ref_points: Vec<(usize, Point3<f64>)> = Vec::with_capacity(init.points.len());
    let mut query_indices: Vec<usize> = Vec::with_capacity(init.points.len());
    for &(ri, qi, x) in &init.points {
        if taken.get(ri).copied().unwrap_or(true) {
            continue;
        }
        taken[ri] = true;
        ref_points.push((ri, x));
        query_indices.push(qi);
    }
    let created = add_keyframe(reference, &none, ref_points, false, map);
    // Current keyframe observes the same points.
    let tracked = MapTrack {
        pose: frame.pose,
        map_indices: created.clone(),
        query_indices,
    };
    add_keyframe(frame, &tracked, Vec::new(), false, map);
    created
}

/// Apply a solved ICP increment to the relative transform and return the new
/// absolute (world -> camera) pose of the current frame.
///
/// `t_rel` is `T_prev · T_curr⁻¹` (current-camera coordinates -> previous-camera
/// coordinates), the transform the dense ICP kernel linearises. Factored out of
/// `refine_pose_with_icp` because that path needs a GPU, while the composition
/// is pure arithmetic that has to be testable everywhere.
pub(crate) fn apply_icp_increment(
    t_rel: &nalgebra::Matrix4<f64>,
    prev_pose: &Pose,
    translation: &nalgebra::Vector3<f64>,
    omega: &nalgebra::Vector3<f64>,
) -> Pose {
    // Create rotation matrix from axis-angle (small angle)
    let angle = omega.norm();
    let rot_update = if angle > 1e-6 {
        let axis = omega / angle;
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
    } else {
        nalgebra::Rotation3::identity()
    };

    // The solver linearizes a LEFT increment on T_rel:
    //   T_rel' = exp(ξ̂) · T_rel,  ξ = [translation(3), omega(3)]
    // i.e. R ← ΔR·R and t ← ΔR·t + v. The increment matrix is therefore
    // exp(ξ̂) = [ΔR | v]: its translation is `v` alone. The `ΔR·t` term comes
    // out of the matrix product below. A previous revision also put `ΔR·t`
    // into the increment's translation, so it was added twice - a zero
    // update doubled the relative translation.
    let mut delta_mat = nalgebra::Matrix4::<f64>::identity();
    delta_mat
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rot_update.into_inner());
    delta_mat
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(translation);

    let t_rel_new = delta_mat * t_rel;

    // Write back the new absolute pose: T_curr' = (T_rel')⁻¹ · T_prev.
    let t_prev = prev_pose.matrix();
    let t_curr_new = t_rel_new.try_inverse().unwrap_or(*t_rel) * t_prev;
    Pose::new(
        t_curr_new.fixed_view::<3, 3>(0, 0).into_owned(),
        t_curr_new.fixed_view::<3, 1>(0, 3).into_owned(),
    )
}

#[cfg(test)]
mod icp_increment_tests {
    use super::*;
    use nalgebra::{Matrix4, UnitQuaternion, Vector3};

    fn max_abs(m: &Matrix4<f64>) -> f64 {
        m.iter().fold(0.0f64, |a, v| a.max(v.abs()))
    }

    fn poses() -> (Pose, Pose) {
        let prev = Pose::from_quat_translation(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.05),
            Vector3::new(0.3, -0.1, 0.8),
        );
        let curr = Pose::from_quat_translation(
            UnitQuaternion::from_euler_angles(0.12, -0.17, 0.02),
            Vector3::new(0.5, 0.05, 0.6),
        );
        (prev, curr)
    }

    /// A zero ICP update must leave the pose exactly where it was.
    #[test]
    fn zero_increment_is_the_identity() {
        let (prev, curr) = poses();
        let t_rel = prev.matrix() * curr.inverse().matrix();
        let out = apply_icp_increment(&t_rel, &prev, &Vector3::zeros(), &Vector3::zeros());
        let err = max_abs(&(out.matrix() - curr.matrix()));
        assert!(
            err < 1e-12,
            "zero ICP increment moved the pose by {err:.6}: expected {:?}, got {:?}",
            curr.translation,
            out.translation
        );
    }

    /// A non-zero increment must be applied exactly as the left update
    /// `T_rel' = [ΔR | v] · T_rel` that the solver linearises.
    #[test]
    fn nonzero_increment_is_a_left_update_on_t_rel() {
        let (prev, curr) = poses();
        let t_rel = prev.matrix() * curr.inverse().matrix();
        let v = Vector3::new(0.01, -0.02, 0.005);
        let w = Vector3::new(0.003, 0.002, -0.004);
        let out = apply_icp_increment(&t_rel, &prev, &v, &w);

        let rot = nalgebra::Rotation3::from_scaled_axis(w);
        let mut delta = Matrix4::identity();
        delta.fixed_view_mut::<3, 3>(0, 0).copy_from(rot.matrix());
        delta.fixed_view_mut::<3, 1>(0, 3).copy_from(&v);
        let expected = (delta * t_rel).try_inverse().unwrap() * prev.matrix();
        let err = max_abs(&(out.matrix() - expected));
        assert!(err < 1e-12, "left update mismatch {err:.3e}");
    }
}

/// Pick the [`ComputeDevice`] the tracker should run on, given where the input
/// tensor's storage actually lives.
///
/// # Why this exists
///
/// `ResourceGroup::device()` resolves whatever device the group is *bound* to.
/// That has nothing to do with where the caller's tensor is stored, and the two
/// can disagree in the most common configuration there is.
///
/// `ResourceGroup::device` -> `get_device_by_id` -> `GpuContext::global()`, and
/// `GLOBAL_CONTEXT` is a process-wide `OnceLock`. Any other code in the same
/// process that initialises it - `cv_runtime::registry()` does, and so does
/// nearly every GPU test in the workspace - makes `global()` succeed *for
/// everyone*. So on a machine with a GPU:
///
/// * a `CpuStorage` image (what `Tensor::from_vec` produces, what `Slam`
///   hands to the tracker, what every image decoder produces) reached
///   `ComputeDevice::Gpu`, and
/// * `GpuContext::match_descriptors` rejects `CpuStorage` outright, so every
///   frame failed with `Invalid input: GpuContext requires GpuStorage tensors`.
///
/// Because the context is global, that made the tracker **order-dependent**:
/// whether it saw a GPU depended on which tests happened to run first in the
/// process, which is why the same test passed under `cargo nextest` (one binary
/// per process) and failed under `cargo test` on the first run but not later
/// ones.
///
/// # Why the CPU storage selects the CPU device, rather than being uploaded
///
/// Uploading (`to_gpu`) is the other option and it is the tempting one, because
/// the group's device is presumably what the caller asked to use. It is
/// rejected here for three reasons:
///
/// 1. **The tracker needs the pixels back on the host regardless.** One line
///    after detection, `convert_to_cpu` downloads the frame anyway, to hold in
///    `TrackingFrame::image` for frame-to-frame matching. So an upload here is
///    not avoided, only reordered - and is paid on *every* frame.
/// 2. **The hot kernel is not the GPU's.** The tracker always builds both
///    descriptor tensors with `Tensor::from_vec`, i.e. on the host: the query set
///    from the detector's output, and the map set from `WorldMap`'s cached
///    CPU buffer. Those are always `CpuStorage`, so the GPU matcher can never be
///    used without uploading *both* per frame. Accelerating that is a real
///    optimisation and a real API change; it is not a decision to make silently
///    inside a storage-dispatch branch.
/// 3. **It changes behaviour for every caller.** Choosing "upload" means a
///    caller who passes CPU tensors silently gets GPU semantics (extra
///    transfers, GPU-allocator pressure, different rounding in the detector),
///    and on a machine where the GPU is contended or the upload fails partway the
///    frame fails for a reason unrelated to tracking.
///
/// The alternative, if acceleration is ever wanted: keep the device selection
/// here but add an explicit, opt-in upload of *both* descriptor tensors plus the
/// image, fall back to the CPU matcher when either upload fails, and measure that
/// GPU matching actually beats the CPU one for realistic map sizes first. That is
/// a deliberate perf change with a fallback, not a default.
///
/// GPU-resident input still takes the GPU path: the choice follows the data.
fn device_for_u8_tensor<S: Storage<u8> + 'static>(
    image: &Tensor<u8, S>,
    group: &ResourceGroup,
) -> Result<ComputeDevice<'static>, String> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;

    if image
        .storage
        .as_any()
        .downcast_ref::<GpuStorage<u8>>()
        .is_some()
    {
        return group
            .device()
            .map_err(|e| format!("Failed to get compute device: {}", e));
    }

    // CPU-resident data: the kernels below read host slices directly, so the CPU
    // device is both the correct and the cheaper choice. This is also what makes
    // the tracker independent of `GLOBAL_CONTEXT`, and therefore order
    // independent.
    if image
        .storage
        .as_any()
        .downcast_ref::<CpuStorage<u8>>()
        .is_some()
    {
        return cpu_device();
    }

    Err(format!(
        "Unsupported storage type for tracking: {}",
        image.storage.data_type_name()
    ))
}

/// The process-wide CPU compute device.
///
/// `ComputeDevice::Cpu` borrows a `&'a CpuBackend` that has to outlive the
/// returned device, so the backend is kept in a `static` for the same reason
/// `cv_hal::compute` keeps its own in a `OnceLock`.
fn cpu_device() -> Result<ComputeDevice<'static>, String> {
    use std::sync::OnceLock;
    static CPU_BACKEND: OnceLock<CpuBackend> = OnceLock::new();
    if let Some(b) = CPU_BACKEND.get() {
        return Ok(ComputeDevice::Cpu(b));
    }
    let created = CpuBackend::new().ok_or_else(|| "CPU backend unavailable".to_string())?;
    let _ = CPU_BACKEND.set(created);
    // `set` only fails if another thread won the race, in which case the
    // winner's backend is equally valid - use whichever one is installed.
    let backend = CPU_BACKEND
        .get()
        .expect("set above, or another thread set it first");
    Ok(ComputeDevice::Cpu(backend))
}

fn convert_to_cpu<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    device: &ComputeDevice,
    tensor: &Tensor<u8, S>,
) -> std::result::Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>, String> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::TensorToCpu;

    if let Some(gpu_storage) = tensor.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
        let input_gpu = Tensor {
            storage: gpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        let gpu_ctx = match device {
            ComputeDevice::Gpu(g) => g,
            _ => return Err("GpuStorage requires GPU context".into()),
        };
        input_gpu
            .to_cpu_ctx(gpu_ctx)
            .map_err(|e| format!("GPU download failed: {}", e))
    } else if let Some(cpu_storage) = tensor.storage.as_any().downcast_ref::<CpuStorage<u8>>() {
        let input_cpu = Tensor {
            storage: cpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        Ok(input_cpu.clone())
    } else {
        Err("Unsupported storage type".into())
    }
}

#[cfg(test)]
mod two_view_map_tests {
    use super::*;
    use cv_core::{Descriptor, KeyPoint};
    use nalgebra::Point3;

    fn frame(descriptors: &[(f64, f64, u8)]) -> TrackingFrame<CpuStorage<u8>> {
        let mut descs = Descriptors::new();
        for &(x, y, byte) in descriptors {
            descs.push(Descriptor::new(vec![byte; 32], KeyPoint::new(x, y)));
        }
        let keypoints = KeyPoints {
            keypoints: descs.descriptors.iter().map(|d| d.keypoint).collect(),
        };
        TrackingFrame {
            image: Tensor::from_vec(vec![0u8; 4 * 4], cv_core::TensorShape::new(1, 4, 4))
                .expect("image tensor"),
            keypoints,
            descriptors: descs,
            pose: Pose::default(),
        }
    }

    /// Every observation a point lists must be named by that keyframe's own
    /// slot for the feature. This is the two-sided link the map maintenance
    /// (fusion, culling) walks from either direction.
    fn assert_links(map: &WorldMap) {
        for (i, point) in map.points.iter().enumerate() {
            let point = point.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(point.id, i as u64, "point {i} has a stale id");
            for &(kf_id, kp_idx) in &point.observations {
                let kf = map.keyframes[kf_id as usize]
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                assert_eq!(
                    kf.map_points[kp_idx],
                    Some(i as u64),
                    "point {i} lists observation (keyframe {kf_id}, feature {kp_idx}), but \
                     that keyframe slot names {:?}",
                    kf.map_points[kp_idx]
                );
            }
        }
    }

    /// One reference feature can only be one landmark.
    ///
    /// The descriptor matcher works query by query: two current-frame
    /// descriptors whose best match is the same reference descriptor both come
    /// back as matches. `two_view_init` keeps every inlier pair, so
    /// `TwoViewInit::points` can name the same reference feature twice. The
    /// reference keyframe has one slot per feature, so inserting both puts a
    /// second landmark in the map that the keyframe cannot link: its
    /// observation points at a slot naming the other point.
    #[test]
    fn two_matches_to_one_reference_feature_make_one_landmark() {
        let reference = frame(&[(50.0, 50.0, 7), (60.0, 50.0, 9)]);
        let current = frame(&[(50.0, 50.0, 7), (51.0, 50.0, 7)]);
        let init = TwoViewInit {
            pose: Pose::identity(),
            // Two correspondences claim reference feature 0; they differ only
            // in which current-frame feature they were triangulated with.
            points: vec![
                (0, 0, Point3::new(0.0, 0.0, 5.0)),
                (0, 1, Point3::new(0.10, 0.0, 5.0)),
            ],
        };
        let mut map = WorldMap::new();
        let created = insert_two_view_map(&reference, &current, &init, &mut map);

        assert_eq!(
            map.points.len(),
            1,
            "one reference feature must produce one map point, got {} (created {created:?})",
            map.points.len()
        );
        assert_eq!(created.len(), map.points.len());
        assert_eq!(
            map.keyframes[0]
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .map_points
                .iter()
                .filter(|s| s.is_some())
                .count(),
            1,
            "the reference keyframe must link its feature to exactly one point"
        );
        assert_links(&map);
    }
}
