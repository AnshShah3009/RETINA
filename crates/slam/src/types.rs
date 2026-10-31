use cv_core::{Descriptors, KeyPoints, TypedPose};
use nalgebra::Point3;
use std::sync::{Arc, RwLock};

/// A 3D point in the world map.
#[derive(Debug)]
pub struct MapPoint {
    /// Current index of this point in [`WorldMap::points`].
    ///
    /// Culling compacts the vector and rewrites `id`, and every keyframe slot
    /// that stored the old index. [`created_at`] does not change.
    pub id: u64,
    /// Birth order: how many points had been created when this one was. Not
    /// reused when points are removed, so it still identifies the initial map
    /// after culling has compacted `id`.
    pub created_at: u64,
    /// Keyframe id this point was created in (its newest observation at
    /// birth). Culling measures age from this, not from the oldest
    /// observation: a point triangulated against an early keyframe is new,
    /// and must be given later keyframes in which to be seen again.
    pub born_kf: u64,
    pub world_pos: Point3<f32>,
    pub descriptor: Vec<u8>,
    /// Indices of keyframes observing this point
    pub observations: Vec<(u64, usize)>, // (keyframe_id, keypoint_idx)
}

impl MapPoint {
    pub fn new(id: u64, pos: Point3<f32>, descriptor: Vec<u8>) -> Self {
        Self {
            id,
            created_at: id,
            born_kf: 0,
            world_pos: pos,
            descriptor,
            observations: Vec::new(),
        }
    }
}

/// A keyframe in the SLAM system.
#[derive(Debug)]
pub struct KeyFrame {
    pub id: u64,
    pub pose: TypedPose,
    pub keypoints: KeyPoints,
    pub descriptors: Descriptors,
    /// Index into [`WorldMap::points`] of the map point at each keypoint, if
    /// any. Rewritten when culling compacts the map.
    pub map_points: Vec<Option<u64>>,
}

impl KeyFrame {
    pub fn new(id: u64, pose: TypedPose, keypoints: KeyPoints, descriptors: Descriptors) -> Self {
        let num_kps = keypoints.len();
        Self {
            id,
            pose,
            keypoints,
            descriptors,
            map_points: vec![None; num_kps],
        }
    }
}

/// A shared map containing points and keyframes.
#[derive(Debug, Default)]
pub struct WorldMap {
    pub points: Vec<Arc<RwLock<MapPoint>>>,
    pub keyframes: Vec<Arc<RwLock<KeyFrame>>>,
    /// Flattened descriptors for every map point, in `points` order.
    ///
    /// Cached because the tracker needs a contiguous descriptor tensor on every
    /// frame and it was rebuilding one from the whole map each time - O(total map
    /// points) per frame, with a full `flat_map` clone of every descriptor.
    /// Cleared when a point is added or the map is compacted. An in-place
    /// descriptor edit does not change the length, so the cache is also checked
    /// against the bytes (see `mapping::descriptor_bytes`).
    pub(crate) descriptor_cache: Option<std::sync::Arc<Vec<u8>>>,
    /// Number of points the cache was built from, so a stale cache is detectable
    /// even after a push.
    pub(crate) descriptor_cache_len: usize,
    /// Monotonic birth counter. Not reduced when points are culled.
    pub(crate) next_point_id: u64,
}

impl WorldMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many map points have been created, including ones culling has since
    /// removed.
    pub fn points_created(&self) -> u64 {
        self.next_point_id
    }
}
