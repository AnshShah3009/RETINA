use cv_core::{Descriptors, KeyPoints, TypedPose};
use nalgebra::Point3;
use std::sync::{Arc, RwLock};

/// A 3D point in the world map.
#[derive(Debug)]
pub struct MapPoint {
    pub id: u64,
    pub world_pos: Point3<f32>,
    pub descriptor: Vec<u8>,
    /// Indices of keyframes observing this point
    pub observations: Vec<(u64, usize)>, // (keyframe_id, keypoint_idx)
}

impl MapPoint {
    pub fn new(id: u64, pos: Point3<f32>, descriptor: Vec<u8>) -> Self {
        Self {
            id,
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
    /// IDs of map points observed by this keyframe
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
    /// points) per frame, with a full `flat_map` clone of every descriptor. The
    /// map only ever grows by appending, so the cache is invalidated when the
    /// length changes, which is the only way `points` can change through this
    /// API.
    pub(crate) descriptor_cache: Option<std::sync::Arc<Vec<u8>>>,
    /// Number of points the cache was built from, so a stale cache is detectable
    /// even after a push.
    pub(crate) descriptor_cache_len: usize,
}

impl WorldMap {
    pub fn new() -> Self {
        Self::default()
    }
}
