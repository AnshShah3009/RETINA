use crate::{Descriptors, KeyPoints, TypedPose};
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
}

impl WorldMap {
    pub fn new() -> Self {
        Self::default()
    }
}
