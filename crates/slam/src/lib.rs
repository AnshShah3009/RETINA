#![forbid(unsafe_code)]
pub mod mapping;
pub mod pose_graph;
pub mod tracking;
pub mod types;

pub use cv_video::kalman;
pub use cv_video::kalman::{
    DynamicKalmanFilter, ExtendedKalmanFilter, KalmanFilter, KalmanFilterState,
};
pub use types::{KeyFrame, MapPoint, WorldMap};

use crate::tracking::Tracker;
use cv_core::{storage::CpuStorage, CameraIntrinsics, Pose, Tensor};

pub struct Slam {
    pub map: WorldMap,
    pub tracker: Tracker,
    pub intrinsics: CameraIntrinsics,
}

impl Slam {
    pub fn new(
        group: std::sync::Arc<cv_runtime::ResourceGroup>,
        intrinsics: CameraIntrinsics,
    ) -> Self {
        Self {
            map: WorldMap::default(),
            tracker: Tracker::new(group, intrinsics),
            intrinsics,
        }
    }

    /// Track one monocular frame. See [`Tracker::process_frame`]: the first
    /// frame returns `Err` while it is held as the initialisation reference;
    /// the map is built from a later frame with enough parallax.
    pub fn process_image(
        &mut self,
        image: &image::GrayImage,
    ) -> Result<(Pose, Vec<usize>), String> {
        let shape = cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize);
        let tensor: Tensor<u8, CpuStorage<u8>> =
            Tensor::from_vec(image.to_vec(), shape).map_err(|e| e.to_string())?;

        // Front-end tracking
        let res = self.tracker.process_frame(&tensor, &mut self.map);

        // Clear pooled buffers to avoid accumulation
        cv_hal::buffer_utils::global_pool().clear();

        res
    }

    /// Track one RGB-D frame. `depth` is row-major metric depth (camera z),
    /// one value per pixel of `image`; non-finite or non-positive values are
    /// missing. The first frame bootstraps the map. See
    /// [`Tracker::process_frame_with_depth`].
    pub fn process_image_with_depth(
        &mut self,
        image: &image::GrayImage,
        depth: &[f32],
    ) -> Result<(Pose, Vec<usize>), String> {
        let (h, w) = (image.height() as usize, image.width() as usize);
        if depth.len() != h * w {
            return Err(format!(
                "depth has {} values, the {w}x{h} image needs {}",
                depth.len(),
                h * w
            ));
        }
        let shape = cv_core::TensorShape::new(1, h, w);
        let tensor: Tensor<u8, CpuStorage<u8>> =
            Tensor::from_vec(image.to_vec(), shape).map_err(|e| e.to_string())?;
        let depth: Tensor<f32, CpuStorage<f32>> =
            Tensor::from_vec(depth.to_vec(), shape).map_err(|e| e.to_string())?;

        let res = self
            .tracker
            .process_frame_with_depth(&tensor, &depth, &mut self.map);
        cv_hal::buffer_utils::global_pool().clear();
        res
    }
}
