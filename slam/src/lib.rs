pub mod mapping;
pub mod pose_graph;
pub mod tracking;

pub use cv_core::kalman;
pub use cv_core::kalman::{
    DynamicKalmanFilter, ExtendedKalmanFilter, KalmanFilter, KalmanFilterState,
};

use crate::tracking::Tracker;
use cv_core::{slam::WorldMap, storage::CpuStorage, CameraIntrinsics, Pose, Tensor};

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
}
