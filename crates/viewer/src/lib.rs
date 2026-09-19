use cv_core::point_cloud::PointCloud;

/// Point cloud logging interface.
///
/// This replaces the previous Rerun-based logger. Rerun was removed due to
/// wasm-bindgen version conflicts (rerun pins 0.2.100, eframe needs 0.2.101).
/// TODO: Re-add rerun support when version compatibility is resolved.
pub struct PointCloudLogger {
    #[allow(dead_code)]
    entity_path: String,
    point_clouds: Vec<(String, PointCloud)>,
}

impl PointCloudLogger {
    pub fn new(application_id: &str) -> Self {
        Self {
            entity_path: application_id.to_string(),
            point_clouds: Vec::new(),
        }
    }

    /// Log a point cloud for later visualization.
    pub fn log_point_cloud(
        &mut self,
        entity_path: &str,
        pc: &PointCloud,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.point_clouds
            .push((entity_path.to_string(), pc.clone()));
        Ok(())
    }

    /// Get all logged point clouds.
    pub fn logged(&self) -> &[(String, PointCloud)] {
        &self.point_clouds
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod native_viewer;
