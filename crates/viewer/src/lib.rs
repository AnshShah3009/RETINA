#![forbid(unsafe_code)]
use cv_core::point_cloud::PointCloud;

/// Point cloud collection, and a window that lists what has been collected.
///
/// **It does not render anything.** `native_viewer` opens an egui window that
/// reports each cloud's point and normal count; there is no 3D viewport, no wgpu
/// pipeline, and no use of `CreationContext::wgpu_render_state`. The window says
/// so rather than implying a renderer exists.
///
/// This replaced a Rerun-based logger, which was removed over a wasm-bindgen
/// version conflict (rerun pins 0.2.100, eframe needs 0.2.101). "Replaced" is
/// the right word for the removal and the wrong one for what replaced it, since
/// nothing visual took its place.
pub struct PointCloudLogger {
    /// The application id. Kept for API compatibility; nothing reads it, and
    /// `PointCloudLogger::new` would otherwise warn.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `logged` must return what was logged, in order.
    ///
    /// This is the whole surface the crate offers today, so it is the only thing
    /// that can be wrong: a logger that silently discarded its input would look
    /// identical to one that worked, and nothing else in the workspace reads it.
    #[test]
    fn logger_returns_everything_it_was_given() {
        let mut logger = PointCloudLogger::new("app");
        assert!(logger.logged().is_empty());

        // Built by pushing onto the public `points` field: `Point3` comes from
        // nalgebra, which this crate does not depend on directly, and
        // `Default` gives the origin, which is a perfectly good test point.
        let mut a = PointCloud::default();
        a.points.push(Default::default());
        a.points.push(Default::default());
        let mut b = PointCloud::default();
        b.points.push(Default::default());

        logger.log_point_cloud("first", &a).unwrap();
        logger.log_point_cloud("second", &b).unwrap();

        let logged = logger.logged();
        assert_eq!(logged.len(), 2, "a cloud was dropped");
        assert_eq!(logged[0].0, "first", "order was not preserved");
        assert_eq!(logged[0].1.points.len(), 2, "a point was lost");
        assert_eq!(logged[1].0, "second");
        assert_eq!(logged[1].1.points.len(), 1);
    }
}
