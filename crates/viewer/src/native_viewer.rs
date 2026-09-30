use cv_core::point_cloud::PointCloud;
use eframe::egui;
use std::sync::Arc;

/// A window that lists the point clouds it has been given.
///
/// It does not render them. There is no 3D viewport here, no wgpu pipeline, and
/// no use of `CreationContext::wgpu_render_state` - the window this opens is
/// egui's own, and the only thing drawn in the canvas area is a message saying
/// so.
///
/// An earlier version of this file claimed to be "accessing GPU" in its heading
/// and took a `CreationContext` it never read. Both overstated what the program
/// does, which is worse than a visible gap: a user would reasonably conclude a
/// rendering path existed and worked.
///
/// The 2D plotting path is real and is in `cv-plot`.
pub struct NativeViewer {
    point_clouds: Vec<Arc<PointCloud>>,
    _camera_pitch: f32,
    _camera_yaw: f32,
    _camera_dist: f32,
}

impl NativeViewer {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            point_clouds: Vec::new(),
            _camera_pitch: 0.0,
            _camera_yaw: 0.0,
            _camera_dist: 5.0,
        }
    }

    pub fn add_point_cloud(&mut self, pc: PointCloud) {
        self.point_clouds.push(Arc::new(pc));
    }

    /// Number of point clouds held. The window shows nothing, so this is the only
    /// way a caller can tell a cloud was accepted.
    pub fn cloud_count(&self) -> usize {
        self.point_clouds.len()
    }
}

impl eframe::App for NativeViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Point cloud viewer");

            ui.horizontal(|ui| {
                if ui.button("Load Mock PC").clicked() {
                    println!("Loading mock PC");
                }
            });

            if self.point_clouds.is_empty() {
                ui.label("No point clouds loaded.");
            } else {
                // Reported rather than drawn. Saying what is actually held is
                // more use than an empty canvas, and it does not imply a
                // renderer exists.
                for pc in &self.point_clouds {
                    ui.label(format!(
                        "{} points, {} normals",
                        pc.points.len(),
                        pc.normals.as_ref().map(|n| n.len()).unwrap_or(0)
                    ));
                }
                ui.separator();
                ui.label("Point clouds are listed, not rendered. See cv-plot for 2D plots.");
            }
        });
    }
}

/// Launcher function
pub fn run_native_viewer() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust CV Viewer",
        options,
        Box::new(|cc| Ok(Box::new(NativeViewer::new(cc)))),
    )
}
