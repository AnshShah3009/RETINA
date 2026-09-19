use cv_core::point_cloud::PointCloud;
use eframe::egui;
use std::sync::Arc;

pub struct NativeViewer {
    // Scene data
    point_clouds: Vec<Arc<PointCloud>>,

    // Camera state (orbit) - planned for future use
    _camera_pitch: f32,
    _camera_yaw: f32,
    _camera_dist: f32,
}

impl NativeViewer {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Here we could access cc.wgpu_render_state to init resources
        // For now, simple init
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
}

impl eframe::App for NativeViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Native Viewer (Accessing GPU...)");

            ui.horizontal(|ui| {
                if ui.button("Load Mock PC").clicked() {
                    println!("Loading mock PC");
                }
            });

            // Placeholder for 3D Viewport
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (rect, _response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                // Custom wgpu painting would go here via PaintCallback
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "3D Rendering Not Yet Implemented",
                    egui::FontId::proportional(20.0),
                    egui::Color32::WHITE,
                );
            });
        });
    }
}

/// Launcher function
pub fn run_native_viewer() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        // viewport: eframe::egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust CV Viewer",
        options,
        Box::new(|cc| Ok(Box::new(NativeViewer::new(cc)))),
    )
}
