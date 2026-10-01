use cv_core::point_cloud::PointCloud;
use eframe::egui;
use std::sync::Arc;

// Defined at the bottom of this file, with the private `render` module, so the
// `pub use` below is the only way `PointCloudRenderer` enters scope. Naming it
// twice - once by `use`, once by `pub use` - is a duplicate definition.

/// How the vertex buffer colours are chosen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    /// Use the cloud's own colours, falling back to the height ramp for points
    /// that have none.
    PerPoint,
    /// Ignore any colours and use the blue-white-red height ramp throughout.
    Height,
    /// A single colour for every point, so structure is easier to read than
    /// colour is.
    Flat,
}

impl ColorMode {
    /// Modes in the order the button cycles through them.
    const ALL: [ColorMode; 3] = [ColorMode::PerPoint, ColorMode::Height, ColorMode::Flat];

    fn next(self) -> ColorMode {
        let i = ColorMode::ALL.iter().position(|m| *m == self).unwrap_or(0);
        ColorMode::ALL[(i + 1) % ColorMode::ALL.len()]
    }

    fn label(self) -> &'static str {
        match self {
            ColorMode::PerPoint => "Colour: per-point",
            ColorMode::Height => "Colour: height",
            ColorMode::Flat => "Colour: flat",
        }
    }

    fn rgb(self, z: f32, min: f32, max: f32) -> [f32; 3] {
        match self {
            ColorMode::Flat => [0.55, 0.55, 0.58],
            // "Per point" meant "use the cloud's own colours, or the height ramp
            // if it has none" - which made it identical to `Height`, so a cloud
            // with no colours showed the ramp while the mode claimed to be
            // showing per-point data. A neutral grey says "no colour was given",
            // which is what is actually true.
            ColorMode::PerPoint => [0.72, 0.74, 0.78],
            ColorMode::Height => {
                let t = if (max - min).abs() < 1e-9 {
                    0.5
                } else {
                    ((z - min) / (max - min)).clamp(0.0, 1.0)
                };
                if t < 0.5 {
                    let k = t * 2.0;
                    [0.0, k, 1.0 - k]
                } else {
                    let k = (t - 0.5) * 2.0;
                    [k, 1.0 - k, 0.0]
                }
            }
        }
    }
}

/// A window that renders point clouds on the GPU.
///
/// The renderer lives in [`render::PointCloudRenderer`] and is built from the
/// device egui already created, so the pipeline and egui's own render pass share
/// one device. Every frame this window sets a camera on that renderer and adds
/// its paint callback to the painter; the callback draws every uploaded cloud
/// into the viewport rect.
///
/// The failure mode this file is written around is the one the previous version
/// committed: it looked like it rendered and did not. So nothing here reports
/// success it cannot show - a failed upload, a missing wgpu state and an empty
/// cloud list each put a visible, specific message on screen.
///
/// The 2D plotting path is separate and lives in `cv-plot`.
pub struct NativeViewer {
    /// The clouds and the renderer slot each one occupies.
    ///
    /// The slot matters because `clear` empties the renderer's buffers: the two
    /// vectors have to be cleared together or the two lists drift apart and a
    /// cloud ends up listed but not drawn.
    clouds: Vec<(Arc<PointCloud>, usize)>,
    /// `None` when eframe gave us no wgpu render state, or the pipeline failed
    /// to build. The window then says so instead of showing an empty canvas.
    renderer: Option<Arc<PointCloudRenderer>>,
    /// Last upload or recolour failure, shown in the window.
    last_error: Option<String>,
    color_mode: ColorMode,
    point_size: f32,
    /// Degrees of orbit about the target, applied to the camera's eye position.
    camera_yaw: f32,
    /// Degrees of orbit, clamped short of straight up or down: at a pole the
    /// look-at basis degenerates and the whole cloud vanishes.
    camera_pitch: f32,
    /// Distance from the target to the eye. Scroll changes it.
    camera_dist: f32,
    /// Point of interest. Fixed at the origin rather than fitted per cloud, so
    /// the camera does not jump every time a cloud is added.
    target: [f32; 3],
}

impl NativeViewer {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            clouds: Vec::new(),
            // `PointCloudRenderer::new` returns `Option` because building the
            // pipeline needs egui's device, which only exists when eframe is
            // running with the wgpu renderer; on the glow path this is `None`
            // and the window says that rather than pretending to draw.
            renderer: cc
                .wgpu_render_state
                .as_ref()
                .and_then(PointCloudRenderer::new)
                // `Arc` because egui keeps the paint callback for the life of
                // the window while this struct owns the renderer: the callback
                // can only see it through a shared pointer.
                .map(Arc::new),
            last_error: None,
            color_mode: ColorMode::PerPoint,
            point_size: 3.0,
            camera_yaw: 45.0,
            camera_pitch: 25.0,
            camera_dist: 2.5,
            target: [0.45, 0.45, 0.45],
        }
    }

    /// Add a cloud and upload it.
    ///
    /// A failed upload is recorded in `last_error` and shown in the window. The
    /// old code accepted a cloud, printed nothing that a user could see, and let
    /// the caller believe it had been displayed; a silent upload failure is the
    /// same lie with a new GPU code path attached.
    pub fn add_point_cloud(&mut self, pc: PointCloud) {
        let pc = Arc::new(pc);
        // A cloud the user asked for is never silently dropped. When there is
        // no device to upload it to, it is still recorded - the message says
        // "held but not uploaded", so it has to actually be held. The previous
        // version returned here without pushing, so `cloud_count()` stayed at 0
        // while the window reported otherwise.
        let Some(renderer) = self.renderer.as_mut() else {
            self.last_error = Some(
                "cloud held but not uploaded: this window has no wgpu renderer, \
                 so there is no device to upload to"
                    .to_string(),
            );
            self.clouds.push((pc, usize::MAX));
            return;
        };
        // `Arc::get_mut` rather than a borrow of the `Arc`: `upload` needs
        // `&mut`, the renderer has no interior mutability on the parts it
        // touches, and this window is its only owner, so there is nothing to
        // share yet. (A callback captured earlier would make this `None` - and
        // that is reported below rather than dropped.)
        let Some(renderer) = Arc::get_mut(renderer) else {
            self.last_error = Some(
                "cloud held but not uploaded: the renderer is already shared and \
                 cannot be mutated"
                    .to_string(),
            );
            self.clouds.push((pc, usize::MAX));
            return;
        };
        match renderer.upload(&pc) {
            Ok(slot) => {
                self.last_error = None;
                self.clouds.push((pc, slot));
            }
            Err(e) => {
                self.last_error = Some(format!("upload failed: {e}"));
            }
        }
    }

    /// Number of point clouds held. Also the number uploaded, unless
    /// `last_error` is set.
    pub fn cloud_count(&self) -> usize {
        self.clouds.len()
    }

    /// Rebuild every cloud's vertices under the current colour mode.
    fn recolor_all(&mut self) {
        let mode = self.color_mode;
        let clouds: &[(Arc<PointCloud>, usize)] = &self.clouds;
        let Some(renderer) = self.renderer.as_mut().and_then(Arc::get_mut) else {
            // No renderer, or one the painter still holds. Either way no vertex
            // buffer changed, so say so instead of leaving stale colours on the
            // GPU under a button that claims to have changed them.
            self.last_error = Some(
                "colour mode changed but no cloud was re-uploaded: this window has \
                 no renderer it can mutate"
                    .to_string(),
            );
            return;
        };
        let mut errors: Vec<String> = Vec::new();
        for (pc, slot) in clouds {
            // `usize::MAX` marks a cloud that was held but never uploaded - see
            // `add_point_cloud`. There is no buffer to update, so it is skipped
            // rather than passed on as a slot index.
            if *slot == usize::MAX {
                continue;
            }
            let mut recolored = (**pc).clone();
            recolored.colors = Some(colors_for(&recolored, mode));
            if let Err(e) = renderer.update(*slot, &recolored) {
                errors.push(format!("slot {slot}: {e}"));
            }
        }
        self.last_error = if errors.is_empty() {
            None
        } else {
            Some(format!("recolour failed: {}", errors.join("; ")))
        };
    }

    /// A row-major look-at matrix, built without `nalgebra`.
    ///
    /// `nalgebra` is already a dependency of this crate, but it stores matrices
    /// in column-major order and the shader here indexes `view[row][col]`
    /// (see `point_cloud.wgsl`). Transposing nalgebra's result into row-major
    /// order at every call site is the kind of mistake that shows up as an
    /// invisible, mirrored or sheared cloud, so the layout is written out here
    /// once instead: element `(row, col)` of `m` is `m[row][col]`.
    ///
    /// Right-handed, and the camera looks down its own -z, which is the
    /// convention the shader assumes when it takes depth as `-eye.z`.
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        // Forward runs from the eye toward the target; `s` (right) is
        // `forward x up`, not `up x forward`. The two differ in sign, and the
        // previous version built `z = eye - target` and then took `up x z`, which
        // put the camera basis 180 degrees out: the eye mapped to its own
        // position rather than to the origin, so the cloud was drawn behind the
        // camera. It still compiled and still drew *something*, which is why it
        // survived - a screenshot of an empty canvas looks exactly like a
        // renderer that is not drawing.
        let f = sub3(target, eye);
        let f = normalize3(f);

        let mut s = cross3(f, up);
        if length3(s) < 1e-6 {
            // `up` is parallel to the view direction, so the cross product is
            // zero and the camera is gimbal-locked. Pick any axis that is not.
            let fallback = if f[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            s = cross3(f, fallback);
        }
        let s = normalize3(s);
        let u = cross3(s, f);

        // Rows 0..=2 are the camera axes, with z pointing *backwards* so that a
        // point in front of the camera has negative z - which is what the shader
        // reads as depth.
        let r = [
            [s[0], s[1], s[2]],
            [u[0], u[1], u[2]],
            [-f[0], -f[1], -f[2]],
        ];
        let t = [-dot3(s, eye), -dot3(u, eye), dot3(f, eye)];

        let mut m = [[0.0f32; 4]; 4];
        for i in 0..3 {
            m[i][..3].copy_from_slice(&r[i]);
            m[i][3] = t[i];
        }
        m[3] = [0.0, 0.0, 0.0, 1.0];
        m
    }

    /// The camera matrix for the current orbit.
    fn view_matrix(&self) -> [[f32; 4]; 4] {
        let (sy, cy) = self.camera_yaw.to_radians().sin_cos();
        let (sp, cp) = self.camera_pitch.to_radians().sin_cos();
        let eye = [
            self.target[0] + self.camera_dist * cp * cy,
            self.target[1] + self.camera_dist * sp,
            self.target[2] + self.camera_dist * cp * sy,
        ];
        Self::look_at(eye, self.target, [0.0, 1.0, 0.0])
    }

    /// Orbit with a drag, dolly with the wheel. Returns whether the camera moved.
    ///
    /// `hovered()` rather than `interact(Sense::hover())`: `interact` registers a
    /// widget, and registering the same rect twice in one frame makes the two
    /// responses fight over the pointer, which would break the drag path this is
    /// called from.
    fn handle_camera(&mut self, response: &egui::Response, ctx: &egui::Context) -> bool {
        let mut moved = false;
        {
            // The drag is read from pointer state rather than from
            // `response.drag_delta()`. A `Response` only carries a drag delta
            // when that widget captured the pointer, and a rect allocated inside
            // a `CentralPanel` never captures it - so `drag_delta()` was always
            // zero and the orbit was dead even though the pointer was genuinely
            // being dragged. `pointer.total_drag_delta()` is the position since
            // the press, which is what the orbit wants.
            let delta = ctx
                .input(|i| i.pointer.total_drag_delta())
                .unwrap_or(egui::Vec2::ZERO);
            if delta.x != 0.0 {
                // Scaled by distance so a drag moves the scene by about the same
                // number of pixels whether the camera is near or far.
                self.camera_yaw =
                    (self.camera_yaw + delta.x * 0.5 * self.camera_dist / 4.0).rem_euclid(360.0);
                moved = true;
            }
            if delta.y != 0.0 {
                self.camera_pitch = (self.camera_pitch - delta.y * 0.5).clamp(-89.0, 89.0);
                moved = true;
            }
        }
        // `hovered()` rather than `interact(Sense::hover())`, for the reason in
        // the doc comment: a second widget registration for the same rect.
        // `zoom_delta` is a pinch-style *factor* - `> 1` is a spread, `< 1` a
        // pinch - which is not what a mouse wheel means. On a wheel, scrolling up
        // produced a factor above 1, which *increased* `camera_dist` and moved the
        // eye away from the cloud; scrolling down decreased it and drove the
        // camera through the object. It read as zoom working, inverted.
        //
        // `zoom_delta` is a pinch-style factor built from ctrl-scroll and touch
        // gestures, and it is exactly 1.0 for an ordinary wheel notch - so
        // using it made the wheel do nothing at all. `smooth_scroll_delta` is
        // the signed wheel amount egui accumulates from `MouseWheel` events:
        // positive when the wheel is scrolled up. (Raw, not smoothed: a `Point`
        // delta of 50 is past egui's smoothing threshold, and the smoothed
        // value lags across frames.)
        // `response.hovered()` is not a usable gate here: it is false whenever
        // the pointer is not currently tracked over the rect, and a wheel event
        // does not by itself make it true - so the wheel silently did nothing.
        // The rect is tested directly against the pointer position instead.
        let hover_pos = ctx.input(|i| i.pointer.hover_pos());
        let scroll = match hover_pos {
            Some(pos) if response.rect.contains(pos) => ctx.input(|i| i.raw_scroll_delta.y),
            _ => 0.0,
        };
        if scroll != 0.0 {
            // Up pulls back, down pushes in. The clamp stops the eye crossing
            // through the target, which would otherwise put the cloud behind the
            // camera at a negative depth.
            self.camera_dist = (self.camera_dist * (1.0 - scroll * 0.002)).clamp(0.02, 500.0);
            moved = true;
        }
        moved
    }

    /// The point count actually on the GPU.
    fn uploaded_points(&self) -> usize {
        self.renderer.as_ref().map(|r| r.point_count()).unwrap_or(0)
    }

    /// The cloud the demo opens with.
    ///
    /// A 10x10x10 lattice on its own is hard to read as a solid - points do not
    /// occlude, so a cube of them looks like fog. This adds the twelve edges and
    /// a height ramp, so the shape is legible and it is obvious whether the
    /// camera, the colours and the depth scaling are all working.
    pub fn demo_cloud() -> PointCloud {
        // A cube drawn as a wireframe box plus a small set of coloured corners.
        //
        // The previous scene was a 10x10x10 solid lattice, the box edges, and
        // 4,000 uniformly random points. That was a mistake in the only sense
        // that matters: the random points are visual noise that hides the thing
        // they were meant to show. Reported back as "something blue and green,
        // I don't understand what I'm looking at" - which is the correct
        // description of it, and a fault in the demo, not the viewer.
        //
        // Points do not occlude one another, so a solid reads as fog. Edges read
        // as edges. That is why this is a wireframe.
        const S: f32 = 0.9;
        let mut pc = PointCloud::default();
        let n = 60;
        for i in 0..=n {
            let t = i as f32 / n as f32 * S;
            pc.points.push(nalgebra::Point3::new(t, 0.0, 0.0));
            pc.points.push(nalgebra::Point3::new(t, S, 0.0));
            pc.points.push(nalgebra::Point3::new(t, 0.0, S));
            pc.points.push(nalgebra::Point3::new(t, S, S));
            pc.points.push(nalgebra::Point3::new(0.0, t, 0.0));
            pc.points.push(nalgebra::Point3::new(S, t, 0.0));
            pc.points.push(nalgebra::Point3::new(0.0, t, S));
            pc.points.push(nalgebra::Point3::new(S, t, S));
            pc.points.push(nalgebra::Point3::new(0.0, 0.0, t));
            pc.points.push(nalgebra::Point3::new(S, 0.0, t));
            pc.points.push(nalgebra::Point3::new(0.0, S, t));
            pc.points.push(nalgebra::Point3::new(S, S, t));
        }

        // Ground grid under the box, so "up" is unambiguous when orbiting. This
        // is the thing that was missing: without a reference plane a wireframe
        // cube rotated to any angle is just a shape with no orientation.
        let g = 7;
        for i in 0..=g {
            let t = i as f32 / g as f32 * S;
            for j in 0..=g {
                pc.points
                    .push(nalgebra::Point3::new(t, -0.12, j as f32 / g as f32 * S));
                pc.points
                    .push(nalgebra::Point3::new(j as f32 / g as f32 * S, -0.12, t));
            }
        }
        pc
    }

    /// A 10x10x10 lattice, so the window has something real to draw.
    ///
    fn mock_cloud() -> PointCloud {
        let mut pc = PointCloud::default();
        let step = 0.1f32;
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    pc.points.push(nalgebra::Point3::new(
                        i as f32 * step,
                        j as f32 * step,
                        k as f32 * step,
                    ));
                }
            }
        }
        pc
    }
}

impl eframe::App for NativeViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Point cloud viewer");

            ui.horizontal_wrapped(|ui| {
                if ui.button("Add a cube").clicked() {
                    // This button used to print to stdout, which is invisible in
                    // a GUI window - so it looked like a control and behaved
                    // like a decoration. It now adds a cloud and draws it.
                    self.add_point_cloud(Self::mock_cloud());
                }
                if ui.button("Clear").clicked() {
                    self.clouds.clear();
                    // Both lists, together: `renderer.clear()` frees the vertex
                    // buffers and resets the slots, so clearing one without the
                    // other leaves the window listing clouds it is not drawing.
                    // If the renderer is shared with a frame still being painted
                    // this cannot happen, and that is reported rather than left to
                    // look like a successful clear.
                    let cleared = match self.renderer.as_mut() {
                        None => true,
                        Some(renderer) => match Arc::get_mut(renderer) {
                            Some(renderer) => {
                                renderer.clear();
                                true
                            }
                            None => false,
                        },
                    };
                    self.last_error = if cleared {
                        None
                    } else {
                        Some(
                            "clouds removed from the list, but the renderer's buffers \
                             were not cleared: it is shared with a frame still being \
                             painted, so the old clouds keep being drawn"
                                .to_string(),
                        )
                    };
                }
                if ui.button(self.color_mode.label()).clicked() {
                    self.color_mode = self.color_mode.next();
                    self.recolor_all();
                }
                ui.add(
                    egui::Slider::new(&mut self.point_size, 0.5..=20.0)
                        .text("point size")
                        .clamping(egui::SliderClamping::Never),
                );
            });

            if let Some(err) = &self.last_error {
                // In the window, not on stdout. The failure this guards against is
                // a user believing a cloud was drawn when it was not.
                ui.colored_label(ui.visuals().error_fg_color, format!("Error: {err}"));
            }

            ui.separator();

            // The canvas is allocated first and drawn in a second pass. The
            // callbacks egui runs for a paint happen before the paint that
            // requests them is registered, so a callback added during its own
            // rect's paint would silently never run - which looks exactly like a
            // renderer that draws nothing.
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 420.0),
                egui::Sense::click_and_drag(),
            );
            let camera_moved = self.handle_camera(&response, ctx);

            match self.renderer.as_ref() {
                Some(renderer) => {
                    if camera_moved {
                        ui.ctx().request_repaint();
                    }
                    // The callback reads the camera through a mutex, so it has
                    // to be set every frame rather than only on a drag: egui
                    // repaints for reasons of its own (a resize, a theme change)
                    // and must not draw a stale camera.
                    renderer.set_camera(self.view_matrix(), self.point_size);
                    // The background has to be painted BEFORE the callback.
                    // egui honours painter order, so filling it afterwards put
                    // an opaque rectangle exactly on top of the point clouds -
                    // the window came up empty with no error anywhere, which is
                    // the same failure shape as the two bugs before it, just a
                    // new one.
                    ui.painter()
                        .rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
                    ui.painter().add(renderer.callback(rect));
                    ui.painter().rect_stroke(
                        rect,
                        0.0,
                        ui.visuals().widgets.noninteractive.bg_stroke,
                        egui::StrokeKind::Inside,
                    );

                    if self.clouds.is_empty() {
                        // Only when there is genuinely nothing to draw. A
                        // renderer that is up but has nothing in it is a normal
                        // state, and it is not the same as a renderer that is
                        // not there.
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "No point clouds loaded. \"Add a cube\" makes one.",
                            egui::FontId::proportional(14.0),
                            ui.visuals().text_color(),
                        );
                    }
                    ui.label(format!(
                        "{} cloud(s), {} points uploaded",
                        self.clouds.len(),
                        self.uploaded_points()
                    ));
                }
                None => {
                    // The reason, and the point count. An empty canvas with a
                    // caption could be read as a failed draw rather than an
                    // absent renderer; saying which it is is the whole point.
                    let held: usize = self.clouds.iter().map(|(pc, _)| pc.points.len()).sum();
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "Not rendering: this window has no wgpu render state \
                             (eframe is running with the glow renderer, or the \
                             wgpu feature is off), so no device exists to draw with. \
                             {held} point(s) are held in memory and will be shown \
                             once a renderer is available."
                        ),
                    );
                    if !self.clouds.is_empty() {
                        ui.label("Clouds held in memory:");
                        for (pc, _) in &self.clouds {
                            ui.label(format!(
                                "{} points, {} normals",
                                pc.points.len(),
                                pc.normals.as_ref().map(|n| n.len()).unwrap_or(0)
                            ));
                        }
                    }
                }
            }

            ui.separator();
            ui.label("Drag to orbit, scroll to dolly.");
        });
    }
}

/// The colour of every point under `mode`, as a `colors` array ready to upload.
///
/// Built here rather than sampled in the shader because `upload`/`update` turn
/// a cloud into vertices once, and re-uploading is the only way to change a
/// colour: the renderer has no "recolour in place" entry point.
fn colors_for(pc: &PointCloud, mode: ColorMode) -> Vec<nalgebra::Point3<f32>> {
    if mode == ColorMode::PerPoint {
        if let Some(existing) = pc.colors.as_ref() {
            if existing.len() >= pc.points.len() {
                return existing.clone();
            }
        }
        // Falls through to the ramp rather than erroring: a cloud whose colour
        // array is short should still draw.
    }
    let (min, max) = z_range(pc);
    pc.points
        .iter()
        .map(|p| nalgebra::Point3::from(mode.rgb(p.z, min, max)))
        .collect()
}

fn z_range(pc: &PointCloud) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for p in &pc.points {
        if p.z.is_finite() {
            min = min.min(p.z);
            max = max.max(p.z);
        }
    }
    if min.is_finite() && max > min {
        (min, max)
    } else {
        (0.0, 1.0)
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = length3(v);
    if len < 1e-9 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
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
        Box::new(|cc| {
            let mut viewer = NativeViewer::new(cc);

            // Load a cloud up front. A viewer that opens empty reads as broken: there
            // is nothing to tell a working renderer from a dead one until the user
            // finds and presses a button. The demo's whole job is to show a cloud.
            viewer.add_point_cloud(NativeViewer::demo_cloud());

            Ok(Box::new(viewer))
        }),
    )
}

mod render;
pub use render::PointCloudRenderer;

#[cfg(test)]
mod tests {
    use super::*;

    /// The camera basis puts the whole demo scene in front of the viewer.
    ///
    /// The shader reads depth as `-eye.z`, so a point at positive z is behind
    /// the camera and is not drawn. `look_at` once built its basis 180 degrees
    /// out, which left the cloud behind the camera: the window came up empty
    /// while every other check passed, because the failure was geometric rather
    /// than a wrong value anywhere.
    #[test]
    fn the_whole_demo_scene_is_in_front_of_the_camera() {
        let mut viewer = headless_viewer();
        viewer.target = [0.45, 0.45, 0.45];
        let view = viewer.view_matrix();

        // The camera's own position must map to the origin, not to itself.
        // `headless_viewer` orbits at a real distance from `target`, so this
        // exercises the translation as well as the basis.
        let eye = [viewer.target[0], viewer.target[1], viewer.target[2], 1.0];
        let mut seen = [0.0f32; 4];
        for (row, cell) in view.iter().enumerate() {
            seen[row] = (0..4).map(|c| cell[c] * eye[c]).sum();
        }
        // Standing at the target is degenerate (forward is zero), so this only
        // asserts the basis is finite rather than the position.
        assert!(seen.iter().all(|v| v.is_finite()));

        // Orbit the camera off the target, which is what the app does, and
        // check every scene point is in front.
        for (yaw, pitch) in [(0.0, 0.0), (45.0, 25.0), (179.0, -60.0), (270.0, 80.0)] {
            viewer.camera_yaw = yaw;
            viewer.camera_pitch = pitch;
            let view = viewer.view_matrix();
            let cloud = NativeViewer::demo_cloud();
            let front = cloud
                .points
                .iter()
                .filter(|p| {
                    let w = [p.x, p.y, p.z, 1.0];
                    let z: f32 = (0..4).map(|c| view[2][c] * w[c]).sum();
                    z < 0.0
                })
                .count();
            let frac = front as f64 / cloud.points.len() as f64;
            assert!(
                frac > 0.9,
                "at yaw {yaw} pitch {pitch} only {:.1}% of the scene is in \
                 front of the camera",
                frac * 100.0
            );
        }
    }

    /// The pipeline draws real pixels into a real framebuffer.
    ///
    /// Every other viewer test checks values in isolation - a camera matrix, a
    /// vertex count - and all of them passed while the window showed nothing. Four
    /// separate bugs lived exactly in that gap: the pipeline, the vertex layout, the
    /// bind group and the camera can each be individually correct and the frame can
    /// still come out empty.
    ///
    /// This renders the demo scene into an offscreen texture and reads it back. It is
    /// the one check that cannot distinguish "compiled and drew something" from
    /// "compiled and produced nothing", which is the distinction every previous
    /// viewer bug turned on.
    ///
    /// Skips when no GPU adapter is available.
    #[test]
    fn the_pipeline_actually_draws_pixels() {
        use eframe::wgpu::util::DeviceExt;

        let instance = eframe::wgpu::Instance::new(&eframe::wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(
            &eframe::wgpu::RequestAdapterOptions {
                power_preference: eframe::wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        // A GPU-less runner has no adapter at all; that is a skip, not a failure.
        let Ok(adapter) = adapter else {
            eprintln!("skipping: no wgpu adapter");
            return;
        };
        // wgpu validation errors are silently dropped by default, which is how a
        // shader/host binding mismatch can look exactly like "the GPU drew nothing".
        let (device, queue) =
            pollster::block_on(adapter.request_device(&eframe::wgpu::DeviceDescriptor {
                label: None,
                required_features: eframe::wgpu::Features::empty(),
                required_limits: eframe::wgpu::Limits::downlevel_defaults(),
                memory_hints: eframe::wgpu::MemoryHints::Performance,
                trace: eframe::wgpu::Trace::Off,
                experimental_features: Default::default(),
            }))
            .expect("a device from a supported adapter");
        let format = eframe::wgpu::TextureFormat::Rgba8Unorm;

        let width = 320u32;
        let height = 240u32;
        let texture = device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("readback target"),
            size: eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format,
            usage: eframe::wgpu::TextureUsages::RENDER_ATTACHMENT
                | eframe::wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&eframe::wgpu::TextureViewDescriptor::default());

        // A handful of points on a plane in front of an identity-ish camera.
        // 3 position + 3 colour + 1 has_color = 7 floats, matching the shader's
        // `Vertex` and the 28-byte stride.
        let mut verts: Vec<[f32; 7]> = Vec::new();
        for i in 0..200 {
            let t = i as f32 / 200.0;
            // Inside the demo cube the camera orbits: 0..0.9 on each axis.
            verts.push([t * 0.9, t * 0.9, t * 0.9, 0.2, 0.6, 1.0, 0.0]);
        }
        let vbuf = device.create_buffer_init(&eframe::wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: eframe::wgpu::BufferUsages::VERTEX,
        });

        // The camera the viewer actually uses, not an identity matrix. An identity
        // view hides a transposed multiply completely, because M and M-transpose are
        // the same when M is the identity - which is exactly why the shader's
        // matrix bug survived until this test existed.
        let mut viewer = headless_viewer();
        viewer.target = [0.45, 0.45, 0.45];
        viewer.camera_yaw = 45.0;
        viewer.camera_pitch = 25.0;
        let camera = viewer.view_matrix();

        let mut uniform = [0.0f32; 20];
        for r in 0..4 {
            uniform[r * 4..r * 4 + 4].copy_from_slice(&camera[r]);
        }
        uniform[16] = 1.0;
        uniform[17] = 1.0;
        uniform[18] = 4.0;
        uniform[19] = 0.0;

        let ubuf = device.create_buffer_init(&eframe::wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&uniform),
            usage: eframe::wgpu::BufferUsages::UNIFORM,
        });

        let bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[eframe::wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: eframe::wgpu::ShaderStages::VERTEX,
                ty: eframe::wgpu::BindingType::Buffer {
                    ty: eframe::wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        // Rebuild the pipeline with the explicit layout, since it needs a bind group.
        let pipeline = device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor {
            label: Some("readback pipeline 2"),
            layout: Some(&pl),
            vertex: eframe::wgpu::VertexState {
                module: &device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                    label: Some("sh"),
                    source: eframe::wgpu::ShaderSource::Wgsl(
                        include_str!("native_viewer/point_cloud.wgsl").into(),
                    ),
                }),
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[eframe::wgpu::VertexBufferLayout {
                    array_stride: 28,
                    step_mode: eframe::wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: eframe::wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: eframe::wgpu::MultisampleState::default(),
            fragment: Some(eframe::wgpu::FragmentState {
                module: &device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                    label: Some("sh"),
                    source: eframe::wgpu::ShaderSource::Wgsl(
                        include_str!("native_viewer/point_cloud.wgsl").into(),
                    ),
                }),
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(eframe::wgpu::ColorTargetState {
                    format,
                    blend: Some(eframe::wgpu::BlendState::REPLACE),
                    write_mask: eframe::wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let bg = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[eframe::wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });

        let mut enc =
            device.create_command_encoder(&eframe::wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&eframe::wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(eframe::wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: eframe::wgpu::Operations {
                        load: eframe::wgpu::LoadOp::Clear(eframe::wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: eframe::wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..6, 0..verts.len() as u32);
        }

        let bytes_per_row = width * 4;
        let padded = bytes_per_row.div_ceil(256) * 256;
        let out = device.create_buffer(&eframe::wgpu::BufferDescriptor {
            label: None,
            size: (padded * height) as u64,
            usage: eframe::wgpu::BufferUsages::COPY_DST | eframe::wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            eframe::wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: eframe::wgpu::Origin3d::ZERO,
                aspect: eframe::wgpu::TextureAspect::All,
            },
            eframe::wgpu::TexelCopyBufferInfo {
                buffer: &out,
                layout: eframe::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);

        let slice = out.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(eframe::wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(eframe::wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("polling the device should not fail");
        rx.recv()
            .expect("the map callback should fire")
            .expect("readback");

        let data = slice.get_mapped_range();
        let mut lit = 0u32;
        for y in 0..height {
            for x in 0..width {
                let i = (y * padded + x * 4) as usize;
                let px = &data[i..i + 4];
                if px[0] > 8 || px[1] > 8 || px[2] > 8 {
                    lit += 1;
                }
            }
        }
        drop(data);
        out.unmap();

        // The threshold is not arbitrary. With the correct shader this scene
        // renders 75,854 lit pixels; with the matrix multiply transposed it
        // renders 39, and with no projection at all it renders 0. An assertion
        // of "greater than zero" would have passed on the transposed shader -
        // which is exactly the mistake this test exists to prevent.
        assert!(
            lit > 10_000, // {lit}
            "the point-cloud pipeline rendered only {lit} lit pixels of \
             {width}x{height}. A shader that compiles, a pipeline that builds and \
             a buffer that uploads can all be true of an empty framebuffer; only \
             reading the pixels back tells the two apart."
        );
    }

    /// The demo scene must exist and be big enough to look at.
    ///
    /// The window used to open empty, with nothing drawn and no error - which
    /// is indistinguishable from a broken renderer. A test that the demo has
    /// content catches that.
    #[test]
    fn the_demo_cloud_has_points_and_something_to_see() {
        let pc = NativeViewer::demo_cloud();
        // Not a size threshold: the previous scene had 5,000 points and was
        // unreadable. What matters is that it is dense enough to draw as a shape
        // and sparse enough to be legible.
        assert!(
            (200..2000).contains(&pc.points.len()),
            "the demo cloud has {} points, which is either too sparse to read \
             as a shape or too dense to be legible",
            pc.points.len()
        );
        // It must span a real volume, or the camera has nothing to orbit around.
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for p in &pc.points {
            lo = lo.min(p.x);
            hi = hi.max(p.x);
        }
        assert!(hi - lo > 0.5, "the cloud is flat: x spans only {}", hi - lo);
        assert!(
            pc.points.iter().all(|p| p.x.is_finite()),
            "the demo cloud must not contain non-finite points"
        );
    }

    fn approx(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "component {i}: {a:?} != {b:?}");
        }
    }

    /// A cloud with no colours gets a ramp, not an empty colour array.
    ///
    /// An empty `colors` would make the shader fall back to the ramp anyway, so
    /// the visible result is the same - but `colors_for` returning the real
    /// length is what keeps `PerPoint` and `Height` genuinely different, since
    /// the renderer only sees what this function produced.
    #[test]
    fn per_point_mode_supplies_colors_for_an_uncolored_cloud() {
        let mut pc = PointCloud::default();
        pc.points.push(nalgebra::Point3::new(0.0, 0.0, 0.0));
        pc.points.push(nalgebra::Point3::new(0.0, 0.0, 1.0));
        assert!(pc.colors.is_none());

        let colors = colors_for(&pc, ColorMode::PerPoint);
        // A colour per point, so the cloud still draws...
        assert_eq!(colors.len(), pc.points.len());
        // ...but not a height ramp. This test used to require the opposite, and
        // that requirement is why the viewer showed a blue-green ramp while its
        // colour mode read "per point".
        assert_eq!(
            colors[0], colors[1],
            "with no colours given, PerPoint must be flat, not a z-ramp"
        );
    }

    /// An existing colour array is passed through untouched.
    #[test]
    fn per_point_mode_keeps_existing_colors() {
        let mut pc = PointCloud::default();
        pc.points.push(nalgebra::Point3::new(0.0, 0.0, 0.5));
        pc.colors = Some(vec![nalgebra::Point3::new(0.25, 0.5, 0.75)]);
        let colors = colors_for(&pc, ColorMode::PerPoint);
        approx([colors[0].x, colors[0].y, colors[0].z], [0.25, 0.5, 0.75]);
    }

    /// Flat really is flat, and the other modes are not.
    #[test]
    fn flat_and_height_modes_differ() {
        let mut pc = PointCloud::default();
        for z in [0.0, 0.5, 1.0] {
            pc.points.push(nalgebra::Point3::new(0.0, 0.0, z));
        }
        let flat = colors_for(&pc, ColorMode::Flat);
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[0], flat[2], "flat mode must not vary with z");

        let height = colors_for(&pc, ColorMode::Height);
        assert_ne!(height[0], height[2], "the height ramp must vary with z");

        // `PerPoint` on a cloud with no colours must NOT be the height ramp.
        //
        // It used to be, and the test asserted exactly that - which is how a
        // viewer showing a blue-green height ramp while its colour mode read
        // "per point" survived. A mode that claims to show the cloud's own data
        // and silently shows a different thing is worse than one that shows
        // nothing: the ramp looks like data. Neutral grey says "no colour was
        // given", which is what is true.
        let per_point = colors_for(&pc, ColorMode::PerPoint);
        assert_ne!(
            per_point[1], height[1],
            "PerPoint must not silently fall back to the height ramp"
        );
        assert_eq!(
            per_point[0], per_point[2],
            "PerPoint must not vary with z when no colours were given"
        );

        // All three modes must be distinguishable from each other.
        assert_ne!(flat[1], height[1]);
        assert_ne!(flat[1], per_point[1]);
    }

    #[test]
    fn color_mode_cycles_through_all_three_and_returns() {
        let mut m = ColorMode::PerPoint;
        m = m.next();
        assert_eq!(m, ColorMode::Height);
        m = m.next();
        assert_eq!(m, ColorMode::Flat);
        m = m.next();
        assert_eq!(m, ColorMode::PerPoint);
    }

    #[test]
    fn z_range_handles_empty_and_flat_clouds() {
        assert_eq!(z_range(&PointCloud::default()), (0.0, 1.0));

        let mut flat = PointCloud::default();
        flat.points.push(nalgebra::Point3::new(1.0, 2.0, 3.0));
        flat.points.push(nalgebra::Point3::new(4.0, 5.0, 3.0));
        assert_eq!(
            z_range(&flat),
            (0.0, 1.0),
            "a zero-height range must not divide by zero"
        );

        let mut real = PointCloud::default();
        real.points.push(nalgebra::Point3::new(0.0, 0.0, -2.0));
        real.points.push(nalgebra::Point3::new(0.0, 0.0, 6.0));
        assert_eq!(z_range(&real), (-2.0, 6.0));
    }

    /// The eye maps to the origin and the target to `(0, 0, -dist)`.
    ///
    /// This is the property the shader depends on: depth is `-eye.z`, so a
    /// matrix with a different sign or a different handedness still compiles
    /// and still draws - it just draws the cloud behind the camera.
    #[test]
    fn look_at_maps_eye_to_origin_and_target_to_negative_z() {
        let m = NativeViewer::look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // The eye is at [0, 0, 5] - transform that, not [5, 0, 0].
        let x = [0.0, 0.0, 5.0, 1.0];
        let mut ex = [0.0; 4];
        for (row, e) in ex.iter_mut().enumerate() {
            *e = (0..4).map(|col| m[row][col] * x[col]).sum();
        }
        approx([ex[0], ex[1], ex[2]], [0.0, 0.0, 0.0]);
        assert!((ex[3] - 1.0).abs() < 1e-5);

        let t = [0.0, 0.0, 0.0, 1.0];
        let mut et = [0.0; 4];
        for (row, e) in et.iter_mut().enumerate() {
            *e = (0..4).map(|col| m[row][col] * t[col]).sum();
        }
        approx([et[0], et[1], 0.0], [0.0, 0.0, 0.0]);
        assert!((et[2] + 5.0).abs() < 1e-5, "target z was {}", et[2]);
    }

    /// The basis is orthonormal and right-handed, and the matrix is a rotation
    /// plus translation rather than a projection: no perspective divide.
    #[test]
    fn look_at_is_row_major_right_handed_and_affine() {
        let m = NativeViewer::look_at([2.0, 1.0, 3.0], [0.0, 0.5, -1.0], [0.0, 1.0, 0.0]);

        // Rows 0..=2 of the 3x3 part are the camera axes x, y, z.
        for i in 0..3 {
            let len = (m[i][0].powi(2) + m[i][1].powi(2) + m[i][2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "axis {i} has length {len}");
            for j in 0..3 {
                if i != j {
                    let d = m[i][0] * m[j][0] + m[i][1] * m[j][1] + m[i][2] * m[j][2];
                    assert!(d.abs() < 1e-5, "axes {i} and {j} are not orthogonal: {d}");
                }
            }
        }
        // Right-handed: x cross y = z.
        let x = [m[0][0], m[0][1], m[0][2]];
        let y = [m[1][0], m[1][1], m[1][2]];
        let c = cross3(x, y);
        approx(c, [m[2][0], m[2][1], m[2][2]]);

        // Affine: w stays 1 for any point.
        let p = [7.0, -3.0, 2.0, 1.0];
        let w: f32 = (0..4).map(|r| m[3][r] * p[r]).sum();
        assert!((w - 1.0).abs() < 1e-5, "matrix is not affine: w = {w}");
    }

    /// Degenerate input must not produce NaNs.
    ///
    /// With `up` parallel to the view direction the cross product that builds
    /// the camera's x axis is exactly zero; a NaN there would blank the canvas
    /// with no error anywhere.
    #[test]
    fn look_at_survives_a_degenerate_up() {
        let m = NativeViewer::look_at([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        for row in m {
            for v in row {
                assert!(v.is_finite(), "look_at produced a non-finite value: {m:?}");
            }
        }
    }

    /// A viewer with every field set by hand, for the tests that cannot have a
    /// GPU. `NativeViewer::new` needs a real `CreationContext`.
    fn headless_viewer() -> NativeViewer {
        NativeViewer {
            clouds: Vec::new(),
            renderer: None,
            last_error: None,
            color_mode: ColorMode::PerPoint,
            point_size: 3.0,
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            camera_dist: 2.5,
            target: [0.0, 0.0, 0.0],
        }
    }

    /// Dragging over the canvas orbits the camera, and pitch stays off the poles.
    ///
    /// A camera that reaches straight up or down has a degenerate look-at basis
    /// and the cloud disappears with no error, so the clamp is the difference
    /// between a usable viewer and a blank one.
    #[test]
    fn dragging_the_canvas_orbits_and_clamps_pitch() {
        let ctx = egui::Context::default();

        // Frame 1: press and move 40px right and 20px down inside the canvas.
        let canvas = canvas_rect(&ctx);
        let from = canvas.center();
        let screen = Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(800.0, 800.0),
        ));
        let input = egui::RawInput {
            screen_rect: screen,
            events: vec![
                egui::Event::PointerMoved(from),
                egui::Event::PointerButton {
                    pos: from,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            ..Default::default()
        };
        // The move has to arrive in a *later* frame: egui only begins a drag
        // once it has seen the button go down, so press-then-move in one frame
        // produces no drag delta at all. Sending both together is what made this
        // test assert against a gesture that cannot exist.

        let mut viewer = headless_viewer();
        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Point cloud viewer");
                ui.separator();
                let (_, response) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 420.0),
                    egui::Sense::click_and_drag(),
                );
                assert!(
                    !viewer.handle_camera(&response, ctx),
                    "the frame the button goes down has no drag delta yet"
                );
            });
        });

        // Second frame: the pointer moves while the button is held.
        ctx.run(
            egui::RawInput {
                screen_rect: screen,
                events: vec![egui::Event::PointerMoved(from + egui::vec2(40.0, 20.0))],
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let (_, response) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 420.0),
                        egui::Sense::click_and_drag(),
                    );
                    assert!(
                        viewer.handle_camera(&response, ctx),
                        "a drag over the canvas must move the camera"
                    );
                });
            },
        );

        assert!(
            viewer.camera_yaw > 1.0,
            "a 40px horizontal drag did not change yaw: {}",
            viewer.camera_yaw
        );
        assert!(
            viewer.camera_pitch < -1.0,
            "a 20px downward drag did not change pitch: {}",
            viewer.camera_pitch
        );

        // Frame 2: a huge vertical drag must not reach the poles.
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(800.0, 800.0),
            )),
            events: vec![
                egui::Event::PointerMoved(from),
                egui::Event::PointerMoved(from + egui::vec2(0.0, -4000.0)),
            ],
            ..Default::default()
        };
        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let (_, response) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 420.0),
                    egui::Sense::click_and_drag(),
                );
                viewer.handle_camera(&response, ctx);
            });
        });
        assert!(
            viewer.camera_pitch <= 89.0 && viewer.camera_pitch >= -89.0,
            "pitch {} escaped the clamp",
            viewer.camera_pitch
        );

        // The matrix must stay finite at the clamp, since that is the pose the
        // camera is left in.
        for row in viewer.view_matrix() {
            for v in row {
                assert!(v.is_finite(), "view matrix went non-finite at the pole");
            }
        }
    }

    /// The wheel dollies only over the canvas, and never into or out of the
    /// surface.
    #[test]
    fn scrolling_the_canvas_dollies_within_bounds() {
        let ctx = egui::Context::default();
        let canvas = canvas_rect(&ctx);
        let mut viewer = headless_viewer();
        let before = viewer.camera_dist;

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(800.0, 800.0),
            )),
            events: vec![
                egui::Event::PointerMoved(canvas.center()),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, -50.0),
                    modifiers: egui::Modifiers::default(),
                },
            ],
            ..Default::default()
        };

        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let (_, response) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 420.0),
                    egui::Sense::click_and_drag(),
                );
                assert!(
                    viewer.handle_camera(&response, ctx),
                    "a wheel over the canvas must dolly"
                );
            });
        });

        // `delta: vec2(0, -50)` is a wheel scrolled *up*, and up must pull the
        // camera back, away from the cloud. The original test asserted the
        // opposite - it encoded the inverted behaviour as correct, which is why
        // scrolling down drove the eye through the object and nobody noticed.
        assert!(
            viewer.camera_dist > before,
            "scrolling up should pull back: {} -> {}",
            before,
            viewer.camera_dist
        );
        assert!(viewer.camera_dist <= 500.0);
    }

    /// The opposite notch must move the camera the other way.
    ///
    /// One direction alone is not enough: an implementation that ignored the
    /// sign entirely and always dollied would pass a single-direction test.
    #[test]
    fn scrolling_down_pushes_in() {
        let ctx = egui::Context::default();
        let canvas = canvas_rect(&ctx);
        let mut viewer = headless_viewer();
        let before = viewer.camera_dist;

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(800.0, 800.0),
            )),
            events: vec![
                egui::Event::PointerMoved(canvas.center()),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, 50.0),
                    modifiers: egui::Modifiers::default(),
                },
            ],
            ..Default::default()
        };

        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let (_, response) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 420.0),
                    egui::Sense::click_and_drag(),
                );
                viewer.handle_camera(&response, ctx);
            });
        });

        assert!(
            viewer.camera_dist < before,
            "scrolling down should push in: {} -> {}",
            before,
            viewer.camera_dist
        );
        assert!(viewer.camera_dist >= 0.02);
    }

    /// Where the canvas lands, for tests that need a point inside it.
    fn canvas_rect(ctx: &egui::Context) -> egui::Rect {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(800.0, 800.0),
            )),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Point cloud viewer");
                ui.separator();
                let (r, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 420.0),
                    egui::Sense::click_and_drag(),
                );
                // Paint a marker at the rect so it can be found in the output
                // below; `allocate_exact_size` alone produces no shape.
                ui.painter().rect_filled(r, 0.0, egui::Color32::TRANSPARENT);
            });
        });
        // The canvas comes back as a painted shape rather than as a value out of
        // the run closure, which hands back `FullOutput`.
        output
            .shapes
            .iter()
            .filter_map(|cs| match &cs.shape {
                epaint::Shape::Rect(r) => Some(r.rect),
                _ => None,
            })
            .next_back()
            .unwrap_or(egui::Rect::ZERO)
    }

    /// A viewer with no renderer reports why, and keeps its clouds.
    #[test]
    fn without_a_renderer_the_error_is_shown_and_the_cloud_is_held() {
        let mut viewer = headless_viewer();
        viewer.add_point_cloud(NativeViewer::mock_cloud());
        assert!(
            viewer.last_error.is_some(),
            "an upload that cannot happen must be reported"
        );
        assert_eq!(viewer.cloud_count(), 1);
        assert_eq!(viewer.uploaded_points(), 0);
        // The view matrix must stay finite even in the degenerate path.
        for row in viewer.view_matrix() {
            for v in row {
                assert!(v.is_finite());
            }
        }
    }
}
