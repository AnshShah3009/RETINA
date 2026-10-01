//! A minimal wgpu point-cloud renderer for the viewer window.
//!
//! Point clouds are bags of independent samples, so this draws one screen-space
//! quad per point rather than meshing anything: no adjacency, no indices, no
//! lighting. What it does do is size each quad by depth and discard the corners
//! of the sprite, so a dense cloud reads as a cloud instead of a solid block.
//!
//! Integration is through `egui_wgpu::Callback`, which is how egui 0.33 expects
//! custom wgpu drawing to be done. The detail that matters is that the
//! callback's `viewport` rect is *already* in normalised device coordinates,
//! `[-1, +1]`, so no projection is needed here - only the view transform and
//! the egui-provided mapping into that space.

use cv_core::point_cloud::PointCloud;
use eframe::wgpu::util::DeviceExt;
use std::sync::Arc;

/// Vertex data uploaded per point.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    /// Normalised RGB.
    color: [f32; 3],
    /// 1 when `color` came from the cloud, 0 when it came from the height ramp.
    has_color: f32,
}

/// Uniforms: the view transform and the point radius in NDC.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    /// Row-major view matrix.
    view: [[f32; 4]; 4],
    /// The drawable area in **pixels**, `(width, height)`.
    ///
    /// This was `viewport` and carried `(1, 1)` - the NDC half-extent - while
    /// the shader used it as a pixel count. A point radius of 3 therefore became
    /// `3 / depth` in NDC units, which at depth 2.5 is 1.2: a sprite 192 px in
    /// radius on a 320 px canvas, filling the frame with one disc. Renaming it
    /// to `pixel_size` is the fix as much as the value is; the two units are not
    /// interchangeable and the name said NDC while the use was pixels.
    pixel_size: [f32; 2],
    point_radius: f32,
    _pad: f32,
}

const SHADER: &str = include_str!("point_cloud.wgsl");

struct CloudBuffer {
    buffer: eframe::wgpu::Buffer,
    len: usize,
}

/// Owns the wgpu pipeline and the per-cloud vertex buffers.
pub struct PointCloudRenderer {
    device: eframe::wgpu::Device,
    queue: eframe::wgpu::Queue,
    pipeline: eframe::wgpu::RenderPipeline,
    format: eframe::wgpu::TextureFormat,
    uniform_buffer: eframe::wgpu::Buffer,
    bind_group_layout: eframe::wgpu::BindGroupLayout,
    clouds: Vec<CloudBuffer>,
    /// Shared with the egui callback, which needs `Send + Sync`.
    view: std::sync::Arc<std::sync::Mutex<[[f32; 4]; 4]>>,
    point_radius: std::sync::Arc<std::sync::Mutex<f32>>,
}

impl PointCloudRenderer {
    /// Build the pipeline from egui's render state.
    ///
    /// Takes the device egui already created rather than requesting a second
    /// one: the callback's render pass targets egui's own textures, and two
    /// devices on the same adapter cannot share them.
    pub fn new(state: &egui_wgpu::RenderState) -> Option<Self> {
        let device = state.device.clone();
        let queue = state.queue.clone();
        let format = state.target_format;

        let shader = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Shader"),
            source: eframe::wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                label: Some("Point Cloud BGL"),
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

        let layout = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor {
            label: Some("Point Cloud PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor {
            label: Some("Point Cloud Pipeline"),
            layout: Some(&layout),
            vertex: eframe::wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[eframe::wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: eframe::wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32x3,
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 1,
                        },
                        eframe::wgpu::VertexAttribute {
                            format: eframe::wgpu::VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 6]>() as u64,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: eframe::wgpu::PrimitiveState {
                topology: eframe::wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(eframe::wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                // Alpha blending, so a sparse cloud needs no depth sorting to
                // look right.
                targets: &[Some(eframe::wgpu::ColorTargetState {
                    format,
                    blend: Some(eframe::wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: eframe::wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&eframe::wgpu::BufferDescriptor {
            label: Some("Point Cloud Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            format,
            uniform_buffer,
            bind_group_layout,
            clouds: Vec::new(),
            view: std::sync::Arc::new(std::sync::Mutex::new(identity())),
            point_radius: std::sync::Arc::new(std::sync::Mutex::new(3.0)),
        })
    }

    /// Set the camera and point radius used by the next frame.
    pub fn set_camera(&self, view: [[f32; 4]; 4], point_radius: f32) {
        if let Ok(mut v) = self.view.lock() {
            *v = view;
        }
        if let Ok(mut r) = self.point_radius.lock() {
            *r = point_radius;
        }
    }

    /// Append `pc`'s vertices, returning the slot used to update or drop it.
    pub fn upload(&mut self, pc: &PointCloud) -> Result<usize, String> {
        let (min_z, max_z) = z_range(pc);
        let verts = build_vertices(pc, min_z, max_z);
        let buffer = self
            .device
            .create_buffer_init(&eframe::wgpu::util::BufferInitDescriptor {
                label: Some("Point Cloud Vertices"),
                contents: bytemuck::cast_slice(&verts),
                usage: eframe::wgpu::BufferUsages::VERTEX,
            });
        self.clouds.push(CloudBuffer {
            len: verts.len(),
            buffer,
        });
        Ok(self.clouds.len() - 1)
    }

    /// Replace the vertices of an existing slot.
    pub fn update(&mut self, slot: usize, pc: &PointCloud) -> Result<(), String> {
        let Some(existing) = self.clouds.get_mut(slot) else {
            return Err(format!("no cloud in slot {slot}"));
        };
        let (min_z, max_z) = z_range(pc);
        let verts = build_vertices(pc, min_z, max_z);
        existing.buffer =
            self.device
                .create_buffer_init(&eframe::wgpu::util::BufferInitDescriptor {
                    label: Some("Point Cloud Vertices"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: eframe::wgpu::BufferUsages::VERTEX,
                });
        existing.len = verts.len();
        Ok(())
    }

    pub fn clear(&mut self) {
        self.clouds.clear();
    }

    pub fn cloud_count(&self) -> usize {
        self.clouds.len()
    }

    pub fn point_count(&self) -> usize {
        self.clouds.iter().map(|c| c.len).sum()
    }

    /// An `egui::PaintCallback` that draws every uploaded cloud into `rect`.
    pub fn callback(self: &Arc<Self>, rect: egui::Rect) -> egui::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(rect, CloudCallback(Arc::clone(self)))
    }

    fn draw_into(&self, pass: &mut eframe::wgpu::RenderPass<'static>, pixel_size: [f32; 2]) {
        let view = self.view.lock().map(|v| *v).unwrap_or_else(|_| identity());
        let radius = self.point_radius.lock().map(|r| *r).unwrap_or(3.0);
        let uniforms = Uniforms {
            view,
            pixel_size: [pixel_size[0].max(1.0), pixel_size[1].max(1.0)],
            point_radius: radius,
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self
            .device
            .create_bind_group(&eframe::wgpu::BindGroupDescriptor {
                label: Some("Point Cloud BG"),
                layout: &self.bind_group_layout,
                entries: &[eframe::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                }],
            });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        for cloud in &self.clouds {
            if cloud.len == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, cloud.buffer.slice(..));
            // Six vertices per point: two triangles making a sprite.
            pass.draw(0..6, 0..cloud.len as u32);
        }
    }
}

/// Newtype so the callback trait can be implemented locally.
///
/// The orphan rule forbids `impl CallbackTrait for Arc<PointCloudRenderer>` -
/// both the trait and `Arc` are foreign - and the renderer needs to be behind an
/// `Arc` because egui stores the callback for the life of the window while the
/// window owns it. A local wrapper satisfies both constraints.
struct CloudCallback(Arc<PointCloudRenderer>);

impl egui_wgpu::CallbackTrait for CloudCallback {
    fn paint(
        &self,
        info: epaint::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        self.0.draw_into(
            render_pass,
            [info.screen_size_px[0] as f32, info.screen_size_px[1] as f32],
        );
    }
}

fn build_vertices(pc: &PointCloud, min_z: f32, max_z: f32) -> Vec<Vertex> {
    let mut verts: Vec<Vertex> = Vec::with_capacity(pc.points.len());
    for (i, p) in pc.points.iter().enumerate() {
        // The point's own colour when it has one, otherwise a height ramp. Both
        // are decided here so the shader stays trivial and the two choices stay
        // directly comparable.
        let (color, has_color) = match pc.colors.as_ref().and_then(|c| c.get(i)) {
            Some(c) => {
                let m = c.x.abs().max(c.y.abs()).max(c.z.abs()).max(1e-6);
                ([c.x / m, c.y / m, c.z / m], 1.0)
            }
            None => (height_ramp(p.z, min_z, max_z), 0.0),
        };
        verts.push(Vertex {
            position: [p.x, p.y, p.z],
            color,
            has_color,
        });
    }
    verts
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
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

/// Blue-white-red ramp over `t` in `[0, 1]`.
fn height_ramp(z: f32, min: f32, max: f32) -> [f32; 3] {
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
