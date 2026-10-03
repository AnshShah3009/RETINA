#![forbid(unsafe_code)]
//! What one point sprite actually covers on screen, measured in pixels.
//!
//! The host says a point radius is in **pixels** (`Uniforms::pixel_size` carries
//! the viewport in pixels for exactly this conversion, and `NativeViewer`'s own
//! test `a_point_radius_in_pixels_is_constant_on_screen` pins the arithmetic).
//! The shader is the only place that arithmetic is applied, so it is measured
//! here: a small cloud is rendered into an offscreen texture and the frame is
//! read back, which makes the lit bounding box and the lit pixel count the
//! sprite's real size.
//!
//! This mirrors the pipeline `render.rs` builds - same shader file, same
//! 28-byte layout, same `draw(0..6, 0..points)`. A change to either side that
//! this file does not follow shows up as a failure here, which is the point:
//! `render.rs`'s own test module pins the step mode and the vertex layout
//! directly, and this file pins the pixels that come out of them.
//!
//! Skips (with a printed note) when no wgpu adapter is available, because CI has
//! no GPU.
use eframe::wgpu;
use wgpu::util::DeviceExt;

/// The shader the renderer ships, compiled here as-is.
const SHADER: &str = include_str!("../src/native_viewer/point_cloud.wgsl");

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
/// `NativeViewer::FOV_SCALE`, the tangent of the camera's half-angle.
const FOV_SCALE: f32 = 0.5;
/// Distance from the camera to the origin.
const CAMERA_DISTANCE: f32 = 5.0;

/// A device and queue, or `None` when the machine has no wgpu adapter.
fn device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .ok()
}

/// A camera `CAMERA_DISTANCE` along +z looking at the origin, in the row-major
/// convention `look_at` documents: `eye[i] = sum_j m[i][j] * world[j]`, with
/// `eye[3]` the depth in front of the camera.
fn camera() -> [[f32; 4]; 4] {
    let k = 1.0 / FOV_SCALE;
    [
        [k, 0.0, 0.0, 0.0],
        [0.0, k, 0.0, 0.0],
        [0.0, 0.0, k, -CAMERA_DISTANCE * k],
        [0.0, 0.0, -1.0, CAMERA_DISTANCE],
    ]
}

/// Where a world point lands in the frame, in pixels, for [`camera`].
fn project(point: [f32; 3]) -> (f32, f32) {
    let m = camera();
    let p = [point[0], point[1], point[2], 1.0];
    let mut eye = [0.0f32; 4];
    for (r, e) in eye.iter_mut().enumerate() {
        *e = (0..4).map(|c| m[r][c] * p[c]).sum();
    }
    let ndc = [eye[0] / eye[3], eye[1] / eye[3]];
    (
        (ndc[0] + 1.0) * 0.5 * WIDTH as f32,
        (1.0 - ndc[1]) * 0.5 * HEIGHT as f32,
    )
}

struct Measured {
    lit: u32,
    x_extent: i64,
    y_extent: i64,
    /// Whether each requested point has a lit pixel at its own centre.
    centres_lit: Vec<bool>,
}

/// Render `points` as point sprites of radius `radius_px` and read the frame
/// back.
fn measure_sprite(points: &[[f32; 3]], radius_px: f32) -> Measured {
    let (device, queue) = device_and_queue().expect("checked by the caller");
    let format = wgpu::TextureFormat::Rgba8Unorm;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sprite geometry shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    // 3 position + 3 colour + 1 has_color, 28 bytes, as `render::Vertex`.
    let vertices: Vec<[f32; 7]> = points
        .iter()
        .map(|p| [p[0], p[1], p[2], 1.0, 1.0, 1.0, 1.0])
        .collect();
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut uniform = [0.0f32; 20];
    for (r, row) in camera().iter().enumerate() {
        uniform[r * 4..r * 4 + 4].copy_from_slice(row);
    }
    uniform[16] = WIDTH as f32;
    uniform[17] = HEIGHT as f32;
    uniform[18] = radius_px;
    uniform[19] = 0.0;
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sprite geometry pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 28,
                // One vertex per point, six vertices per point drawn: the buffer
                // advances per instance and the shader's `vertex_index` picks the
                // corner. `render.rs`'s `vertex_layout()` is the shipping copy of
                // this and its test module pins the step mode.
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 24,
                        shader_location: 2,
                    },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }],
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.draw(0..6, 0..vertices.len() as u32);
    }

    let bytes_per_row = WIDTH * 4;
    let padded = bytes_per_row.div_ceil(256) * 256;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    let slice = out.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
    rx.recv().expect("map callback").expect("readback");

    let data = slice.get_mapped_range();
    let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    let mut lit = 0u32;
    let mut centres_lit = Vec::with_capacity(points.len());
    for point in points {
        let (cx, cy) = project(*point);
        let (cx, cy) = (cx.round() as i64, cy.round() as i64);
        let mut found = false;
        for (dy, dx) in [(0i64, 0i64), (0, 1), (1, 0), (0, -1), (-1, 0)] {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= WIDTH as i64 || y >= HEIGHT as i64 {
                continue;
            }
            let i = (y as u32 * padded + x as u32 * 4) as usize;
            let px = &data[i..i + 4];
            if px[0] > 8 || px[1] > 8 || px[2] > 8 {
                found = true;
            }
        }
        centres_lit.push(found);
    }
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let i = (y * padded + x * 4) as usize;
            let px = &data[i..i + 4];
            if px[0] > 8 || px[1] > 8 || px[2] > 8 {
                lit += 1;
                lo_x = lo_x.min(x as i64);
                hi_x = hi_x.max(x as i64);
                lo_y = lo_y.min(y as i64);
                hi_y = hi_y.max(y as i64);
            }
        }
    }
    drop(data);
    out.unmap();

    Measured {
        lit,
        x_extent: if lit == 0 { 0 } else { hi_x - lo_x + 1 },
        y_extent: if lit == 0 { 0 } else { hi_y - lo_y + 1 },
        centres_lit,
    }
}

/// A point radius is a radius in pixels: the sprite is `2 * radius_px` across,
/// the same number of pixels on both axes, whatever the viewport's shape.
#[test]
fn a_point_radius_is_a_radius_in_pixels() {
    if device_and_queue().is_none() {
        eprintln!("skipping: no wgpu adapter");
        return;
    }

    let centre = [[0.0f32, 0.0, 0.0]];
    let mut extents = Vec::new();
    for radius_px in [4.0f32, 8.0] {
        let m = measure_sprite(&centre, radius_px);
        eprintln!(
            "radius {radius_px} px -> sprite {} x {} px, {} lit pixels",
            m.x_extent, m.y_extent, m.lit
        );
        assert!(
            m.centres_lit == vec![true],
            "the point itself was not drawn"
        );
        let expected = (2.0 * radius_px) as i64;
        assert!(
            (m.x_extent - expected).abs() <= 2,
            "a point radius of {radius_px} px must cover {expected} px across; measured \
             {} px of {WIDTH}. The sprite offset is scaled by the half viewport twice - \
             once into NDC (`point_radius / half_width`) and once back out \
             (`* half_width`) - so what survives the perspective divide is \
             `point_radius` in NDC, i.e. point_radius * (width / 2) pixels.",
            m.x_extent
        );
        assert!(
            (m.y_extent - expected).abs() <= 2,
            "a point radius of {radius_px} px must cover {expected} px down; measured {} \
             px of {HEIGHT}. The y offset used the *x* half-extent, so on a \
             {WIDTH}x{HEIGHT} viewport it was off by height / width.",
            m.y_extent
        );
        extents.push(m.y_extent);
    }
    assert_eq!(
        extents[1],
        2 * extents[0],
        "doubling the point radius must double the sprite: measured {extents:?} px"
    );
}

/// Every point in a cloud gets its own sprite, at its own projected position.
///
/// This is the check the step mode broke: with the vertex buffer stepped per
/// vertex, each of the six vertices of a sprite read a *different* buffer entry,
/// so a cloud's sprites were built from the first six points and the triangles
/// between them spanned the cloud. A one-point cloud went further and was not a
/// valid draw at all.
#[test]
fn every_point_gets_a_sprite_at_its_own_position() {
    if device_and_queue().is_none() {
        eprintln!("skipping: no wgpu adapter");
        return;
    }

    // Eight points in a row, 0.4 world units apart: 51.2 px apart on screen,
    // which is far enough that their 8 px sprites cannot touch.
    let points: Vec<[f32; 3]> = (0..8).map(|i| [-1.4 + 0.4 * i as f32, 0.0, 0.0]).collect();
    let m = measure_sprite(&points, 4.0);
    eprintln!(
        "8 points -> {} x {} px, {} lit pixels, centres lit {:?}",
        m.x_extent, m.y_extent, m.lit, m.centres_lit
    );

    assert!(
        m.centres_lit.iter().all(|lit| *lit),
        "every point must have a sprite at its own projected position; measured \
         {:?}. A sprite whose six vertices come from six different buffer entries is \
         assembled from the wrong points entirely.",
        m.centres_lit
    );
    // A 4 px radius disc covers pi * 16 = 50 px plus antialiasing.
    let per_sprite = m.lit as f64 / points.len() as f64;
    assert!(
        (20.0..160.0).contains(&per_sprite),
        "8 sprites of radius 4 px should light about 400 pixels in total, measured \
         {} ({per_sprite:.1} per point) of {}",
        m.lit,
        WIDTH * HEIGHT
    );
    // Seven gaps of 51.2 px plus two 8 px sprites, and 8 px across vertically.
    assert!(
        (350..=380).contains(&m.x_extent),
        "the eight sprites span the projected row: measured {} px, expected about 366",
        m.x_extent
    );
    assert!(
        (6..=10).contains(&m.y_extent),
        "each sprite is 8 px tall: measured {} px, expected about 8",
        m.y_extent
    );

    // A single point is the degenerate case that needs the buffer stepped per
    // instance to be drawable at all.
    let one = measure_sprite(&[[0.0, 0.0, 0.0]], 4.0);
    assert_eq!(one.centres_lit, vec![true]);
    assert!(
        (6..=10).contains(&one.y_extent),
        "a one-point cloud must draw one 8 px sprite, measured {} px",
        one.y_extent
    );
}
