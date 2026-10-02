use crate::gpu::GpuContext;
use crate::Result;
use nalgebra::{Point3, Vector3};
use wgpu::util::DeviceExt;

/// Matches `struct RaycastParams` in `ray_mesh_intersection.wgsl`. A WGSL uniform
/// struct must be a multiple of its alignment (16), so the two counts are
/// padded out to four scalars.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RaycastUniforms {
    num_rays: u32,
    num_faces: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVec3 {
    data: [f32; 3],
    padding: f32,
}

impl From<Point3<f32>> for GpuVec3 {
    fn from(p: Point3<f32>) -> Self {
        Self {
            data: [p.x, p.y, p.z],
            padding: 0.0,
        }
    }
}

impl From<Vector3<f32>> for GpuVec3 {
    fn from(v: Vector3<f32>) -> Self {
        Self {
            data: [v.x, v.y, v.z],
            padding: 0.0,
        }
    }
}

/// Matches `struct Ray` in `ray_mesh_intersection.wgsl`: `vec3<f32>` has a
/// 16-byte stride inside an array or struct, so a ray occupies 32 bytes with
/// the direction starting at byte 16 - exactly two `GpuVec3`s back to back.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRay {
    origin: GpuVec3,
    direction: GpuVec3,
}

/// Matches `struct Hit` in `ray_mesh_intersection.wgsl`: `point_dist` at byte 0
/// (xyz = point, w = distance) and `normal_pad` at byte 16 (xyz = normal).
/// 32-byte stride.
///
/// Note this is deliberately two `vec4`s rather than three `vec3`s plus an
/// `f32`: WGSL puts the trailing scalar at byte 28 in that layout and rounds the
/// struct up to 48, which is not where `#[repr(C)]` puts it, so the two
/// representations would disagree about where the distance lives.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHit {
    point_dist: [f32; 4],
    normal_pad: [f32; 4],
}

const _: () = {
    // The shader's layout is load-bearing: these are the offsets the shader
    // writes to and the host reads back.
    assert!(std::mem::size_of::<GpuVec3>() == 16);
    assert!(std::mem::size_of::<GpuRay>() == 32);
    assert!(std::mem::size_of::<GpuHit>() == 32);
    assert!(std::mem::align_of::<GpuHit>() == 4);
};

#[allow(clippy::type_complexity)]
pub fn cast_rays(
    ctx: &GpuContext,
    rays: &[(Point3<f32>, Vector3<f32>)],
    vertices: &[Point3<f32>],
    faces: &[[u32; 3]],
) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>> {
    let num_rays = rays.len();
    if num_rays == 0 {
        return Ok(Vec::new());
    }

    // 1. Prepare buffers with proper alignment (vec3 in storage = 16 bytes)
    //
    // Origins and directions are interleaved into one buffer because the device
    // caps storage buffers per shader stage at four (downlevel_defaults) and
    // the shader's `main` needs mesh vertices, mesh faces and hit outputs as
    // well. See the header comment in ray_mesh_intersection.wgsl.
    let gpu_rays: Vec<GpuRay> = rays
        .iter()
        .map(|(o, d)| GpuRay {
            origin: GpuVec3::from(*o),
            direction: GpuVec3::from(*d),
        })
        .collect();
    let gpu_vertices: Vec<GpuVec3> = vertices.iter().map(|&v| GpuVec3::from(v)).collect();

    // faces in WGSL: array<vec3<u32>> -> also 16 byte stride
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuFace {
        indices: [u32; 3],
        padding: u32,
    }
    let gpu_faces: Vec<GpuFace> = faces
        .iter()
        .map(|f| GpuFace {
            indices: *f,
            padding: 0,
        })
        .collect();

    let rays_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rays (origin + direction)"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let vertices_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertices"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let faces_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Faces"),
            contents: bytemuck::cast_slice(&gpu_faces),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Output buffer: point, normal and distance interleaved per ray.
    let hits_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hit Points + Normals + Distances"),
        size: (num_rays * std::mem::size_of::<GpuHit>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = RaycastUniforms {
        num_rays: num_rays as u32,
        num_faces: faces.len() as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let uniforms_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Raycast Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // 2. Pipeline
    let shader_source = include_str!("ray_mesh_intersection.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raycast BG0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: rays_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: faces_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hits_buf.as_entire_binding(),
            },
        ],
    });

    // `num_rays` and `num_faces` both moved into the single @group(1) uniform
    // struct. Binding only one entry here while the shader's layout derives
    // binding 1 as well was rejected: "Number of bindings in bind group
    // descriptor (1) does not match the number of bindings defined in the
    // bind group layout (2)".
    let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raycast BG1"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms_buf.as_entire_binding(),
        }],
    });

    // 3. Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(num_rays.div_ceil(256) as u32, 1, 1);
    }
    ctx.submit(encoder);

    // 4. Read back the interleaved hits
    let final_hits: Vec<GpuHit> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &hits_buf,
            0,
            num_rays * std::mem::size_of::<GpuHit>(),
        ))?;

    Ok(final_hits
        .into_iter()
        .map(|h| {
            let dist = h.point_dist[3];
            if dist >= 0.0 {
                Some((
                    dist,
                    Point3::new(h.point_dist[0], h.point_dist[1], h.point_dist[2]),
                    Vector3::new(h.normal_pad[0], h.normal_pad[1], h.normal_pad[2]),
                ))
            } else {
                None
            }
        })
        .collect())
}
