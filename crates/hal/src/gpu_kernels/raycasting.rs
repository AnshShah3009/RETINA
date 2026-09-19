use crate::gpu::GpuContext;
use crate::Result;
use nalgebra::{Point3, Vector3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RaycastUniforms {
    num_rays: u32,
    num_faces: u32,
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
    let gpu_origins: Vec<GpuVec3> = rays.iter().map(|(o, _)| GpuVec3::from(*o)).collect();
    let gpu_dirs: Vec<GpuVec3> = rays.iter().map(|(_, d)| GpuVec3::from(*d)).collect();
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

    let origins_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ray Origins"),
            contents: bytemuck::cast_slice(&gpu_origins),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let dirs_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ray Directions"),
            contents: bytemuck::cast_slice(&gpu_dirs),
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

    // Output buffers
    let dists_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hit Distances"),
        size: (num_rays * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let points_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hit Points"),
        size: (num_rays * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let normals_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hit Normals"),
        size: (num_rays * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = RaycastUniforms {
        num_rays: num_rays as u32,
        num_faces: faces.len() as u32,
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

    // Fix: Use 2 bind groups as defined in shader
    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Raycast BG0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: origins_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dirs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vertices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: faces_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dists_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: points_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: normals_buf.as_entire_binding(),
            },
        ],
    });

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

    // 4. Read back separate arrays
    let final_dists: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &dists_buf,
        0,
        num_rays * 4,
    ))?;
    let final_points: Vec<GpuVec3> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &points_buf,
            0,
            num_rays * 16,
        ))?;
    let final_normals: Vec<GpuVec3> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &normals_buf,
            0,
            num_rays * 16,
        ))?;

    Ok(final_dists
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            if d >= 0.0 {
                let p = final_points[i].data;
                let n = final_normals[i].data;
                Some((
                    d,
                    Point3::new(p[0], p[1], p[2]),
                    Vector3::new(n[0], n[1], n[2]),
                ))
            } else {
                None
            }
        })
        .collect())
}
