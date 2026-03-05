use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use wgpu::util::DeviceExt;

mod tables {
    pub const EDGE_TABLE: [u32; 256] = [0; 256];
    pub const TRI_TABLE: [i32; 4096] = [-1; 4096];
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct McParams {
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    voxel_size: f32,
    iso_level: f32,
    max_triangles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pos: [f32; 4],
    norm: [f32; 4],
}

pub fn extract_mesh(
    ctx: &GpuContext,
    voxel_volume: &Tensor<f32, GpuStorage<f32>>,
    voxel_size: f32,
    iso_level: f32,
    max_triangles: u32,
) -> Result<Vec<Vertex>> {
    let vol_shape = voxel_volume.shape;
    let (vx, vy, vz) = (
        vol_shape.width as u32,
        vol_shape.height as u32,
        (vol_shape.channels / 2) as u32,
    );

    let params = McParams {
        vol_x: vx,
        vol_y: vy,
        vol_z: vz,
        voxel_size,
        iso_level,
        max_triangles,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Combine tables into one i32 buffer
    let mut combined_tables = vec![0i32; 256 + 4096];
    for i in 0..256 {
        combined_tables[i] = tables::EDGE_TABLE[i] as i32;
    }
    for i in 0..4096 {
        combined_tables[256 + i] = tables::TRI_TABLE[i];
    }

    let tables_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Tables"),
            contents: bytemuck::cast_slice(&combined_tables),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_size = (max_triangles as u64) * 3 * 32;
    let vertices_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC Vertices"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let counter_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC Counter"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&counter_buffer, 0, bytemuck::bytes_of(&0u32));

    let shader_source = include_str!("../../shaders/marching_cubes.wgsl");
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Bind Group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_volume.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: counter_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tables_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MC Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(vx.div_ceil(8), vy.div_ceil(8), vz.div_ceil(4));
    }
    ctx.submit(encoder);

    let count_bytes: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counter_buffer,
        0,
        4,
    ))?;
    let num_verts = count_bytes[0] as u64;

    if num_verts == 0 {
        return Ok(Vec::new());
    }

    let vert_data: Vec<Vertex> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &vertices_buffer,
            0,
            (num_verts * 32) as usize,
        ))?;

    Ok(vert_data)
}
