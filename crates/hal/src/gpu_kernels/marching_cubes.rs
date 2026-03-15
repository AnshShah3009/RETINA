use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use wgpu::util::DeviceExt;

use super::marching_cubes_tables as tables;

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

impl Vertex {
    /// Create a new vertex with position and normal.
    pub fn new(pos: [f32; 4], norm: [f32; 4]) -> Self {
        Self { pos, norm }
    }

    /// Get the position.
    pub fn pos(&self) -> &[f32; 4] {
        &self.pos
    }

    /// Get the normal.
    pub fn norm(&self) -> &[f32; 4] {
        &self.norm
    }
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
    for (dst, &src) in combined_tables[..256]
        .iter_mut()
        .zip(tables::EDGE_TABLE.iter())
    {
        *dst = src as i32;
    }
    combined_tables[256..256 + 4096].copy_from_slice(&tables::TRI_TABLE[..4096]);

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

/// Two-pass marching cubes: count triangles, CPU prefix sum, emit at exact offsets.
/// Eliminates atomic contention and allocates exact-size output buffers.
pub fn extract_mesh_v2(
    ctx: &GpuContext,
    voxel_volume: &Tensor<f32, GpuStorage<f32>>,
    voxel_size: f32,
    iso_level: f32,
) -> Result<Vec<Vertex>> {
    let vol_shape = voxel_volume.shape;
    let (vx, vy, vz) = (
        vol_shape.width as u32,
        vol_shape.height as u32,
        (vol_shape.channels / 2) as u32,
    );

    if vx < 2 || vy < 2 || vz < 2 {
        return Ok(Vec::new());
    }

    let num_cells = ((vx - 1) * (vy - 1) * (vz - 1)) as usize;

    // --- Params (shared by both passes) ---
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct McParams2 {
        vol_x: u32,
        vol_y: u32,
        vol_z: u32,
        voxel_size: f32,
        iso_level: f32,
        _pad: u32,
    }
    let params = McParams2 {
        vol_x: vx,
        vol_y: vy,
        vol_z: vz,
        voxel_size,
        iso_level,
        _pad: 0,
    };
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC2 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // --- TRI_COUNT table (256 u32s) ---
    let tri_count_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC2 TriCount"),
            contents: bytemuck::cast_slice(&tables::TRI_COUNT),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // --- Counts buffer (one u32 per cell) ---
    let counts_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC2 Counts"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ======================= PASS 1: Count =======================
    let count_shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC2 Count Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/marching_cubes_count.wgsl").into(),
            ),
        });

    let count_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC2 Count Pipeline"),
            layout: None,
            module: &count_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let count_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC2 Count BG"),
        layout: &count_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_volume.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: counts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tri_count_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MC2 Count"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&count_pipeline);
        pass.set_bind_group(0, &count_bg, &[]);
        pass.dispatch_workgroups(
            (vx - 1).div_ceil(8),
            (vy - 1).div_ceil(8),
            (vz - 1).div_ceil(4),
        );
    }
    ctx.submit(encoder);

    // Read back counts
    let counts: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buf,
        0,
        num_cells * 4,
    ))?;

    // ======================= CPU PREFIX SUM =======================
    let mut offsets = vec![0u32; num_cells + 1];
    for i in 0..num_cells {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let total_tris = offsets[num_cells] as usize;

    if total_tris == 0 {
        return Ok(Vec::new());
    }

    // ======================= PASS 2: Emit =======================
    let offsets_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC2 Offsets"),
            contents: bytemuck::cast_slice(&offsets[..num_cells]),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Exact-size vertex buffer
    let total_verts = total_tris * 3;
    let vertices_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC2 Vertices"),
        size: (total_verts as u64) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Combined edge+tri tables (same format as v1)
    let mut combined_tables = vec![0i32; 256 + 4096];
    for (dst, &src) in combined_tables[..256]
        .iter_mut()
        .zip(tables::EDGE_TABLE.iter())
    {
        *dst = src as i32;
    }
    combined_tables[256..256 + 4096].copy_from_slice(&tables::TRI_TABLE[..4096]);
    let tables_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC2 Tables"),
            contents: bytemuck::cast_slice(&combined_tables),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let emit_shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC2 Emit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/marching_cubes_emit.wgsl").into(),
            ),
        });

    let emit_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC2 Emit Pipeline"),
            layout: None,
            module: &emit_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let emit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC2 Emit BG"),
        layout: &emit_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_volume.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tables_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: offsets_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MC2 Emit"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&emit_pipeline);
        pass.set_bind_group(0, &emit_bg, &[]);
        pass.dispatch_workgroups(
            (vx - 1).div_ceil(8),
            (vy - 1).div_ceil(8),
            (vz - 1).div_ceil(4),
        );
    }
    ctx.submit(encoder);

    // Read exact-size result
    let vert_data: Vec<Vertex> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &vertices_buf,
            0,
            total_verts * 32,
        ))?;

    Ok(vert_data)
}
