use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ICPParams {
    num_src: u32,
    num_tgt: u32,
    max_dist_sq: f32,
}

pub fn icp_correspondences(
    ctx: &GpuContext,
    src: &Tensor<f32, GpuStorage<f32>>,
    tgt: &Tensor<f32, GpuStorage<f32>>,
    max_dist: f32,
) -> Result<Vec<(usize, usize, f32)>> {
    let num_src = src.shape.height;
    let num_tgt = tgt.shape.height;

    if num_src == 0 || num_tgt == 0 {
        return Ok(Vec::new());
    }

    let byte_size = (num_src * 16) as u64; // [src_idx, tgt_idx, dist_sq, valid]
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICP Correspondence Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ICPParams {
        num_src: num_src as u32,
        num_tgt: num_tgt as u32,
        max_dist_sq: max_dist * max_dist,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/icp_correspondence.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICP Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tgt.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_src as u32).div_ceil(64);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    let raw_results: Vec<[f32; 4]> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &output_buffer,
            0,
            byte_size as usize,
        ))?;

    let mut correspondences = Vec::new();
    for res in raw_results {
        if res[3] > 0.5 {
            correspondences.push((res[0] as usize, res[1] as usize, res[2].sqrt()));
        }
    }

    Ok(correspondences)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    num_points: u32,
    _pad: [u32; 3],
    transform: [[f32; 4]; 4],
}

pub fn icp_accumulate(
    ctx: &GpuContext,
    source: &Tensor<f32, GpuStorage<f32>>,
    target: &Tensor<f32, GpuStorage<f32>>,
    target_normals: &Tensor<f32, GpuStorage<f32>>,
    correspondences: &[(u32, u32)],
    transform: &nalgebra::Matrix4<f32>,
) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
    use crate::gpu_kernels::buffer_utils::create_buffer;

    let num_corr = correspondences.len();
    if num_corr == 0 {
        return Ok((nalgebra::Matrix6::zeros(), nalgebra::Vector6::zeros()));
    }

    // 1. Prepare data
    let corr_data: Vec<[u32; 2]> = correspondences.iter().map(|&(s, t)| [s, t]).collect();
    let corr_buffer = create_buffer(&ctx.device, &corr_data, wgpu::BufferUsages::STORAGE);

    let ata_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("AtA Accumulator"),
        size: 36 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let atb_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Atb Accumulator"),
        size: 6 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Zero out buffers
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&ata_buffer, 0, None);
    encoder.clear_buffer(&atb_buffer, 0, None);

    let params = AccumulateParams {
        num_points: num_corr as u32,
        _pad: [0; 3],
        transform: (*transform).into(),
    };
    let params_buffer = create_buffer(&ctx.device, &[params], wgpu::BufferUsages::UNIFORM);

    // 2. Pipeline & Bind Group
    let shader_source = include_str!("../../shaders/icp_accumulate.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Accumulate Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: target_normals.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: corr_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: atb_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // 3. Dispatch
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (num_corr as u32).div_ceil(256);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    // 4. Read back and convert from fixed-point i32 to f32
    let ata_raw: Vec<i32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &ata_buffer,
        0,
        36 * 4,
    ))?;

    let atb_raw: Vec<i32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &atb_buffer,
        0,
        6 * 4,
    ))?;

    let mut ata = nalgebra::Matrix6::<f32>::zeros();
    for i in 0..6 {
        for j in 0..6 {
            ata[(i, j)] = ata_raw[i * 6 + j] as f32 / 1000000.0;
        }
    }

    let mut atb = nalgebra::Vector6::<f32>::zeros();
    for i in 0..6 {
        atb[i] = atb_raw[i] as f32 / 1000000.0;
    }

    Ok((ata, atb))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IcpDenseParams {
    width: u32,
    height: u32,
    max_dist: f32,
    max_angle: f32,
}

pub fn dense_step(
    ctx: &GpuContext,
    source_depth: &Tensor<f32, GpuStorage<f32>>,
    target_data: &Tensor<f32, GpuStorage<f32>>,
    intrinsics: &[f32; 4],
    initial_guess: &nalgebra::Matrix4<f32>,
    max_dist: f32,
    max_angle: f32,
) -> Result<(nalgebra::Matrix6<f32>, nalgebra::Vector6<f32>)> {
    let (h, w) = source_depth.shape.hw();
    let num_pixels = (w * h) as u32;

    let scratch_size = (num_pixels * 27 * 4) as u64;
    let scratch_buffer = ctx.get_buffer(
        scratch_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );

    let params = IcpDenseParams {
        width: w as u32,
        height: h as u32,
        max_dist,
        max_angle: max_angle.cos(),
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Dense Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let intrinsics_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Intrinsics"),
            contents: bytemuck::cast_slice(intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let inv_intrinsics = [
        1.0 / intrinsics[0],
        1.0 / intrinsics[1],
        intrinsics[2],
        intrinsics[3],
    ];
    let inv_intrinsics_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Inv Intrinsics"),
            contents: bytemuck::cast_slice(&inv_intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let mut pose_flat = [0.0f32; 16];
    pose_flat.copy_from_slice(initial_guess.as_slice());
    let pose_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ICP Pose"),
            contents: bytemuck::cast_slice(&pose_flat),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/icp_dense.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICP Dense BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_depth.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_data.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scratch_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: intrinsics_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: inv_intrinsics_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: pose_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ICP Dense Compute"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((w as u32).div_ceil(16), (h as u32).div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // Reduction
    let reduce_shader = include_str!("../../shaders/icp_reduce.wgsl");
    let reduce_pipeline = ctx.create_compute_pipeline(reduce_shader, "main");

    let mut current_elements = num_pixels;
    let mut current_input = scratch_buffer;

    while current_elements > 1 {
        let workgroups = current_elements.div_ceil(128);
        let out_size = (workgroups * 27 * 4) as u64;
        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Step"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reduce_params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&current_elements),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let reduce_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ICP Reduce BG"),
            layout: &reduce_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduce_params.as_entire_binding(),
                },
            ],
        });

        let mut red_enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = red_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&reduce_pipeline);
            pass.set_bind_group(0, &reduce_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.submit(red_enc);

        current_input = out_buffer;
        current_elements = workgroups;
    }

    let final_data: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &current_input, // Final result is here
        0,
        27 * 4,
    ))?;

    let mut ata = nalgebra::Matrix6::zeros();
    let mut atb = nalgebra::Vector6::zeros();

    let mut idx = 0;
    for i in 0..6 {
        for j in i..6 {
            ata[(i, j)] = final_data[idx];
            ata[(j, i)] = final_data[idx];
            idx += 1;
        }
    }
    for i in 0..6 {
        atb[i] = final_data[idx];
        idx += 1;
    }

    Ok((ata, atb))
}
