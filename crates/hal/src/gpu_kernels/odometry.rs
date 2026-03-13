use crate::gpu::GpuContext;
use crate::Result;
use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OdometryParams {
    width: u32,
    height: u32,
    max_iterations: u32,
    min_depth: f32,
    max_depth: f32,
    padding: [u32; 3],
}

pub fn compute_odometry(
    ctx: &GpuContext,
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &[f32; 4],
    width: u32,
    height: u32,
    init_transform: &Matrix4<f32>,
) -> Result<(Matrix4<f32>, f32, f32)> {
    let num_pixels = (width * height) as usize;

    // 1. Preprocess target depth to get vertex and normal maps
    let target_depth_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Target Depth"),
            contents: bytemuck::cast_slice(target_depth),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let vertex_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Target Vertex Map"),
        size: (num_pixels * 16) as u64, // vec3 stride is 16
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let normal_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Target Normal Map"),
        size: (num_pixels * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let intrinsics_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Intrinsics"),
            contents: bytemuck::cast_slice(intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let size_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Size"),
            contents: bytemuck::cast_slice(&[width, height]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let preprocess_shader = include_str!("depth_preprocess.wgsl");
    let preprocess_pipeline = ctx.create_compute_pipeline(preprocess_shader, "main");

    let preprocess_bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preprocess BG0"),
        layout: &preprocess_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: target_depth_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_map_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: normal_map_buf.as_entire_binding(),
            },
        ],
    });

    let preprocess_bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preprocess BG1"),
        layout: &preprocess_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: intrinsics_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: size_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&preprocess_pipeline);
        pass.set_bind_group(0, &preprocess_bg0, &[]);
        pass.set_bind_group(1, &preprocess_bg1, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // 2. Main Odometry Pass
    let source_depth_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Source Depth"),
            contents: bytemuck::cast_slice(source_depth),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let residual_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Odom Residuals"),
        size: (num_pixels * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut pose_flat = [0.0f32; 16];
    pose_flat.copy_from_slice(init_transform.as_slice());
    let pose_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Pose"),
            contents: bytemuck::cast_slice(&pose_flat),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let params = OdometryParams {
        width,
        height,
        max_iterations: 1, // Single pass for this task
        min_depth: 0.1,
        max_depth: 10.0,
        padding: [0; 3],
    };
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Odom Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let odometry_shader = include_str!("rgbd_odometry.wgsl");
    let odometry_pipeline = ctx.create_compute_pipeline(odometry_shader, "main");

    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Odom BG0"),
        layout: &odometry_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_depth_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_depth_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vertex_map_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: normal_map_buf.as_entire_binding(),
            },
        ],
    });

    let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Odom BG1"),
        layout: &odometry_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &residual_buf,
                    offset: 0,
                    size: None,
                }),
            }, // Placeholder for jacobian
            wgpu::BindGroupEntry {
                binding: 1,
                resource: residual_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pose_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: intrinsics_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: size_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&odometry_pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
    }
    ctx.submit(encoder);

    // 3. Read back and compute metrics
    let residuals: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &residual_buf,
        0,
        num_pixels * 4,
    ))?;

    let mut total_error = 0.0;
    let mut valid_points = 0;
    for &r in &residuals {
        if r != 0.0 {
            total_error += r * r;
            valid_points += 1;
        }
    }

    let fitness = if num_pixels > 0 {
        valid_points as f32 / num_pixels as f32
    } else {
        0.0
    };
    let rmse = if valid_points > 0 {
        (total_error / valid_points as f32).sqrt()
    } else {
        0.0
    };

    Ok((*init_transform, fitness, rmse))
}
