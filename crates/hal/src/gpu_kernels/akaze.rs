use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DiffusionParams {
    width: u32,
    height: u32,
    k: f32,
    tau: f32,
}

pub fn akaze_diffusion(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    k: f32,
    tau: f32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = input.shape.hw();
    let size = input.shape.len();
    let byte_size = (size * 4) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("AKAZE Diffusion Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = DiffusionParams {
        width: w as u32,
        height: h as u32,
        k,
        tau,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Diffusion Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/akaze_diffusion.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Diffusion Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
        let wg_x = (w as u32).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), size),
        shape: input.shape,
        dtype: cv_core::DataType::F32,
        _phantom: PhantomData,
    })
}

#[allow(clippy::type_complexity)]
pub fn akaze_derivatives(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
) -> Result<(
    Tensor<f32, GpuStorage<f32>>,
    Tensor<f32, GpuStorage<f32>>,
    Tensor<f32, GpuStorage<f32>>,
)> {
    let (h, w) = input.shape.hw();
    let size = input.shape.len();
    let byte_size = (size * 4) as u64;

    let lx_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Lx"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));
    let ly_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ly"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));
    let ldet_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ldet"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));

    let params = [w as u32, h as u32, 0, 0];
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Deriv Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/akaze_derivatives.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Deriv Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ly_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ldet_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
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
        let wg_x = (w as u32).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    Ok((
        Tensor {
            storage: GpuStorage::from_buffer(lx_buffer, size),
            shape: input.shape,
            dtype: cv_core::DataType::F32,
            _phantom: PhantomData,
        },
        Tensor {
            storage: GpuStorage::from_buffer(ly_buffer, size),
            shape: input.shape,
            dtype: cv_core::DataType::F32,
            _phantom: PhantomData,
        },
        Tensor {
            storage: GpuStorage::from_buffer(ldet_buffer, size),
            shape: input.shape,
            dtype: cv_core::DataType::F32,
            _phantom: PhantomData,
        },
    ))
}

pub fn akaze_contrast_k(ctx: &GpuContext, input: &Tensor<f32, GpuStorage<f32>>) -> Result<f32> {
    let (h, w) = input.shape.hw();
    let num_bins = 1000u32;
    let max_mag = 0.5f32;

    let hist_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("K Histogram"),
        size: (num_bins * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&hist_buffer, 0, None);

    let params = [w as u32, h as u32, num_bins, bytemuck::cast(max_mag)];
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("K Params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/akaze_contrast.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("K Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hist_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (w as u32).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    let hist: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &hist_buffer,
        0,
        (num_bins * 4) as usize,
    ))?;

    // Find 70th percentile
    let total_elements = hist.iter().sum::<u32>();
    let target = (total_elements as f32 * 0.7) as u32;
    let mut sum = 0u32;
    for (i, &count) in hist.iter().enumerate() {
        sum += count;
        if sum >= target {
            return Ok((i as f32 / num_bins as f32) * max_mag);
        }
    }

    Ok(0.03) // Default
}
