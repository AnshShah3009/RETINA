use crate::context::BorderMode;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvolveParams {
    width: u32,
    height: u32,
    kernel_width: u32,
    kernel_height: u32,
    border_mode: u32,
    border_const: f32,
}

pub fn convolve_2d(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    kernel: &Tensor<f32, GpuStorage<f32>>,
    border_mode: BorderMode,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = input.shape.hw();
    let (kh, kw) = kernel.shape.hw();

    // Create output buffer
    let output_size = input.shape.len();
    let output_byte_size = (output_size * std::mem::size_of::<f32>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Convolve Output"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Prepare params
    let (mode_int, const_val) = match border_mode {
        BorderMode::Constant(v) => (0, v),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Reflect => (2, 0.0),
        BorderMode::Wrap => (3, 0.0),
    };

    let params = ConvolveParams {
        width: w as u32,
        height: h as u32,
        kernel_width: kw as u32,
        kernel_height: kh as u32,
        border_mode: mode_int,
        border_const: const_val,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Convolve Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/convolve_2d.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Convolve Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel.storage.buffer().as_entire_binding(),
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
        let wg_x = (w as u32).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), output_size),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SeparableParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    is_vertical: u32,
    border_mode: u32,
    padding: u32,
}

pub fn gaussian_blur(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    sigma: f32,
    k_size: usize,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = input.shape.hw();
    let kernel_1d = crate::cpu::gaussian_kernel_1d(sigma, k_size);

    let output_size = input.shape.len();
    let output_byte_size = (output_size * std::mem::size_of::<f32>()) as u64;

    let temp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Blur Temp Buffer"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Blur Final Buffer"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let kernel_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur Kernel"),
            contents: bytemuck::cast_slice(&kernel_1d),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let shader_source = include_str!("../../shaders/gaussian_blur_separable.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let h_params = SeparableParams {
        width: w as u32,
        height: h as u32,
        kernel_size: k_size as u32,
        is_vertical: 0,
        border_mode: 1,
        padding: 0,
    };
    let h_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&h_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let h_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Horizontal Blur Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: h_params_buf.as_entire_binding(),
            },
        ],
    });

    let v_params = SeparableParams {
        width: w as u32,
        height: h as u32,
        kernel_size: k_size as u32,
        is_vertical: 1,
        border_mode: 1,
        padding: 0,
    };
    let v_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&v_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let v_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Vertical Blur Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: temp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let wg_x = (w as u32).div_ceil(16);
    let wg_y = (h as u32).div_ceil(16);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HPass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &h_bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    // Implicit barrier between passes in the same encoder
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VPass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &v_bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), output_size),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
