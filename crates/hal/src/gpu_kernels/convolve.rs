use crate::context::BorderMode;
use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use crate::Result;
use cv_core::Float;
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

pub fn convolve_2d<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    kernel: &crate::GpuTensor<T>,
    border_mode: BorderMode<T>,
) -> Result<crate::GpuTensor<T>> {
    // Only f32 WGSL shader available
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Convolve2D GPU kernel only supports f32".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let (kh, kw) = kernel.shape.hw();

    let output_size = input.shape.len();
    let output_byte_size = (output_size * std::mem::size_of::<f32>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Convolve Output"),
        size: output_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let (mode_int, const_val) = match border_mode {
        BorderMode::Constant(v) => (0, v.to_f32()),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Reflect => (2, 0.0),
        BorderMode::Wrap => (3, 0.0),
        BorderMode::Reflect101 => (4, 0.0),
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

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(output_buffer), output_size),
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

/// Convert a `BorderMode` to its integer representation for GPU shaders.
fn border_mode_to_int<T: Float>(mode: &BorderMode<T>) -> u32 {
    match mode {
        BorderMode::Constant(_) => 0,
        BorderMode::Replicate => 1,
        BorderMode::Reflect => 2,
        BorderMode::Wrap => 3,
        BorderMode::Reflect101 => 4,
    }
}

pub fn gaussian_blur<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    sigma: T,
    k_size: usize,
) -> Result<crate::GpuTensor<T>> {
    gaussian_blur_with_border(ctx, input, sigma, k_size, BorderMode::Replicate)
}

pub fn gaussian_blur_with_border<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    sigma: T,
    k_size: usize,
    border_mode: BorderMode<T>,
) -> Result<crate::GpuTensor<T>> {
    // Only f32 WGSL shader available
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Gaussian Blur GPU kernel only supports f32".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let kernel_1d = crate::cpu::gaussian_kernel_1d(sigma.to_f32(), k_size);

    let output_size = input.shape.len();
    let output_byte_size = (output_size * std::mem::size_of::<f32>()) as u64;

    let border_int = border_mode_to_int(&border_mode);

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
        border_mode: border_int,
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
        border_mode: border_int,
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

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(output_buffer), output_size),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
