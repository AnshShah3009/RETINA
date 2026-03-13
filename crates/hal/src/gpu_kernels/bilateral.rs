use crate::gpu::GpuContext;
use crate::Result;
use cv_core::storage::Storage;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone)]
#[repr(C)]
struct BilateralParams {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
}

unsafe impl bytemuck::Pod for BilateralParams {}
unsafe impl bytemuck::Zeroable for BilateralParams {}

pub fn bilateral_filter<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    d: i32,
    sigma_color: T,
    sigma_space: T,
) -> Result<crate::GpuTensor<T>> {
    use crate::storage::GpuStorage;
    let sigma_color_f = sigma_color.to_f32();
    let sigma_space_f = sigma_space.to_f32();
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Bilateral currently only for grayscale".into(),
        ));
    }

    let out_len = h * w * c;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bilateral Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let radius = if d <= 0 {
        (sigma_space_f * 1.5).ceil() as i32
    } else {
        d / 2
    };
    let params = BilateralParams {
        width: w as u32,
        height: h as u32,
        radius,
        sigma_color_sq_inv: -0.5 / (sigma_color_f * sigma_color_f),
        sigma_space_sq_inv: -0.5 / (sigma_space_f * sigma_space_f),
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bilateral Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = match cv_core::DataType::from_type::<T>() {
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::F16) => include_str!("../../shaders/bilateral_f16.wgsl"),
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::BF16) => include_str!("../../shaders/bilateral_bf16.wgsl"),
        _ => {
            include_str!("../../shaders/bilateral_f32.wgsl")
        }
    };
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bilateral Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input
                    .storage
                    .as_any()
                    .downcast_ref::<GpuStorage<f32>>()
                    .unwrap()
                    .buffer()
                    .as_entire_binding(),
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
        let x = (w as u32).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let res_gpu = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    };
    Ok(res_gpu)
}
