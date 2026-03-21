use crate::gpu::GpuContext;
use crate::GpuTensor;
use crate::Result;
use cv_core::storage::Storage;
use cv_core::TensorShape;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResizeParams {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub channels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Bilinear,
    Lanczos4,
}

pub fn resize<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
) -> Result<GpuTensor<T>> {
    resize_with_mode(
        ctx,
        input,
        new_width,
        new_height,
        InterpolationMode::Bilinear,
    )
}

pub fn resize_with_mode<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
    mode: InterpolationMode,
) -> Result<GpuTensor<T>> {
    use crate::storage::GpuStorage;
    let (src_h, src_w) = input.shape.hw();
    let (dst_w, dst_h) = (new_width as usize, new_height as usize);
    let c = input.shape.channels;

    if c == 0 {
        return Err(crate::Error::InvalidInput(
            "Resize requires at least one channel".into(),
        ));
    }

    let _out_shape = TensorShape::new(c, new_height as usize, new_width as usize);
    let out_len = (new_width * new_height) as usize * c;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resize Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ResizeParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w: dst_w as u32,
        dst_h: dst_h as u32,
        channels: c as u32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resize Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = match mode {
        InterpolationMode::Bilinear => {
            if c == 1 {
                include_str!("../../shaders/resize_f32.wgsl").to_string()
            } else {
                include_str!("../../shaders/resize_f32_multichannel.wgsl").to_string()
            }
        }
        InterpolationMode::Lanczos4 => include_str!("../../shaders/lanczos4.wgsl").to_string(),
    };
    let pipeline = ctx.create_compute_pipeline(&shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resize Bind Group"),
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
        let x = (dst_w as u32).div_ceil(16);
        let y = (dst_h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    let res_gpu = crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: cv_core::TensorShape::new(c, dst_h, dst_w),
        dtype: input.dtype,
        _phantom: PhantomData,
    };
    Ok(res_gpu)
}

pub fn resize_lanczos4<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
) -> Result<GpuTensor<T>> {
    resize_with_mode(
        ctx,
        input,
        new_width,
        new_height,
        InterpolationMode::Lanczos4,
    )
}
