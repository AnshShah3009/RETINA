use crate::context::ColorConversion;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorParams {
    len: u32,
    code: u32,
}

pub fn cvt_color(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    code: ColorConversion,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let (h, w) = input.shape.hw();
    let num_pixels = h * w;

    let (out_channels, code_int) = match code {
        ColorConversion::RgbToGray => (1, 0),
        ColorConversion::BgrToGray => (1, 1),
        ColorConversion::GrayToRgb => (3, 2),
    };

    if code_int == 2 {
        return Err(crate::Error::NotSupported("GPU GrayToRgb pending".into()));
    }

    let out_len = num_pixels * out_channels;
    let byte_size = (out_len.div_ceil(4) * 4) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Color Cvt Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = ColorParams {
        len: num_pixels as u32,
        code: code_int,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/color_cvt.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Color Bind Group"),
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
        let x = (num_pixels as u32).div_ceil(4).div_ceil(64);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: TensorShape::new(out_channels, h, w),
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
