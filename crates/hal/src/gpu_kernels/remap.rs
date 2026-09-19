use crate::context::{BorderMode, Interpolation};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RemapParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    interpolation: u32,
    border_mode: u32,
    border_val: f32,
}

pub fn remap(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    map_x: &Tensor<f32, GpuStorage<f32>>,
    map_y: &Tensor<f32, GpuStorage<f32>>,
    interpolation: Interpolation,
    border_mode: BorderMode<f32>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (src_h, src_w) = input.shape.hw();
    let (dst_h, dst_w) = map_x.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Remap currently only for grayscale".into(),
        ));
    }

    let out_len = dst_w * dst_h;
    let byte_size = (out_len * 4) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let (b_mode, b_val) = match border_mode {
        BorderMode::Constant(v) => (0, v),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Wrap => (2, 0.0),
        BorderMode::Reflect => (3, 0.0),
        BorderMode::Reflect101 => (4, 0.0),
    };

    let params = RemapParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w: dst_w as u32,
        dst_h: dst_h as u32,
        interpolation: match interpolation {
            Interpolation::Nearest => 0,
            Interpolation::Linear => 1,
            Interpolation::Cubic => 2,
            Interpolation::Lanczos => 1, // Fallback
        },
        border_mode: b_mode,
        border_val: b_val,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Remap Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/remap.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Remap Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: map_x.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: map_y.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Remap Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (dst_w as u32).div_ceil(4).div_ceil(16);
        let y = (dst_h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: TensorShape::new(c, dst_h, dst_w),
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
