use crate::context::StereoMatchParams;
use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use crate::Result;
use cv_core::Float;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StereoGpuParams {
    width: u32,
    height: u32,
    min_disparity: i32,
    num_disparities: u32,
    block_size: u32,
    method: u32,
}

pub fn stereo_match<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    left: &crate::GpuTensor<T>,
    right: &crate::GpuTensor<T>,
    params: &StereoMatchParams,
) -> Result<crate::GpuTensor<T>> {
    let _precision = crate::gpu_kernels::shader_template::precision_for_type::<T>()?;

    let (h, w) = left.shape.hw();
    let out_len = w * h;
    let byte_size = (out_len * std::mem::size_of::<f32>()) as u64;
    let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let gpu_params = StereoGpuParams {
        width: w as u32,
        height: h as u32,
        min_disparity: params.min_disparity,
        num_disparities: params.num_disparities as u32,
        block_size: params.block_size as u32,
        method: match params.method {
            crate::context::StereoMatchMethod::BlockMatching => 0,
            crate::context::StereoMatchMethod::SemiGlobalMatching => 1,
        },
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stereo Match Params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/stereo_match.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Stereo Match Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: left.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: right.storage.buffer().as_entire_binding(),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Stereo Match Dispatch"),
        });
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

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: left.shape,
        dtype: left.dtype,
        _phantom: PhantomData,
    })
}
