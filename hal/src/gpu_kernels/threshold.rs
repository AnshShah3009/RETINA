use crate::context::ThresholdType;
use crate::gpu::GpuContext;
use crate::GpuTensor;
use crate::Result;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Copy, Clone)]
#[repr(C)]
struct ThresholdParams<T: bytemuck::Pod + bytemuck::Zeroable> {
    width: u32,
    height: u32,
    thresh: T,
    max_value: T,
    thresh_type: u32,
    len: u32,
}

unsafe impl<T: bytemuck::Pod + bytemuck::Zeroable> bytemuck::Pod for ThresholdParams<T> {}
unsafe impl<T: bytemuck::Pod + bytemuck::Zeroable> bytemuck::Zeroable for ThresholdParams<T> {}

pub fn threshold<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    thresh: T,
    max_value: T,
    thresh_type: ThresholdType,
) -> Result<GpuTensor<T>> {
    let len = input.shape.len();

    let byte_size = (len * std::mem::size_of::<T>()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Threshold Output Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Prepare params
    let params = ThresholdParams {
        width: input.shape.width as u32,
        height: input.shape.height as u32,
        thresh,
        max_value,
        thresh_type: thresh_type as u32,
        len: len as u32,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Threshold Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Pipeline setup
    let shader_source = match cv_core::DataType::from_type::<T>() {
        Ok(cv_core::DataType::F32) => include_str!("../../shaders/threshold_f32.wgsl"),
        Ok(_) => {
            return Err(crate::Error::NotSupported(
                "Unsupported threshold precision type".into(),
            ))
        }
        _ => {
            include_str!("../../shaders/threshold_f32.wgsl")
        }
    };
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Threshold Bind Group"),
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

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Threshold Dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Threshold Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = (len as u32).div_ceil(64);
        pass.dispatch_workgroups(wg_x, 1, 1);
    }

    ctx.submit(encoder);

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(output_buffer), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
