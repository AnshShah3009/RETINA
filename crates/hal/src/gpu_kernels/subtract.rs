use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;

pub fn subtract(
    ctx: &GpuContext,
    a: &Tensor<f32, GpuStorage<f32>>,
    b: &Tensor<f32, GpuStorage<f32>>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    assert_eq!(a.shape, b.shape);
    let size = a.shape.len();
    let byte_size = (size * std::mem::size_of::<f32>()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Subtract Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader_source = include_str!("../../shaders/subtract.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Subtract Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
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
        let workgroups = (size as u32).div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), size),
        shape: a.shape,
        dtype: a.dtype,
        _phantom: PhantomData,
    })
}
