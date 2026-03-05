use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn spmv(
    ctx: &GpuContext,
    row_ptr: &[u32],
    col_indices: &[u32],
    values: &[f32],
    x: &Tensor<f32, GpuStorage<f32>>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let rows = row_ptr.len() - 1;
    let size = rows;
    let byte_size = (size * 4) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SpMV Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let row_ptr_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Row Ptr"),
            contents: bytemuck::cast_slice(row_ptr),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let col_indices_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Col Indices"),
            contents: bytemuck::cast_slice(col_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let values_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let shader_source = include_str!("../../shaders/spmv.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "spmv_main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SpMV Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: row_ptr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
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
        let x = (rows as u32).div_ceil(256);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), size),
        shape: cv_core::TensorShape::new(1, size, 1),
        dtype: cv_core::DataType::F32,
        _phantom: PhantomData,
    })
}
