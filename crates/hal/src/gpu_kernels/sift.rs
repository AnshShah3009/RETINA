use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SiftExtremaParams {
    width: u32,
    height: u32,
    threshold: f32,
    edge_threshold: f32,
}

pub fn sift_extrema(
    ctx: &GpuContext,
    dog_prev: &Tensor<f32, GpuStorage<f32>>,
    dog_curr: &Tensor<f32, GpuStorage<f32>>,
    dog_next: &Tensor<f32, GpuStorage<f32>>,
    threshold: f32,
    edge_threshold: f32,
) -> Result<Tensor<u8, GpuStorage<u8>>> {
    let (h, w) = dog_curr.shape.hw();
    // The shader packs 4 pixels per u32, writing with row stride ceil(w/4).
    // Total u32 elements = h * ceil(w/4), total bytes = h * ceil(w/4) * 4.
    let row_stride_u32 = w.div_ceil(4);
    let total_u32 = h * row_stride_u32;
    let byte_size = (total_u32 * 4) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SIFT Extrema Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = SiftExtremaParams {
        width: w as u32,
        height: h as u32,
        threshold,
        edge_threshold,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SIFT Extrema Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/sift_extrema.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SIFT Extrema Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dog_prev.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dog_curr.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dog_next.storage.buffer().as_entire_binding(),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (w as u32).div_ceil(4).div_ceil(16);
        let y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    // Store the total bytes (total_u32 * 4) as the storage length so readback
    // retrieves the full packed buffer. Use a 1-D shape for the packed data.
    let total_bytes = total_u32 * 4;
    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), total_bytes),
        shape: cv_core::TensorShape::new(1, h, row_stride_u32 * 4),
        dtype: cv_core::DataType::U8,
        _phantom: PhantomData,
    })
}
