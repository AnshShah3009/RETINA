use crate::gpu::GpuContext;
use crate::GpuTensor;
use crate::Result;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters for the pointcloud transform kernel.
/// Must match the WGSL `Params` struct layout exactly (std140).
#[derive(Copy, Clone)]
#[repr(C)]
struct TransformParams {
    num_points: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // mat4x4<f32> in WGSL std140: 4 columns of vec4<f32>, each 16-byte aligned
    transform: [[f32; 4]; 4],
}

unsafe impl bytemuck::Pod for TransformParams {}
unsafe impl bytemuck::Zeroable for TransformParams {}

/// Apply a 4x4 transformation matrix to a point cloud on the GPU.
///
/// Points are stored as Nx4 (x, y, z, w) where w is typically 1.0.
/// The transform is applied as: p' = M * p for each point.
pub fn pointcloud_transform(
    ctx: &GpuContext,
    input: &GpuTensor<f32>,
    transform: &[[f32; 4]; 4],
) -> Result<GpuTensor<f32>> {
    let num_points = input.shape.height as u32;
    let len = input.shape.len();
    let byte_size = (len * std::mem::size_of::<f32>()) as u64;

    // Create output buffer
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PointCloud Transform Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // WGSL mat4x4 is column-major. Our [[f32; 4]; 4] from Rust is row-major
    // (transform[row][col]). WGSL `mat4x4<f32>` stores columns, so we must transpose.
    let params = TransformParams {
        num_points,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        transform: [
            [transform[0][0], transform[1][0], transform[2][0], transform[3][0]],
            [transform[0][1], transform[1][1], transform[2][1], transform[3][1]],
            [transform[0][2], transform[1][2], transform[2][2], transform[3][2]],
            [transform[0][3], transform[1][3], transform[2][3], transform[3][3]],
        ],
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PointCloud Transform Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create pipeline from shader
    let shader_source = include_str!("../../shaders/pointcloud_transform.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    // Bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PointCloud Transform Bind Group"),
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
            label: Some("PointCloud Transform Dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PointCloud Transform Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = num_points.div_ceil(256);
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
