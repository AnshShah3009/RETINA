use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SobelParams {
    width: u32,
    height: u32,
    ksize: u32,
    border_mode: u32,
}

pub fn sobel(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    dx: i32,
    dy: i32,
    ksize: usize,
) -> Result<(Tensor<u8, GpuStorage<u8>>, Tensor<u8, GpuStorage<u8>>)> {
    if dx != 1 || dy != 1 || ksize != 3 {
        return Err(crate::Error::NotSupported(
            "GPU Sobel only supports dx=1, dy=1, ksize=3 currently".into(),
        ));
    }

    let len = input.shape.len();
    let (h, w) = input.shape.hw();
    let byte_size = (len.div_ceil(4) * 4) as u64;

    // Output buffers
    let gx_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sobel Gx"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let gy_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sobel Gy"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = SobelParams {
        width: w as u32,
        height: h as u32,
        ksize: ksize as u32,
        border_mode: 1, // Replicate
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sobel Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/sobel.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sobel Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gy_buffer.as_entire_binding(),
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
            label: Some("Sobel Dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = (w as u32).div_ceil(4).div_ceil(16);
        let wg_y = (h as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    ctx.submit(encoder);

    Ok((
        Tensor {
            storage: GpuStorage::from_buffer(Arc::new(gx_buffer), len),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: PhantomData,
        },
        Tensor {
            storage: GpuStorage::from_buffer(Arc::new(gy_buffer), len),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: PhantomData,
        },
    ))
}
