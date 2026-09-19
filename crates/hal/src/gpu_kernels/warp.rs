use crate::context::WarpType;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{Tensor, TensorShape};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    warp_type: u32,
}

pub fn warp(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    matrix: &[[f32; 3]; 3],
    new_shape: (usize, usize),
    typ: WarpType,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (src_h, src_w) = input.shape.hw();
    let (dst_w, dst_h) = new_shape;
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Warp currently only for grayscale".into(),
        ));
    }

    let out_len = dst_w * dst_h;
    let byte_size = (out_len * 4) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let params = WarpParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w: dst_w as u32,
        dst_h: dst_h as u32,
        warp_type: match typ {
            WarpType::Affine => 0,
            WarpType::Perspective => 1,
        },
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Warp Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // We need to flatten the 3x3 matrix for the shader
    // WGSL mat3x3 expects column-major or 12-byte aligned rows?
    // Safer to pass as array of 9 floats and manually index or use mat3x3 with padding.
    // Standard mat3x3 in WGSL is actually 3x vec3, where each vec3 is 16-byte aligned.
    // So 3 * 16 = 48 bytes.
    let mut matrix_data = [0.0f32; 12]; // 3 columns * 4 floats (for vec3 alignment)
    for c in 0..3 {
        for r in 0..3 {
            matrix_data[c * 4 + r] = matrix[r][c]; // Column-major
        }
    }

    let matrix_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Warp Matrix"),
            contents: bytemuck::cast_slice(&matrix_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/warp.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Warp Bind Group"),
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
                resource: matrix_buffer.as_entire_binding(),
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
            label: Some("Warp Dispatch"),
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
