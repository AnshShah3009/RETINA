use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{KeyPointF32, Tensor};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OrientationParams {
    width: u32,
    height: u32,
    num_keypoints: u32,
    radius: i32,
}

pub fn compute_orientation(
    ctx: &GpuContext,
    image: &Tensor<u8, GpuStorage<u8>>,
    keypoints: &[KeyPointF32],
    radius: i32,
) -> Result<Vec<f32>> {
    if keypoints.is_empty() {
        return Ok(Vec::new());
    }

    let (h, w) = image.shape.hw();
    let num_kp = keypoints.len() as u32;

    // Create buffers
    let kp_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Orientation Keypoints"),
            contents: bytemuck::cast_slice(keypoints),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let angle_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Orientation Angles"),
        size: (num_kp as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = OrientationParams {
        width: w as u32,
        height: h as u32,
        num_keypoints: num_kp,
        radius,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Orientation Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("orientation.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Orientation Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: image.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kp_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: angle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
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
        pass.dispatch_workgroups(num_kp, 1, 1);
    }
    ctx.submit(encoder);

    // Read back results
    let angles = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &angle_buffer,
        0,
        (num_kp as usize) * 4,
    ))?;

    Ok(angles)
}
