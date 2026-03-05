use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{KeyPointF32, Tensor};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BRIEFPoint {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BriefParams {
    width: u32,
    height: u32,
    num_keypoints: u32,
    num_pairs: u32,
}

pub fn compute_brief(
    ctx: &GpuContext,
    image: &Tensor<u8, GpuStorage<u8>>,
    keypoints: &[KeyPointF32],
    pattern: &[BRIEFPoint],
) -> Result<Vec<u8>> {
    if keypoints.is_empty() || pattern.is_empty() {
        return Ok(Vec::new());
    }

    let (h, w) = image.shape.hw();
    let num_kp = keypoints.len() as u32;
    let num_pairs = pattern.len() as u32;

    // Create buffers
    let kp_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brief Keypoints"),
            contents: bytemuck::cast_slice(keypoints),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let pattern_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brief Pattern"),
            contents: bytemuck::cast_slice(pattern),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let desc_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Brief Descriptors"),
        size: (num_kp as u64) * 32, // 256 bits = 32 bytes
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = BriefParams {
        width: w as u32,
        height: h as u32,
        num_keypoints: num_kp,
        num_pairs,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brief Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("brief.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Brief Bind Group"),
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
                resource: pattern_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: desc_buffer.as_entire_binding(),
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
        pass.dispatch_workgroups(num_kp.div_ceil(256), 1, 1);
    }
    ctx.submit(encoder);

    // Read back descriptors
    let descriptors = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &desc_buffer,
        0,
        (num_kp as usize) * 32,
    ))?;

    Ok(descriptors)
}
