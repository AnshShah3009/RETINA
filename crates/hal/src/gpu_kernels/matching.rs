use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{FeatureMatch, Matches, Tensor};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatchingParams {
    query_len: u32,
    train_len: u32,
    desc_size: u32,
    ratio_threshold: f32,
}

pub fn match_descriptors(
    ctx: &GpuContext,
    query: &Tensor<u8, GpuStorage<u8>>,
    train: &Tensor<u8, GpuStorage<u8>>,
    ratio_threshold: f32,
) -> Result<Matches> {
    let q_len = query.shape.height;
    let t_len = train.shape.height;
    let d_size = query.shape.width;

    if q_len == 0 || t_len == 0 {
        return Ok(Matches {
            matches: Vec::new(),
            mask: None,
        });
    }

    // Output buffer: one vec4<f32> per query point
    let byte_size = (q_len * 16) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matching Output"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = MatchingParams {
        query_len: q_len as u32,
        train_len: t_len as u32,
        desc_size: d_size as u32,
        ratio_threshold,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matching Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/matching.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matching Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: query.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: train.storage.buffer().as_entire_binding(),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = (q_len as u32).div_ceil(64);
        pass.dispatch_workgroups(x, 1, 1);
    }
    ctx.submit(encoder);

    // Read back and filter by ratio test on host
    let raw_matches: Vec<[f32; 4]> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &output_buffer,
            0,
            byte_size as usize,
        ))?;

    let mut matches = Vec::new();
    for (i, res) in raw_matches.iter().enumerate() {
        let train_idx = res[0] as i32;
        let dist = res[1];
        let second_dist = res[2];

        if dist <= ratio_threshold * second_dist {
            matches.push(FeatureMatch::new(i as i32, train_idx, dist));
        }
    }

    Ok(Matches {
        matches,
        mask: None,
    })
}
