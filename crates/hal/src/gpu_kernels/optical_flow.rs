use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LkParams {
    num_points: u32,
    window_radius: i32,
    max_iters: u32,
    min_eigenvalue: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Point {
    x: f32,
    y: f32,
}

pub fn lucas_kanade(
    ctx: &GpuContext,
    prev_pyramid: &[Tensor<f32, GpuStorage<f32>>],
    next_pyramid: &[Tensor<f32, GpuStorage<f32>>],
    points: &[[f32; 2]],
    window_size: usize,
    max_iters: u32,
) -> Result<Vec<[f32; 2]>> {
    let num_points = points.len();
    if num_points == 0 {
        return Ok(Vec::new());
    }
    if prev_pyramid.is_empty() || next_pyramid.is_empty() {
        return Err(crate::Error::InvalidInput(
            "Pyramids cannot be empty".into(),
        ));
    }
    let levels = prev_pyramid.len();
    if next_pyramid.len() != levels {
        return Err(crate::Error::InvalidInput("Pyramid level mismatch".into()));
    }

    // Initialize points at the coarsest level
    // We start processing at level (levels-1)
    // The input points are in level 0 coordinates.
    // We need to scale them down to the start level.
    let scale = 1.0 / (1 << (levels - 1)) as f32;
    let initial_points: Vec<Point> = points
        .iter()
        .map(|p| Point {
            x: p[0] * scale,
            y: p[1] * scale,
        })
        .collect();

    // Two buffers for ping-ponging point data
    // Buffer A: Input guess
    // Buffer B: Output refined
    let buffer_size = (num_points * 8) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let buffer_a = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LK Points A"),
            contents: bytemuck::cast_slice(&initial_points),
            usage,
        });

    let buffer_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LK Points B"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });

    let shader_source = include_str!("../../shaders/lucas_kanade.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let params = LkParams {
        num_points: num_points as u32,
        window_radius: (window_size / 2) as i32,
        max_iters,
        min_eigenvalue: 0.001,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LK Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    for level in (0..levels).rev() {
        let prev_img = &prev_pyramid[level];
        let next_img = &next_pyramid[level];
        let (w, h) = prev_img.shape.hw();

        let level_params = [w as u32, h as u32, 0, 0]; // offset is 0 as we use separate tensors
        let level_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LK Level Params"),
                contents: bytemuck::cast_slice(&level_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LK Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prev_img.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: next_img.storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: level_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LK Dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_points as u32).div_ceil(64), 1, 1);
        }

        // If not the last level (0), we need to scale up the results for the next iteration
        if level > 0 {
            // We can do this scaling in a simple compute shader or just map/write.
            // For efficiency, we should have a 'scale_points' shader.
            // For now, I'll use a copy command and rely on the fact that next iteration starts with `buffer_b` as input.
            // WAIT: The LK shader outputs the refined position at *current* level.
            // The input for *next* level (level-1) needs to be `current_output * 2.0`.
            // Our shader doesn't scale output. We need to scale it.

            // To avoid CPU roundtrip, we'll dispatch a scaling kernel.
            // Actually, let's just make the LK shader take a 'scale_input' param?
            // No, the input to level L is the output of L+1 scaled up.
            // Let's modify the loop structure to scale *after* processing, or make the shader handle the 2x guess?

            // Simplest approach: Add a 'next_level_guess_scale' to the shader?
            // Or just a tiny kernel "scale_points".
            // Since I cannot easily add a new file right here without breaking flow, I will do a CPU roundtrip for scaling
            // OR reuse the buffer copy.

            // Let's be performant: CPU roundtrip is bad.
            // I'll add a 'scale_points' compute pipeline right here using inline WGSL source string.
        }

        ctx.submit(encoder);

        if level > 0 {
            // We need to scale buffer_b values by 2.0 and put them into buffer_a (or swap and scale in place).
            // I'll implement a quick inline scaler.
            let scale_shader = r#"
                struct Point { x: f32, y: f32 }
                @group(0) @binding(0) var<storage, read> input: array<Point>;
                @group(0) @binding(1) var<storage, read_write> output: array<Point>;
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let idx = id.x;
                    if (idx >= arrayLength(&input)) { return; }
                    let p = input[idx];
                    output[idx] = Point(p.x * 2.0, p.y * 2.0);
                }
             "#;
            let scale_module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Scaler"),
                    source: wgpu::ShaderSource::Wgsl(scale_shader.into()),
                });
            let scale_pipeline =
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Scaler"),
                        layout: None,
                        module: &scale_module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            let scale_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &scale_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer_a.as_entire_binding(),
                    },
                ],
            });
            let mut scale_enc =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Scale"),
                    });
            {
                let mut pass =
                    scale_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&scale_pipeline);
                pass.set_bind_group(0, &scale_bg, &[]);
                pass.dispatch_workgroups((num_points as u32).div_ceil(64), 1, 1);
            }
            ctx.submit(scale_enc);

            // buffer_a now has the scaled guess for the next level.
            // Loop continues with buffer_a as input.
        } else {
            // Level 0 finished. buffer_b has the final result.
        }
    }

    let result_vec: Vec<Point> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &buffer_b, // Final result is in B
            0,
            num_points * 8,
        ))?;

    Ok(result_vec.iter().map(|p| [p.x, p.y]).collect())
}
