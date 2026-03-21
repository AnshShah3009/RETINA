use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::sync::Arc;
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FarnebackParams {
    width: u32,
    height: u32,
    stride: u32,
    flow_stride: u32,
    num_levels: u32,
    poly_n: u32,
    poly_sigma: f32,
    num_iters: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowBlurMode {
    Box,
    Gaussian,
}

pub fn farneback(
    ctx: &GpuContext,
    prev: &Tensor<f32, GpuStorage<f32>>,
    next: &Tensor<f32, GpuStorage<f32>>,
    num_levels: u32,
    pyr_scale: f32,
    _window_size: u32,
    num_iters: u32,
    poly_n: u32,
    poly_sigma: f32,
    _blur_mode: FlowBlurMode,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (height, width) = prev.shape.hw();
    let _stride = width as u32;

    let mut prev_pyr = Vec::new();
    let mut next_pyr = Vec::new();
    let mut current_prev = prev.clone();
    let mut current_next = next.clone();

    for level in 0..num_levels {
        let scale = pyr_scale.powi(level as i32);
        let _scaled_w = ((width as f32 * scale) as u32).max(1);
        let _scaled_h = ((height as f32 * scale) as u32).max(1);

        prev_pyr.push(current_prev.clone());
        next_pyr.push(current_next.clone());

        if level < num_levels - 1 {
            current_prev = crate::gpu_kernels::pyramid::pyramid_down(ctx, &current_prev)?;
            current_next = crate::gpu_kernels::pyramid::pyramid_down(ctx, &current_next)?;
        }
    }

    let mut flow: Option<Tensor<f32, GpuStorage<f32>>> = None;

    for level in (0..num_levels as usize).rev() {
        let level_w = prev_pyr[level].shape.width as u32;
        let level_h = prev_pyr[level].shape.height as u32;
        let level_stride = level_w;

        let flow_len = (level_w * level_h * 2) as usize;
        let flow_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Farneback Flow"),
            size: (flow_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if let Some(prev_flow) = flow {
            let flow_w = prev_flow.shape.width as u32;
            let flow_h = prev_flow.shape.height as u32;
            let _scale_x = level_w as f32 / flow_w as f32;
            let _scale_y = level_h as f32 / flow_h as f32;

            let up_shader = include_str!("../../shaders/flow_upsample.wgsl");
            let up_pipeline = ctx.create_compute_pipeline(up_shader, "main");

            let up_params = crate::gpu_kernels::resize::ResizeParams {
                src_w: flow_w,
                src_h: flow_h,
                dst_w: level_w,
                dst_h: level_h,
                channels: 2,
            };

            let up_params_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Flow Up Params"),
                    contents: bytemuck::bytes_of(&up_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let up_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flow Up Bind Group"),
                layout: &up_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: prev_flow.storage.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: flow_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: up_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut up_enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Flow Upsample"),
                });
            {
                let mut pass = up_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&up_pipeline);
                pass.set_bind_group(0, &up_bind_group, &[]);
                pass.dispatch_workgroups(level_w.div_ceil(16), level_h.div_ceil(16), 1);
            }
            ctx.submit(up_enc);
        }

        let poly_len = (level_w * level_h * 6) as usize;

        let poly1_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poly1"),
            size: (poly_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let poly2_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Poly2"),
            size: (poly_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let poly_params = FarnebackParams {
            width: level_w,
            height: level_h,
            stride: level_stride,
            flow_stride: level_w * 2,
            num_levels: 0,
            poly_n,
            poly_sigma,
            num_iters,
        };

        let poly_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Poly Params"),
                contents: bytemuck::bytes_of(&poly_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let poly_expand_shader = include_str!("../../shaders/farneback_poly_expand.wgsl");
        let poly_expand_pipeline = ctx.create_compute_pipeline(poly_expand_shader, "main");

        let prev_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Poly1 Bind Group"),
            layout: &poly_expand_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prev_pyr[level].storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: poly1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: poly_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut poly_enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Poly Expand"),
            });
        {
            let mut pass = poly_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&poly_expand_pipeline);
            pass.set_bind_group(0, &prev_bind_group, &[]);
            pass.dispatch_workgroups(level_w.div_ceil(16), level_h.div_ceil(16), 1);
        }
        ctx.submit(poly_enc);

        let next_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Poly2 Bind Group"),
            layout: &poly_expand_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: next_pyr[level].storage.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: poly2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: poly_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut poly_enc2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Poly Expand 2"),
            });
        {
            let mut pass = poly_enc2.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&poly_expand_pipeline);
            pass.set_bind_group(0, &next_bind_group, &[]);
            pass.dispatch_workgroups(level_w.div_ceil(16), level_h.div_ceil(16), 1);
        }
        ctx.submit(poly_enc2);

        let flow_update_shader = include_str!("../../shaders/farneback_update_flow.wgsl");
        let flow_update_pipeline = ctx.create_compute_pipeline(flow_update_shader, "main");

        let flow_params = FarnebackParams {
            width: level_w,
            height: level_h,
            stride: level_stride,
            flow_stride: level_w * 2,
            num_levels,
            poly_n,
            poly_sigma,
            num_iters,
        };

        let flow_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flow Update Params"),
                contents: bytemuck::bytes_of(&flow_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        for _iter in 0..num_iters {
            let flow_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flow Update Bind Group"),
                layout: &flow_update_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: poly1_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: poly2_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: flow_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: flow_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: flow_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut flow_enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Flow Update"),
                });
            {
                let mut pass = flow_enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&flow_update_pipeline);
                pass.set_bind_group(0, &flow_bind_group, &[]);
                pass.dispatch_workgroups(level_w.div_ceil(16), level_h.div_ceil(16), 1);
            }
            ctx.submit(flow_enc);
        }

        flow = Some(Tensor {
            storage: GpuStorage::from_buffer(Arc::new(flow_buffer), flow_len),
            shape: cv_core::TensorShape::new(2, level_h as usize, level_w as usize),
            dtype: prev.dtype,
            _phantom: std::marker::PhantomData,
        });
    }

    flow.ok_or_else(|| crate::Error::RuntimeError("No flow computed".into()))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Tvl1Params {
    width: u32,
    height: u32,
    tau: f32,
    lambda: f32,
    theta: f32,
    num_warps: u32,
    num_outer_iters: u32,
    num_inner_iters: u32,
}

pub struct Tvl1Config {
    pub tau: f32,
    pub lambda: f32,
    pub theta: f32,
    pub num_warps: u32,
    pub num_outer_iters: u32,
    pub num_inner_iters: u32,
}

impl Default for Tvl1Config {
    fn default() -> Self {
        Self {
            tau: 0.25,
            lambda: 0.15,
            theta: 0.3,
            num_warps: 5,
            num_outer_iters: 30,
            num_inner_iters: 1,
        }
    }
}

fn centered_gradient_kernel() -> &'static str {
    r#"
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input_img: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_dx: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_dy: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    let x_prev = select(x - 1u, 0u, x > 0u);
    let y_prev = select(y - 1u, 0u, y > 0u);
    let x_next = select(params.width - 1u, x + 1u, x >= params.width - 1u);
    let y_next = select(params.height - 1u, y + 1u, y >= params.height - 1u);
    
    let idx_prev_x = y * params.width + x_prev;
    let idx_next_x = y * params.width + x_next;
    let idx_prev_y = y_prev * params.width + x;
    let idx_next_y = y_next * params.width + x;
    
    let dx = (input_img[idx_next_x] - input_img[idx_prev_x]) * 0.5;
    let dy = (input_img[idx_next_y] - input_img[idx_prev_y]) * 0.5;
    
    output_dx[idx] = dx;
    output_dy[idx] = dy;
}
"#
}

fn tvl1_warp_kernel() -> &'static str {
    r#"
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input_img: array<f32>;
@group(0) @binding(1) var<storage, read> input_flow_x: array<f32>;
@group(0) @binding(2) var<storage, read> input_flow_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_warped: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    let flow_x = input_flow_x[idx];
    let flow_y = input_flow_y[idx];
    
    let src_x = f32(x) - flow_x;
    let src_y = f32(y) - flow_y;
    
    let clamped_x = clamp(src_x, 0.0, f32(params.width) - 1.001);
    let clamped_y = clamp(src_y, 0.0, f32(params.height) - 1.001);
    
    let x0 = i32(clamped_x);
    let y0 = i32(clamped_y);
    let x1 = min(x0 + 1, i32(params.width) - 1);
    let y1 = min(y0 + 1, i32(params.height) - 1);
    
    let fx = clamped_x - f32(x0);
    let fy = clamped_y - f32(y0);
    
    let idx00 = y0 * i32(params.width) + x0;
    let idx10 = y0 * i32(params.width) + x1;
    let idx01 = y1 * i32(params.width) + x0;
    let idx11 = y1 * i32(params.width) + x1;
    
    let v00 = input_img[idx00];
    let v10 = input_img[idx10];
    let v01 = input_img[idx01];
    let v11 = input_img[idx11];
    
    let warped = (v00 * (1.0 - fx) * (1.0 - fy) + 
                  v10 * fx * (1.0 - fy) + 
                  v01 * (1.0 - fx) * fy + 
                  v11 * fx * fy);
    
    output_warped[idx] = warped;
}
"#
}

fn tvl1_estimate_u_kernel() -> &'static str {
    r#"
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input_img: array<f32>;
@group(0) @binding(1) var<storage, read> input_I1w: array<f32>;
@group(0) @binding(2) var<storage, read> input_grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> input_output_u: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    let num_pixels = params.width * params.height;
    
    let I0 = input_img[idx];
    let I1w = input_I1w[idx];
    let grad_x = input_grad[idx];
    let grad_y = input_grad[num_pixels + idx];
    let u1 = input_output_u[idx];
    let u2 = input_output_u[num_pixels + idx];
    
    let grad_I1_sq = grad_x * grad_x + grad_y * grad_y;
    let rho_c = I1w - grad_x * u1 - grad_y * u2 - I0;
    
    let new_u1 = u1 + grad_x * rho_c / (1.0 + grad_I1_sq);
    let new_u2 = u2 + grad_y * rho_c / (1.0 + grad_I1_sq);
    
    input_output_u[idx] = new_u1;
    input_output_u[num_pixels + idx] = new_u2;
}
"#
}

fn estimate_dual_kernel() -> &'static str {
    r#"
struct Params {
    width: u32,
    height: u32,
    tau: f32,
    lambda_theta: f32,
}

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    let num_pixels = params.width * params.height;
    
    let u1 = inputs[idx];
    let u2 = inputs[num_pixels + idx];
    let v1 = inputs[2u * num_pixels + idx];
    let v2 = inputs[3u * num_pixels + idx];
    let grad = inputs[4u * num_pixels + idx];
    let rho = inputs[5u * num_pixels + idx];
    
    let px = min(x + 1u, params.width - 1u);
    let py = min(y + 1u, params.height - 1u);
    
    let du1dx = inputs[y * params.width + px] - u1;
    let du2dy = inputs[num_pixels + py * params.width + x] - u2;
    
    let l_t = params.lambda_theta;
    
    var rho_prime = 0.0;
    if (grad > 1e-6) {
        rho_prime = -rho / grad;
    }
    
    var d1 = 0.0;
    var d2 = 0.0;
    if (rho < -l_t * grad) {
        d1 = -l_t * v1;
        d2 = -l_t * v2;
    } else if (rho > l_t * grad) {
        d1 = l_t * v1;
        d2 = l_t * v2;
    } else if (grad > 1e-6) {
        d1 = rho_prime * v1;
        d2 = rho_prime * v2;
    }
    
    let new_p1 = v1 + d1;
    let new_p2 = v2 + d2;
    
    let norm = sqrt(new_p1 * new_p1 + new_p2 * new_p2);
    let shrink = select(params.tau / norm, params.tau, norm > 1e-6);
    
    let final_p1 = new_p1 * shrink;
    let final_p2 = new_p2 * shrink;
    
    outputs[idx] = final_p1;
    outputs[num_pixels + idx] = final_p2;
}
"#
}

fn divergence_kernel() -> &'static str {
    r#"
struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input_p1: array<f32>;
@group(0) @binding(1) var<storage, read> input_p2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_div: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    var div = 0.0;
    
    let mx = select(x - 1u, 0u, x == 0u);
    let my = select(y - 1u, 0u, y == 0u);
    
    let p1x = input_p1[idx];
    let p2y = input_p2[idx];
    
    if (x > 0u) {
        let p1x_prev = input_p1[y * params.width + mx];
        div = div + p1x - p1x_prev;
    } else {
        div = div + p1x;
    }
    
    if (y > 0u) {
        let p2y_prev = input_p2[my * params.width + x];
        div = div + p2y - p2y_prev;
    } else {
        div = div + p2y;
    }
    
    output_div[idx] = div;
}
"#
}

pub fn tvl1_optical_flow(
    ctx: &GpuContext,
    prev: &Tensor<f32, GpuStorage<f32>>,
    next: &Tensor<f32, GpuStorage<f32>>,
    config: Tvl1Config,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (height, width) = prev.shape.hw();
    let num_pixels = width * height;

    if num_pixels == 0 {
        return Err(crate::Error::InvalidInput("Empty input image".into()));
    }

    let byte_size = (num_pixels * std::mem::size_of::<f32>()) as u64;
    let buffer_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let prev_buffer = prev.storage.buffer();
    let next_buffer = next.storage.buffer();

    let flow_byte_size = byte_size * 2;
    let flow_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 Flow (packed u1,u2)"),
        size: flow_byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let grad_byte_size = byte_size * 2;
    let grad_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 Gradient (packed dx,dy)"),
        size: grad_byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let i1w_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 I1w A"),
        size: byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });
    let i1w_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 I1w B"),
        size: byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let dual_byte_size = byte_size * 6;
    let dual_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 Dual (packed p1,p2,v1,v2,grad,rho)"),
        size: dual_byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let div_byte_size = byte_size * 2;
    let div_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 Divergence (packed)"),
        size: div_byte_size,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct GradientParams {
        width: u32,
        height: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct DualParams {
        width: u32,
        height: u32,
        tau: f32,
        lambda_theta: f32,
    }

    let grad_params = GradientParams {
        width: width as u32,
        height: height as u32,
    };
    let grad_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gradient Params"),
            contents: bytemuck::bytes_of(&grad_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let dual_params = DualParams {
        width: width as u32,
        height: height as u32,
        tau: config.tau,
        lambda_theta: config.theta / config.tau,
    };
    let dual_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dual Params"),
            contents: bytemuck::bytes_of(&dual_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let gradient_pipeline = ctx.create_compute_pipeline(centered_gradient_kernel(), "main");
    let warp_pipeline = ctx.create_compute_pipeline(tvl1_warp_kernel(), "main");
    let estimate_u_pipeline = ctx.create_compute_pipeline(tvl1_estimate_u_kernel(), "main");
    let divergence_pipeline = ctx.create_compute_pipeline(divergence_kernel(), "main");
    let dual_pipeline = ctx.create_compute_pipeline(estimate_dual_kernel(), "main");

    let dispatch_w = (width as u32).div_ceil(16);
    let dispatch_h = (height as u32).div_ceil(16);

    let next_src = Arc::new(next_buffer.clone());
    let prev_src = Arc::new(prev_buffer.clone());

    let mut warp_read = i1w_a.clone();
    let mut warp_write = i1w_b.clone();

    for _outer_iter in 0..config.num_outer_iters {
        for _warp_iter in 0..config.num_warps {
            let warp_src = if _warp_iter == 0 && _outer_iter == 0 {
                next_src.as_ref()
            } else {
                &warp_read
            };

            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Warp BG"),
                    layout: &warp_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: warp_src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: flow_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: flow_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: warp_write.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: grad_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TVL1 Warp"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&warp_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_w, dispatch_h, 1);
                }
                ctx.submit(enc);
            }

            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Gradient BG"),
                    layout: &gradient_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: warp_write.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: grad_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: grad_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: grad_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TVL1 Gradient"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&gradient_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_w, dispatch_h, 1);
                }
                ctx.submit(enc);
            }

            std::mem::swap(&mut warp_read, &mut warp_write);
        }

        for _inner_iter in 0..config.num_inner_iters {
            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Divergence BG"),
                    layout: &divergence_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dual_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dual_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: div_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: grad_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TVL1 Divergence"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&divergence_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_w, dispatch_h, 1);
                }
                ctx.submit(enc);
            }

            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Estimate U BG"),
                    layout: &estimate_u_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: prev_src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: warp_read.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: grad_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: flow_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: grad_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TVL1 Estimate U"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&estimate_u_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_w, dispatch_h, 1);
                }
                ctx.submit(enc);
            }

            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Dual BG"),
                    layout: &dual_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: flow_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dual_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dual_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TVL1 Dual"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&dual_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_w, dispatch_h, 1);
                }
                ctx.submit(enc);
            }
        }
    }

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(flow_buffer), num_pixels * 2),
        shape: cv_core::TensorShape::new(2, height, width),
        dtype: prev.dtype,
        _phantom: std::marker::PhantomData,
    })
}
