use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use crate::Result;
use cv_core::Float;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CannyParams {
    width: u32,
    height: u32,
    low_threshold: f32,
    high_threshold: f32,
}

/// Perform proper queue-based hysteresis thresholding on CPU.
///
/// Strong edges (>= high) are immediately accepted.  Weak edges (>= low, < high)
/// are accepted only if they are connected (8-neighbourhood) to a strong edge.
/// Connectivity is resolved via iterative flood-fill from strong edges.
fn cpu_hysteresis(width: usize, height: usize, nms: &[f32], low: f32, high: f32) -> Vec<f32> {
    const STRONG: u8 = 2;
    const WEAK: u8 = 1;

    let len = width * height;
    let mut state = vec![0u8; len];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    // First pass: classify pixels and seed the stack with strong edges
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let v = nms[idx];
            if v >= high {
                state[idx] = STRONG;
                stack.push((x, y));
            } else if v >= low {
                state[idx] = WEAK;
            }
        }
    }

    // Second pass: flood-fill from strong edges to connected weak edges
    while let Some((x, y)) = stack.pop() {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height - 1);
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(width - 1);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = ny * width + nx;
                if state[nidx] == WEAK {
                    state[nidx] = STRONG;
                    stack.push((nx, ny));
                }
            }
        }
    }

    // Convert state map to f32 output: STRONG → 255.0, everything else → 0.0
    let mut output = vec![0.0f32; len];
    for i in 0..len {
        if state[i] == STRONG {
            output[i] = 255.0;
        }
    }
    output
}

pub fn canny<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    low_threshold: T,
    high_threshold: T,
) -> Result<crate::GpuTensor<T>> {
    // Only f32 shader is available; return NotSupported for other types
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Canny GPU kernel only supports f32; other types not yet implemented".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let len = input.shape.len();
    let byte_size_f32 = (len * 4) as u64;

    // 1. Intermediate buffers (pooled)
    let mag_buffer = ctx.get_buffer(byte_size_f32, wgpu::BufferUsages::STORAGE);
    let dir_buffer = ctx.get_buffer(byte_size_f32, wgpu::BufferUsages::STORAGE);
    let nms_buffer = ctx.get_buffer(
        byte_size_f32,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    let params = CannyParams {
        width: w as u32,
        height: h as u32,
        low_threshold: low_threshold.to_f32(),
        high_threshold: high_threshold.to_f32(),
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/canny.wgsl");
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Canny Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let gradients_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Canny Gradients"),
            layout: None,
            module: &shader_module,
            entry_point: Some("gradients"),
            compilation_options: Default::default(),
            cache: None,
        });

    let nms_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Canny NMS"),
            layout: None,
            module: &shader_module,
            entry_point: Some("nms"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Pass 1: Gradients (GPU)
    let bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny BG 1"),
        layout: &gradients_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mag_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dir_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Pass 2: NMS (GPU)
    let bind_group_2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny BG 2"),
        layout: &nms_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: mag_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dir_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: nms_buffer.as_entire_binding(),
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
            label: Some("Canny Dispatch (Gradients + NMS)"),
        });
    let wg_x = (w as u32).div_ceil(16);
    let wg_y = (h as u32).div_ceil(16);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gradients"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&gradients_pipeline);
        pass.set_bind_group(0, &bind_group_1, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("NMS"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&nms_pipeline);
        pass.set_bind_group(0, &bind_group_2, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    ctx.submit(encoder);

    // Pass 3: Hysteresis — CPU fallback for proper multi-pass connectivity.
    //
    // The GPU hysteresis shader only checks immediate 8-neighbours for strong
    // edges (1-level connectivity), which misses weak edges connected to strong
    // edges through chains of other weak edges.  True iterative hysteresis
    // requires multiple dispatch rounds or atomic flag propagation on GPU.
    //
    // Instead, we download the NMS result to CPU, run a queue-based flood-fill
    // from strong edges through connected weak edges, and upload the result back.

    let nms_byte_size = len * std::mem::size_of::<f32>();
    let nms_data: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &nms_buffer,
        0,
        nms_byte_size,
    ))?;

    // Return intermediate buffers to pool
    ctx.return_buffer(mag_buffer, wgpu::BufferUsages::STORAGE);
    ctx.return_buffer(dir_buffer, wgpu::BufferUsages::STORAGE);
    ctx.return_buffer(
        nms_buffer,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    // Run proper iterative hysteresis on CPU
    let low_f32 = low_threshold.to_f32();
    let high_f32 = high_threshold.to_f32().max(low_f32);
    let result_data = cpu_hysteresis(w, h, &nms_data, low_f32, high_f32);

    // Upload the hysteresis result back to GPU
    let final_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Final (hysteresis result)"),
            contents: bytemuck::cast_slice(&result_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(final_buffer), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
