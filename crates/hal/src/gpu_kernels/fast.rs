use crate::Result;
use crate::gpu::GpuContext;
#[cfg(feature = "half-precision")]
use crate::gpu_kernels::packed16::{self, Packed16};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Uniform shared by the detection and suppression passes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    width: u32,
    height: u32,
    /// As f32 rather than truncated to u32, which destroyed fractional values
    /// like 0.1. The f32 shaders declare it u32 and bitcast. For f16 / bf16 it
    /// is the threshold as that type, widened (exact).
    threshold: f32,
    /// The packed16 shaders' `format` (0: f16, 1: bf16); the f32 shaders'
    /// unused `nonmax`.
    mode: u32,
}

/// FAST-9 corner score map of a single-channel image, optionally with 3x3
/// non-maximum suppression.
///
/// f32 runs `fast_f32.wgsl` / `fast_nms_f32.wgsl`. f16 and bf16 run
/// `fast_packed16.wgsl` / `fast_nms_packed16.wgsl` on every device (no
/// `SHADER_F16` needed), which reproduce the CPU backend's per-operation
/// rounding, so their score maps are bit-identical to the CPU's for the same
/// type. Other types return `NotSupported`.
pub fn fast_detect<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    threshold: T,
    nonmax_suppression: bool,
) -> Result<crate::GpuTensor<T>> {
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "FAST detection expects 1 channel image".into(),
        ));
    }

    let out_len = w * h;
    let (detect_src, nms_src, mode, byte_size): (&str, &str, u32, u64) =
        match cv_core::DataType::from_type::<T>() {
            Ok(cv_core::DataType::F32) => (
                include_str!("../../shaders/fast_f32.wgsl"),
                include_str!("../../shaders/fast_nms_f32.wgsl"),
                u32::from(nonmax_suppression),
                (out_len * std::mem::size_of::<f32>()) as u64,
            ),
            #[cfg(feature = "half-precision")]
            Ok(cv_core::DataType::F16) => (
                include_str!("../../shaders/fast_packed16.wgsl"),
                include_str!("../../shaders/fast_nms_packed16.wgsl"),
                Packed16::F16.code(),
                packed16::buffer_bytes(out_len),
            ),
            #[cfg(feature = "half-precision")]
            Ok(cv_core::DataType::BF16) => (
                include_str!("../../shaders/fast_packed16.wgsl"),
                include_str!("../../shaders/fast_nms_packed16.wgsl"),
                Packed16::Bf16.code(),
                packed16::buffer_bytes(out_len),
            ),
            _ => {
                return Err(crate::Error::NotSupported(
                    "FAST GPU kernel supports f32, f16 and bf16".into(),
                ));
            }
        };

    // An empty image would bind a zero-size buffer; its score map is empty.
    if out_len == 0 {
        return Ok(super::zeros_tensor(ctx, input.shape, input.dtype));
    }

    let params = FastParams {
        width: w as u32,
        height: h as u32,
        threshold: threshold.to_f32(),
        mode,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FAST Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // The packed16 shaders OR each result into its half of a word, so both
    // outputs must start zeroed, which a freshly created wgpu buffer is.
    let new_output = |label| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };

    let score_buffer = new_output("FAST Output Buffer");
    run_pass(
        ctx,
        detect_src,
        input.storage.buffer(),
        &score_buffer,
        &params_buffer,
        (w as u32, h as u32),
    );

    let result_buffer = if nonmax_suppression {
        let nms_buffer = new_output("FAST NMS Output");
        run_pass(
            ctx,
            nms_src,
            &score_buffer,
            &nms_buffer,
            &params_buffer,
            (w as u32, h as u32),
        );
        nms_buffer
    } else {
        score_buffer
    };

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(Arc::new(result_buffer), out_len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData::<T>,
    })
}

/// Dispatch one 16x16-tiled pass of `shader_source` over a `w x h` image with
/// bindings (input, output, params).
fn run_pass(
    ctx: &GpuContext,
    shader_source: &str,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    params: &wgpu::Buffer,
    (w, h): (u32, u32),
) {
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FAST Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
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
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    ctx.submit(encoder);
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CollectParams {
    width: u32,
    height: u32,
    num_elements: u32,
    padding: u32,
}

/// Keypoints of an f32 score map from [`fast_detect`]. Other score types
/// return `NotSupported`: the collection shader reads f32 scores.
pub fn extract_keypoints<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    score_map: &crate::GpuTensor<T>,
) -> Result<Vec<cv_core::KeyPointF32>> {
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "GPU keypoint extraction reads f32 score maps only".into(),
        ));
    }
    let score_buffer = score_map.storage.buffer();
    let (h, w) = score_map.shape.hw();
    let num_pixels: u32 = (w * h) as u32;
    if num_pixels == 0 {
        return Ok(Vec::new());
    }

    // Bug 5 fix: Use the f32 collection shader that reads f32 scores directly,
    // instead of the packed-u8 collection shader. For f32 score maps, each
    // element is one pixel, so num_elements = num_pixels.
    let num_elements = num_pixels;

    // 1. Count pass
    //
    // `COPY_DST` is required as well as `COPY_SRC`: after the count pass this
    // same buffer is overwritten with the sorted indices, via
    // `Queue::write_buffer`, which wgpu rejects without it - "Usage flags
    // BufferUsages(COPY_SRC | STORAGE) ... do not contain required usage flags
    // BufferUsages(COPY_DST)". That turned every GPU ORB run into a panic
    // inside `Orb::detect_and_compute_ctx`, which is what the `orb_benchmark`
    // example exercises.
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let counts_buffer = ctx.get_buffer((num_elements as u64) * 4, usages);

    let params = CollectParams {
        width: w as u32,
        height: h as u32,
        num_elements,
        padding: 0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Collect Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("feature_collection_f32.wgsl");
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feature Collection F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let count_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Count Points Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("count_points"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Count Points"),
        });
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Count Bind Group"),
            layout: &count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&count_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (wg_x, wg_y, wg_z) = crate::gpu_kernels::dispatch_1d(num_elements, 256);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }
    ctx.submit(encoder);

    // 2. Scan pass
    let counts_data: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &counts_buffer,
        0,
        (num_elements as usize) * 4,
    ))?;

    let mut total = 0u32;
    let mut indices = Vec::with_capacity(num_elements as usize);
    for &c in &counts_data {
        indices.push(total);
        // f32 collection: each element is 0 or 1
        if c <= 1 {
            total += c;
        }
    }

    if total == 0 {
        ctx.return_buffer(counts_buffer, usages);
        return Ok(Vec::new());
    }

    if total > 1000000 {
        ctx.return_buffer(counts_buffer, usages);
        return Err(crate::Error::InvalidInput(format!(
            "Too many keypoints detected: {}",
            total
        )));
    }

    // Upload scanned indices
    ctx.queue
        .write_buffer(&counts_buffer, 0, bytemuck::cast_slice(&indices));

    // 3. Collect pass
    let kp_size = std::mem::size_of::<cv_core::KeyPointF32>();
    let kp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extracted Keypoints"),
        size: (total as u64) * kp_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let collect_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collect Points Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("collect_points"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Collect Points"),
        });
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collect Bind Group"),
            layout: &collect_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: score_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&collect_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (wg_x, wg_y, wg_z) = crate::gpu_kernels::dispatch_1d(num_elements, 256);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }
    ctx.submit(encoder);

    // Read back keypoints
    let kps = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &kp_buffer,
        0,
        (total as usize) * kp_size,
    ))?;

    ctx.return_buffer(counts_buffer, usages);

    Ok(kps)
}
