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

    // A colour image is invalid input, as on the CPU, not an unsupported
    // variant: FAST scores one plane.
    if c != 1 {
        return Err(crate::Error::InvalidInput(
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

/// Uniform of `feature_compact_f32.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompactParams {
    width: u32,
    height: u32,
    num_elements: u32,
    num_blocks: u32,
}

/// Pixels per workgroup of `feature_compact_f32.wgsl` (`BLOCK` there).
const COMPACT_BLOCK: u32 = 1024;

/// Most keypoints [`extract_keypoints`] returns before reporting an error.
const MAX_KEYPOINTS: u32 = 1_000_000;

/// Keypoints of an f32 score map from [`fast_detect`]: one per pixel with a
/// score above zero, in row-major pixel order. Other score types return
/// `NotSupported`: the compaction shader reads f32 scores.
///
/// The compaction runs on the GPU (`feature_compact_f32.wgsl`: per-block
/// counts, a scan of the block counts, a scatter); only the keypoint total and
/// the keypoints themselves are read back. It used to read back a count per
/// pixel (8 MB at 1080p), scan it on the CPU and upload the offsets again.
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
    let num_pixels = w * h;
    if num_pixels == 0 {
        return Ok(Vec::new());
    }
    let num_elements = u32::try_from(num_pixels)
        .map_err(|_| crate::Error::InvalidInput(format!("score map {w}x{h} is too large")))?;
    let num_blocks = num_elements.div_ceil(COMPACT_BLOCK);

    let src = include_str!("feature_compact_f32.wgsl");
    let count_pipeline = ctx.create_compute_pipeline(src, "block_counts");
    let scan_pipeline = ctx.create_compute_pipeline(src, "scan_blocks");

    // Block counts, turned in place into exclusive offsets, then the total at
    // [num_blocks].
    let usages = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let blocks_buffer = ctx.get_buffer((num_blocks as u64 + 1) * 4, usages);

    let params = CompactParams {
        width: w as u32,
        height: h as u32,
        num_elements,
        num_blocks,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compact Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind = |pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]| {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compact Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    };
    let (gx, gy, gz) = crate::gpu_kernels::dispatch_groups_1d(num_blocks);

    let count_bg = bind(
        &count_pipeline,
        &[score_buffer, &blocks_buffer, &params_buffer],
    );
    let scan_bg = bind(&scan_pipeline, &[&blocks_buffer, &params_buffer]);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Count Points"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&count_pipeline);
        pass.set_bind_group(0, &count_bg, &[]);
        pass.dispatch_workgroups(gx, gy, gz);
        pass.set_pipeline(&scan_pipeline);
        pass.set_bind_group(0, &scan_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.submit(encoder);

    let total: Vec<u32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &blocks_buffer,
        num_blocks as u64 * 4,
        4,
    ))?;
    let total = total[0];

    if total == 0 {
        ctx.return_buffer(blocks_buffer, usages);
        return Ok(Vec::new());
    }

    if total > MAX_KEYPOINTS {
        ctx.return_buffer(blocks_buffer, usages);
        return Err(crate::Error::InvalidInput(format!(
            "Too many keypoints detected: {}",
            total
        )));
    }

    let kp_size = std::mem::size_of::<cv_core::KeyPointF32>();
    let kp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extracted Keypoints"),
        size: (total as u64) * kp_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let scatter_pipeline = ctx.create_compute_pipeline(src, "scatter");
    let scatter_bg = bind(
        &scatter_pipeline,
        &[score_buffer, &blocks_buffer, &kp_buffer, &params_buffer],
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Collect Points"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&scatter_pipeline);
        pass.set_bind_group(0, &scatter_bg, &[]);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    ctx.submit(encoder);

    let kps = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &kp_buffer,
        0,
        (total as usize) * kp_size,
    ))?;

    ctx.return_buffer(blocks_buffer, usages);

    Ok(kps)
}
