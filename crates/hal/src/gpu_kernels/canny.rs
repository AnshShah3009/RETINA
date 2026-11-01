//! GPU Canny, bit-exact with `cv::Canny` and with the CPU reference in
//! [`crate::cpu::canny_opencv`].
//!
//! Sobel, magnitude, non-maximum suppression and the double threshold run on
//! the GPU (`shaders/canny.wgsl`, all integer arithmetic); the 8-connected
//! hysteresis flood fill runs on the host over the downloaded state map (one
//! byte per pixel), with the same function the CPU path uses. A GPU flood fill
//! needs one dispatch per step of the longest weak chain, which on real images
//! costs more than the download.

use crate::Result;
use crate::cpu::canny_opencv::{canny_hysteresis, canny_tg22, canny_thresholds};
use crate::gpu::GpuContext;
use crate::storage::WgpuGpuStorage;
use cv_core::Float;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CannyParams {
    width: u32,
    height: u32,
    low: i32,
    high: i32,
    l2: u32,
    tg22: i32,
    _pad0: u32,
    _pad1: u32,
}

/// Run Sobel + NMS + double threshold on the GPU over an f32 image buffer of
/// `w * h` pixels and return the state map (0 none, 1 weak, 2 strong).
fn canny_states(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    w: usize,
    h: usize,
    low: f64,
    high: f64,
    l2: bool,
) -> Result<Vec<u8>> {
    let n = w * h;
    let (lo, hi) = canny_thresholds(low, high, l2);
    let clamp_i32 = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let params = CannyParams {
        width: w as u32,
        height: h as u32,
        low: clamp_i32(lo),
        high: clamp_i32(hi),
        l2: l2 as u32,
        tg22: canny_tg22() as i32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let grad_usage = wgpu::BufferUsages::STORAGE;
    let state_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let words = n.div_ceil(4);
    let grad_buffer = ctx.get_buffer((n * 4) as u64, grad_usage);
    let state_buffer = ctx.get_buffer((words * 4) as u64, state_usage);

    let source = include_str!("../../shaders/canny.wgsl");
    let sobel_pipeline = ctx.create_compute_pipeline(source, "sobel");
    let nms_pipeline = ctx.create_compute_pipeline(source, "nms");

    let bg_sobel = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny Sobel BG"),
        layout: &sobel_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    let bg_nms = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Canny NMS BG"),
        layout: &nms_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_buffer.as_entire_binding(),
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
            label: Some("Canny"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Canny Sobel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&sobel_pipeline);
        pass.set_bind_group(0, &bg_sobel, &[]);
        let (x, y, z) = crate::gpu_kernels::dispatch_1d(n as u32, 256);
        pass.dispatch_workgroups(x, y, z);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Canny NMS"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&nms_pipeline);
        pass.set_bind_group(0, &bg_nms, &[]);
        let (x, y, z) = crate::gpu_kernels::dispatch_1d(words as u32, 256);
        pass.dispatch_workgroups(x, y, z);
    }
    ctx.submit(encoder);

    let packed: Result<Vec<u32>> =
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            &state_buffer,
            0,
            words * 4,
        ));
    ctx.return_buffer(grad_buffer, grad_usage);
    ctx.return_buffer(state_buffer, state_usage);
    let packed = packed?;

    let mut states: Vec<u8> = bytemuck::cast_slice::<u32, u8>(&packed).to_vec();
    if cfg!(target_endian = "big") {
        for c in states.chunks_mut(4) {
            c.reverse();
        }
    }
    states.truncate(n);
    Ok(states)
}

/// `cv2.Canny(src, low, high, L2gradient=l2)` on a row-major `w x h` 8-bit
/// image; returns the 0/255 edge mask, bit-identical to
/// [`crate::cpu::canny_opencv::canny_u8`].
pub fn canny_u8(
    ctx: &GpuContext,
    src: &[u8],
    w: usize,
    h: usize,
    low: f64,
    high: f64,
    l2: bool,
) -> Result<Vec<u8>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    if src.len() != w * h {
        return Err(crate::Error::InvalidInput(format!(
            "canny_u8: buffer has {} bytes, expected {w} x {h}",
            src.len()
        )));
    }
    let px: Vec<f32> = src.iter().map(|&p| p as f32).collect();
    let input = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Input"),
            contents: bytemuck::cast_slice(&px),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut states = canny_states(ctx, &input, w, h, low, high, l2)?;
    Ok(canny_hysteresis(&mut states, w, h))
}

/// `ComputeContext::canny` on the GPU: L1 gradient, OpenCV semantics on the
/// input rounded and saturated to 8 bits - the same as the CPU backend.
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
    if input.shape.channels != 1 {
        return Err(crate::Error::InvalidInput(
            "Canny requires a single-channel image".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let len = input.shape.len();
    if len == 0 {
        return Ok(input.clone());
    }
    let mask = {
        let mut states = canny_states(
            ctx,
            input.storage.buffer(),
            w,
            h,
            low_threshold.to_f32() as f64,
            high_threshold.to_f32() as f64,
            false,
        )?;
        canny_hysteresis(&mut states, w, h)
    };
    let result: Vec<f32> = mask.iter().map(|&m| m as f32).collect();

    let final_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Canny Result"),
            contents: bytemuck::cast_slice(&result),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

    Ok(cv_core::Tensor {
        storage: WgpuGpuStorage::from_buffer(Arc::new(final_buffer), len),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
