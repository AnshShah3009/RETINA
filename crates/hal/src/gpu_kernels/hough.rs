//! GPU standard Hough lines with a bit-identical accumulator.
//!
//! Voting runs on the GPU (`shaders/hough.wgsl`); the few votes whose f32 bin
//! value lies within rounding noise of a bin boundary are cast on the host
//! with the caller's exact CPU formula (see the shader header), so the
//! accumulator equals the CPU one exactly. Peak extraction then runs on the
//! host with the caller's CPU code, which makes the reported lines identical.
//!
//! [`hough_lines`] is `ComputeContext::hough_lines` for the GPU backend and
//! follows the HAL convention in [`crate::cpu::hough_standard`];
//! [`hough_accumulate_u8`] lets other conventions (cv-imgproc's OpenCV port)
//! supply their own tables and exact bin function.

use crate::cpu::hough_standard::HoughLayout;
use crate::gpu::GpuContext;
use crate::Result;
use cv_core::{Float, HoughLine};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HoughParams {
    width: u32,
    height: u32,
    num_rho: u32,
    num_angle: u32,
    rho_offset: i32,
    ambiguous_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Capacity of the ambiguous-vote list. Typical images produce a handful;
/// on overflow the whole accumulator is recomputed on the host.
const AMBIGUOUS_CAPACITY: u32 = 1 << 20;

/// Exact accumulator index of pixel (x, y) at angle n, or `None` when the bin
/// falls outside the accumulator - the caller's CPU formula.
pub type ExactIndex<'a> = &'a (dyn Fn(usize, usize, usize) -> Option<usize> + Sync);

/// Accumulate standard-Hough votes on the GPU for an f32 image buffer of
/// `w * h` pixels (pixels `> 0` vote). `scaled_cos[n]`, `scaled_sin[n]` are
/// `cos(theta_n) / rho`, `sin(theta_n) / rho` in f32; bin = round(x c + y s) +
/// `rho_offset`; layout theta-major, `num_rho` bins per angle. `exact` must
/// reproduce the CPU binning; it is used for near-boundary votes.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    w: usize,
    h: usize,
    scaled_cos: &[f32],
    scaled_sin: &[f32],
    num_rho: usize,
    rho_offset: i32,
    exact: ExactIndex<'_>,
    capacity: u32,
) -> Result<Vec<u32>> {
    let num_angle = scaled_cos.len();
    assert_eq!(num_angle, scaled_sin.len(), "trig tables differ in length");
    let acc_len = num_rho
        .checked_mul(num_angle)
        .ok_or_else(|| crate::Error::InvalidInput("Hough accumulator size too large".into()))?;
    if acc_len == 0 || w == 0 || h == 0 {
        return Ok(vec![0; acc_len]);
    }

    let trig: Vec<[f32; 2]> = scaled_cos
        .iter()
        .zip(scaled_sin)
        .map(|(&c, &s)| [c, s])
        .collect();
    let trig_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hough Trig"),
            contents: bytemuck::cast_slice(&trig),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let acc_buffer = ctx.get_buffer((acc_len * 4) as u64, usage);
    let ambiguous_words = 1 + 2 * capacity as usize;
    let ambiguous_buffer = ctx.get_buffer((ambiguous_words * 4) as u64, usage);

    let params = HoughParams {
        width: w as u32,
        height: h as u32,
        num_rho: num_rho as u32,
        num_angle: num_angle as u32,
        rho_offset,
        ambiguous_capacity: capacity,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hough Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let pipeline = ctx.create_compute_pipeline(include_str!("../../shaders/hough.wgsl"), "vote");
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Hough BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: trig_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: acc_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ambiguous_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Hough Vote"),
        });
    encoder.clear_buffer(&acc_buffer, 0, None);
    encoder.clear_buffer(&ambiguous_buffer, 0, Some(4));
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hough Vote"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y, z) = crate::gpu_kernels::dispatch_1d((w * h) as u32, 256);
        pass.dispatch_workgroups(x, y, z);
    }
    ctx.submit(encoder);

    let read = |buf: &wgpu::Buffer, bytes: usize| -> Result<Vec<u32>> {
        pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
            ctx.device.clone(),
            &ctx.queue,
            buf,
            0,
            bytes,
        ))
    };
    let count = read(&ambiguous_buffer, 4).map(|v| v[0]);
    let result = count.and_then(|count| {
        if count > capacity {
            // Overflow: the dropped votes are unknown, recount everything on
            // the host from the image itself.
            let px: Vec<f32> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
                ctx.device.clone(),
                &ctx.queue,
                input,
                0,
                w * h * 4,
            ))?;
            let mut acc = vec![0u32; acc_len];
            for (i, _) in px.iter().enumerate().filter(|(_, &v)| v > 0.0) {
                for n in 0..num_angle {
                    if let Some(j) = exact(i % w, i / w, n) {
                        acc[j] += 1;
                    }
                }
            }
            return Ok(acc);
        }
        let mut acc = read(&acc_buffer, acc_len * 4)?;
        if count > 0 {
            let pairs = read(&ambiguous_buffer, (1 + 2 * count as usize) * 4)?;
            for p in pairs[1..].chunks_exact(2) {
                let (i, n) = (p[0] as usize, p[1] as usize);
                if let Some(j) = exact(i % w, i / w, n) {
                    acc[j] += 1;
                }
            }
        }
        Ok(acc)
    });

    ctx.return_buffer(acc_buffer, usage);
    ctx.return_buffer(ambiguous_buffer, usage);
    result
}

/// [`accumulate`] for a host 8-bit edge map (nonzero = edge).
#[allow(clippy::too_many_arguments)]
pub fn hough_accumulate_u8(
    ctx: &GpuContext,
    edges: &[u8],
    w: usize,
    h: usize,
    scaled_cos: &[f32],
    scaled_sin: &[f32],
    num_rho: usize,
    rho_offset: i32,
    exact: ExactIndex<'_>,
) -> Result<Vec<u32>> {
    hough_accumulate_u8_with_capacity(
        ctx,
        edges,
        w,
        h,
        scaled_cos,
        scaled_sin,
        num_rho,
        rho_offset,
        exact,
        AMBIGUOUS_CAPACITY,
    )
}

/// [`hough_accumulate_u8`] with an explicit ambiguous-vote capacity - a test
/// hook for the host recount taken when the list overflows.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn hough_accumulate_u8_with_capacity(
    ctx: &GpuContext,
    edges: &[u8],
    w: usize,
    h: usize,
    scaled_cos: &[f32],
    scaled_sin: &[f32],
    num_rho: usize,
    rho_offset: i32,
    exact: ExactIndex<'_>,
    capacity: u32,
) -> Result<Vec<u32>> {
    if edges.len() != w * h {
        return Err(crate::Error::InvalidInput(format!(
            "hough: edge map has {} bytes, expected {w} x {h}",
            edges.len()
        )));
    }
    if w == 0 || h == 0 {
        return Ok(vec![0; num_rho.saturating_mul(scaled_cos.len())]);
    }
    let px: Vec<f32> = edges.iter().map(|&p| p as f32).collect();
    let input = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hough Edges"),
            contents: bytemuck::cast_slice(&px),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    accumulate(
        ctx, &input, w, h, scaled_cos, scaled_sin, num_rho, rho_offset, exact, capacity,
    )
}

/// `ComputeContext::hough_lines` on the GPU: the same lines, in the same
/// order, as the CPU backend (layout, binning and peak rule from
/// [`crate::cpu::hough_standard::HoughLayout`]).
pub fn hough_lines<T: Float + bytemuck::Pod + bytemuck::Zeroable + 'static>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    rho_res: T,
    theta_res: T,
    threshold: u32,
) -> Result<Vec<HoughLine>> {
    // Only f32 WGSL shader available
    if cv_core::DataType::from_type::<T>().ok() != Some(cv_core::DataType::F32) {
        return Err(crate::Error::NotSupported(
            "Hough Lines GPU kernel only supports f32".into(),
        ));
    }

    let (h, w) = input.shape.hw();
    let layout = HoughLayout::new(w, h, rho_res.to_f32(), theta_res.to_f32());
    let (cos, sin) = layout.trig_tables();
    let scaled_cos: Vec<f32> = cos.iter().map(|&c| c / layout.rho).collect();
    let scaled_sin: Vec<f32> = sin.iter().map(|&s| s / layout.rho).collect();
    let exact = |x: usize, y: usize, n: usize| layout.index(n, layout.bin(x, y, cos[n], sin[n]));

    let acc = if w == 0 || h == 0 {
        vec![0; layout.num_rho * layout.num_theta]
    } else {
        accumulate(
            ctx,
            input.storage.buffer(),
            w,
            h,
            &scaled_cos,
            &scaled_sin,
            layout.num_rho,
            layout.rho_center as i32,
            &exact,
            AMBIGUOUS_CAPACITY,
        )?
    };
    Ok(layout.peaks(&acc, threshold))
}
