use crate::Result;
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
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
    for (level, (p, n)) in prev_pyramid.iter().zip(next_pyramid).enumerate() {
        let (h, w) = p.shape.hw();
        if n.shape.hw() != (h, w) || p.shape.len() == 0 || n.shape.len() == 0 {
            return Err(crate::Error::InvalidInput(format!(
                "optical_flow_lk: level {level} prev and next sizes differ or are empty"
            )));
        }
    }

    // Bouguet's pyramidal LK (OpenCV calcOpticalFlowPyrLK). The feature
    // positions stay in level-0 coordinates (`points_buffer`); the shader scales
    // them by 1/2^L for each level and refines a flow guess (`guess_buffer`,
    // starting at zero) which it hands to the next finer level as 2 (g + v).
    //
    // The previous version ping-ponged *positions*: it bound the doubled
    // coarse result as the template centre and the un-doubled coarse result as
    // the starting guess, so on any multi-level pyramid the template sat on the
    // tracked position rather than the feature and the guess was in the wrong
    // level's coordinates (identical frames moved (60.5, 60.5) by
    // (-217, -54) px over 3 levels).
    let points_data: Vec<Point> = points.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let zero_guess = vec![Point { x: 0.0, y: 0.0 }; num_points];

    let points_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LK Points"),
            contents: bytemuck::cast_slice(&points_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let guess_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LK Guess"),
            contents: bytemuck::cast_slice(&zero_guess),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let shader_source = include_str!("../../shaders/lucas_kanade.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let params = LkParams {
        num_points: num_points as u32,
        window_radius: (window_size / 2) as i32,
        max_iters,
        // Absolute floor, matching video's pyramidal LK (`min_eig < 1e-4`
        // rejects). The shader compares against this value directly.
        min_eigenvalue: 1e-4,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LK Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LK Dispatch"),
        });
    for level in (0..levels).rev() {
        let prev_img = &prev_pyramid[level];
        let next_img = &next_pyramid[level];
        let (h, w) = prev_img.shape.hw();

        let level_params = [w as u32, h as u32, level as u32, 0];
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
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: guess_buffer.as_entire_binding(),
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

        // Successive passes in one encoder are ordered, so each level reads the
        // guess the coarser level wrote.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (wg_x, wg_y, wg_z) = crate::gpu_kernels::dispatch_1d(num_points as u32, 64);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }
    ctx.submit(encoder);

    let flows: Vec<Point> = pollster::block_on(crate::gpu_kernels::buffer_utils::read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &guess_buffer,
        0,
        num_points * 8,
    ))?;

    Ok(points
        .iter()
        .zip(&flows)
        .map(|(p, f)| [p[0] + f.x, p[1] + f.y])
        .collect())
}

// ---------------------------------------------------------------------------
// Dense optical flow (Farneback, TV-L1)
//
// Both return a `Tensor` of shape (2, h, w) in the workspace's planar CHW
// layout: plane 0 (`[0, h * w)`) is the horizontal flow u, plane 1
// (`[h * w, 2 * h * w)`) the vertical flow v, with OpenCV's sign convention
// prev(x, y) ≈ next(x + u, y + v). Every shader in the pipeline (poly-expand
// aside, whose scratch buffer is private) reads and writes flow planar, so
// `Tensor::index` on the result is correct without a transpose.
//
// Before this layout was made planar, the Farneback shaders stored (u, v)
// interleaved per pixel under the same (2, h, w) shape, so anyone indexing the
// tensor by its shape read garbage.
// ---------------------------------------------------------------------------

/// Uniform of `farneback_poly_expand.wgsl`. Only `width`, `height`, `stride`,
/// `poly_n` (byte 20) and `poly_sigma` (byte 24) are read; the other fields
/// keep the layout the shader was written against.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FarnebackParams {
    width: u32,
    height: u32,
    stride: u32,
    flow_stride: u32,
    window_size: u32,
    poly_n: u32,
    poly_sigma: f32,
    num_iters: u32,
}

/// Uniform of `farneback_update_matrices.wgsl` and `farneback_update_flow.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FarnebackFlowParams {
    width: u32,
    height: u32,
    /// The averaging window is `2 * half_window + 1` pixels wide.
    half_window: u32,
    /// `<= 0`: box window. Otherwise the Gaussian window's standard deviation.
    win_sigma: f32,
}

/// Window used to average the Farneback normal equations, as OpenCV's
/// `OPTFLOW_FARNEBACK_GAUSSIAN` flag selects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowBlurMode {
    /// Box window (OpenCV's default, flags = 0).
    Box,
    /// Gaussian window with sigma `0.3 * (window_size / 2)`
    /// (`OPTFLOW_FARNEBACK_GAUSSIAN`).
    Gaussian,
}

/// OpenCV stops building the Farneback pyramid before a level gets narrower
/// than this.
const FARNEBACK_MIN_LEVEL_SIZE: f64 = 32.0;

fn flow_dispatch(pass: &mut wgpu::ComputePass<'_>, w: u32, h: u32) {
    pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
}

/// Bind group with `buffers` at bindings `0..n`.
fn flow_bind_group(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn flow_storage(ctx: &GpuContext, label: &str, len: usize) -> wgpu::Buffer {
    // wgpu zero-initialises every new buffer, which the zero initial flow and
    // the zero initial TV-L1 dual variables rely on.
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn flow_uniform<P: bytemuck::Pod>(ctx: &GpuContext, label: &str, params: &P) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Common checks for the dense-flow entry points: one channel, equal non-empty
/// sizes, storage that holds the whole image.
fn check_flow_frames(
    name: &str,
    prev: &Tensor<f32, GpuStorage<f32>>,
    next: &Tensor<f32, GpuStorage<f32>>,
) -> Result<(usize, usize)> {
    let (h, w) = prev.shape.hw();
    if prev.shape.channels != 1 || next.shape.channels != 1 {
        return Err(crate::Error::InvalidInput(format!(
            "{name}: frames must have one channel, got {} and {}",
            prev.shape.channels, next.shape.channels
        )));
    }
    if next.shape.hw() != (h, w) {
        return Err(crate::Error::InvalidInput(format!(
            "{name}: prev is {w}x{h} but next is {}x{}",
            next.shape.width, next.shape.height
        )));
    }
    if w == 0 || h == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "{name}: empty frame ({w}x{h})"
        )));
    }
    if w > u32::MAX as usize / 16 || h > u32::MAX as usize / 16 {
        return Err(crate::Error::InvalidInput(format!(
            "{name}: frame {w}x{h} is too large"
        )));
    }
    let need = (w * h * std::mem::size_of::<f32>()) as u64;
    if prev.storage.buffer().size() < need || next.storage.buffer().size() < need {
        return Err(crate::Error::InvalidInput(format!(
            "{name}: frame storage is smaller than {w}x{h} f32"
        )));
    }
    Ok((h, w))
}

/// Encode `flow_upsample.wgsl`: planar `src` (`src_w x src_h`) to planar `dst`
/// (`dst_w x dst_h`), flow values scaled by the size ratio per axis.
fn encode_flow_upsample(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    (src_w, src_h): (u32, u32),
    dst: &wgpu::Buffer,
    (dst_w, dst_h): (u32, u32),
) {
    let pipeline =
        ctx.create_compute_pipeline(include_str!("../../shaders/flow_upsample.wgsl"), "main");
    let params = flow_uniform(
        ctx,
        "Flow Upsample Params",
        &crate::gpu_kernels::resize::ResizeParams {
            src_w,
            src_h,
            dst_w,
            dst_h,
            channels: 2,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    );
    let bg = flow_bind_group(ctx, &pipeline, "Flow Upsample", &[src, dst, &params]);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bg, &[]);
    flow_dispatch(&mut pass, dst_w, dst_h);
}

/// Dense optical flow by Farnebäck's polynomial expansion, following OpenCV's
/// `calcOpticalFlowFarneback` (`optflowgf.cpp`).
///
/// * `num_levels`: pyramid levels including the full-resolution frame
///   (1 = no pyramid). As in OpenCV, no level narrower than 32 pixels is
///   built, so fewer may be used.
/// * `pyr_scale`: size ratio between consecutive levels, in `(0, 1)`. Level
///   `k` is the frame blurred with sigma `(1 / s - 1) / 2` and resized to
///   `round(size * s)`, `s = pyr_scale^k`.
/// * `window_size`: averaging window (OpenCV `winsize`).
/// * `num_iters`: iterations per level.
/// * `poly_n`: half-size of the polynomial-expansion neighbourhood, as OpenCV
///   uses it (the fit spans `2 * poly_n + 1` pixels); typically 5 or 7.
/// * `poly_sigma`: Gaussian of the expansion; typically 1.1 or 1.2 for
///   `poly_n = 5`, 1.5 for 7.
/// * `blur_mode`: box (OpenCV's default) or Gaussian window.
///
/// Intensities are taken on OpenCV's 0-255 scale: the solve's `+ 1e-3`
/// regulariser is OpenCV's and assumes it. Frames in `[0, 1]` should be
/// multiplied by 255 first.
///
/// Returns flow of shape `(2, h, w)`, planar: plane 0 is u (x), plane 1 is v
/// (y), with `prev(x, y) ≈ next(x + u, y + v)`.
#[allow(clippy::too_many_arguments)]
pub fn farneback(
    ctx: &GpuContext,
    prev: &Tensor<f32, GpuStorage<f32>>,
    next: &Tensor<f32, GpuStorage<f32>>,
    num_levels: u32,
    pyr_scale: f32,
    window_size: u32,
    num_iters: u32,
    poly_n: u32,
    poly_sigma: f32,
    blur_mode: FlowBlurMode,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = check_flow_frames("farneback", prev, next)?;
    if num_levels == 0 {
        return Err(crate::Error::InvalidInput(
            "farneback: num_levels must be at least 1".into(),
        ));
    }
    if !(pyr_scale.is_finite() && pyr_scale > 0.0 && pyr_scale < 1.0) {
        return Err(crate::Error::InvalidInput(format!(
            "farneback: pyr_scale must be in (0, 1), got {pyr_scale}"
        )));
    }
    if window_size == 0 || num_iters == 0 || poly_n == 0 {
        return Err(crate::Error::InvalidInput(format!(
            "farneback: window_size ({window_size}), num_iters ({num_iters}) and \
             poly_n ({poly_n}) must be at least 1"
        )));
    }
    if window_size > 513 || poly_n > 64 {
        return Err(crate::Error::InvalidInput(format!(
            "farneback: window_size {window_size} (max 513) or poly_n {poly_n} (max 64) too large"
        )));
    }
    if !(poly_sigma.is_finite() && poly_sigma > 0.0) {
        return Err(crate::Error::InvalidInput(format!(
            "farneback: poly_sigma must be finite and > 0, got {poly_sigma}"
        )));
    }

    // OpenCV: count the levels that stay at least 32 pixels on each side.
    let mut top = 0u32;
    let mut scale = 1.0f64;
    for k in 1..num_levels {
        scale *= pyr_scale as f64;
        if (w as f64) * scale < FARNEBACK_MIN_LEVEL_SIZE
            || (h as f64) * scale < FARNEBACK_MIN_LEVEL_SIZE
        {
            break;
        }
        top = k;
    }

    let poly_pipeline = ctx.create_compute_pipeline(
        include_str!("../../shaders/farneback_poly_expand.wgsl"),
        "main",
    );
    let mats_pipeline = ctx.create_compute_pipeline(
        include_str!("../../shaders/farneback_update_matrices.wgsl"),
        "main",
    );
    let solve_pipeline = ctx.create_compute_pipeline(
        include_str!("../../shaders/farneback_update_flow.wgsl"),
        "main",
    );

    let half_window = window_size / 2;
    let win_sigma = match blur_mode {
        FlowBlurMode::Box => 0.0,
        FlowBlurMode::Gaussian => 0.3 * half_window as f32,
    };

    let mut flow: Option<(wgpu::Buffer, u32, u32)> = None;
    for k in (0..=top).rev() {
        let scale = (pyr_scale as f64).powi(k as i32);
        let lw = (((w as f64) * scale).round() as u32).max(1);
        let lh = (((h as f64) * scale).round() as u32).max(1);
        let level_prev = farneback_level(ctx, prev, scale, lw, lh)?;
        let level_next = farneback_level(ctx, next, scale, lw, lh)?;

        let n = (lw * lh) as usize;
        let flow_buf = flow_storage(ctx, "Farneback Flow", 2 * n);
        let poly1 = flow_storage(ctx, "Farneback Poly1", 6 * n);
        let poly2 = flow_storage(ctx, "Farneback Poly2", 6 * n);
        let mats = flow_storage(ctx, "Farneback Matrices", 5 * n);

        let poly_params = flow_uniform(
            ctx,
            "Farneback Poly Params",
            &FarnebackParams {
                width: lw,
                height: lh,
                stride: lw,
                flow_stride: lw,
                window_size,
                poly_n,
                poly_sigma,
                num_iters,
            },
        );
        let flow_params = flow_uniform(
            ctx,
            "Farneback Flow Params",
            &FarnebackFlowParams {
                width: lw,
                height: lh,
                half_window,
                win_sigma,
            },
        );
        let poly1_bg = flow_bind_group(
            ctx,
            &poly_pipeline,
            "Farneback Poly1",
            &[level_prev.storage.buffer(), &poly1, &poly_params],
        );
        let poly2_bg = flow_bind_group(
            ctx,
            &poly_pipeline,
            "Farneback Poly2",
            &[level_next.storage.buffer(), &poly2, &poly_params],
        );
        let mats_bg = flow_bind_group(
            ctx,
            &mats_pipeline,
            "Farneback Matrices",
            &[&poly1, &poly2, &flow_buf, &mats, &flow_params],
        );
        let solve_bg = flow_bind_group(
            ctx,
            &solve_pipeline,
            "Farneback Solve",
            &[&mats, &flow_buf, &flow_params],
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Farneback Level"),
            });
        // The coarsest level starts from zero flow (the buffer is zeroed);
        // finer ones from the coarser estimate, scaled to this level.
        if let Some((coarse, cw, ch)) = &flow {
            encode_flow_upsample(ctx, &mut encoder, coarse, (*cw, *ch), &flow_buf, (lw, lh));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&poly_pipeline);
            pass.set_bind_group(0, &poly1_bg, &[]);
            flow_dispatch(&mut pass, lw, lh);
            pass.set_bind_group(0, &poly2_bg, &[]);
            flow_dispatch(&mut pass, lw, lh);
            for _ in 0..num_iters {
                pass.set_pipeline(&mats_pipeline);
                pass.set_bind_group(0, &mats_bg, &[]);
                flow_dispatch(&mut pass, lw, lh);
                pass.set_pipeline(&solve_pipeline);
                pass.set_bind_group(0, &solve_bg, &[]);
                flow_dispatch(&mut pass, lw, lh);
            }
        }
        ctx.submit(encoder);
        flow = Some((flow_buf, lw, lh));
    }

    let (buf, _, _) = flow.ok_or_else(|| crate::Error::RuntimeError("No flow computed".into()))?;
    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(buf), 2 * w * h),
        shape: cv_core::TensorShape::new(2, h, w),
        dtype: prev.dtype,
        _phantom: std::marker::PhantomData,
    })
}

/// One Farneback pyramid level as OpenCV builds it: the full-resolution frame
/// blurred with sigma `(1 / scale - 1) / 2` (kernel `max(round(5 sigma) | 1, 3)`,
/// and at `scale = 1` OpenCV's 3x3 default sigma of 0.8), then resized.
fn farneback_level(
    ctx: &GpuContext,
    frame: &Tensor<f32, GpuStorage<f32>>,
    scale: f64,
    lw: u32,
    lh: u32,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let sigma = (1.0 / scale - 1.0) * 0.5;
    let ksize = (((sigma * 5.0).round() as usize) | 1).max(3);
    // cv::GaussianBlur with sigma 0 derives it from the kernel size.
    let sigma = if sigma > 0.0 {
        sigma
    } else {
        0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8
    };
    let blurred = crate::gpu_kernels::convolve::gaussian_blur_with_border(
        ctx,
        frame,
        sigma as f32,
        ksize,
        crate::context::BorderMode::Reflect101,
    )?;
    let (h, w) = frame.shape.hw();
    if (lw as usize, lh as usize) == (w, h) {
        Ok(blurred)
    } else {
        crate::gpu_kernels::resize::resize(ctx, &blurred, lw, lh)
    }
}

/// Uniform shared by every TV-L1 shader (`tvl1_*.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Tvl1Params {
    width: u32,
    height: u32,
    /// lambda * theta.
    l_t: f32,
    theta: f32,
    /// tau / theta.
    taut: f32,
    median_radius: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters of [`tvl1_optical_flow`]; names and defaults follow OpenCV's
/// `DualTVL1OpticalFlow`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tvl1Config {
    /// Time step of the dual update (OpenCV `tau`). Stable for `tau <= 0.25`.
    pub tau: f32,
    /// Weight of the data term (OpenCV `lambda`), for 0-255 intensities.
    pub lambda: f32,
    /// Coupling between the flow and its auxiliary variable (OpenCV `theta`).
    pub theta: f32,
    /// Pyramid scales (OpenCV `nscales`). No scale narrower than 16 pixels is
    /// built, as in OpenCV.
    pub num_scales: u32,
    /// Size ratio between consecutive scales, in `(0, 1)` (OpenCV `scaleStep`).
    pub scale_step: f32,
    /// Warps (re-linearisations) per scale (OpenCV `warps`).
    pub num_warps: u32,
    /// Outer iterations per warp; each starts with the median filter
    /// (OpenCV `outerIterations`).
    pub num_outer_iters: u32,
    /// Primal-dual iterations per outer iteration (OpenCV `innerIterations`).
    pub num_inner_iters: u32,
    /// Median filter size applied to the flow every outer iteration: 0 or 1
    /// (off), 3 or 5 (OpenCV `medianFiltering`).
    pub median_filtering: u32,
}

impl Default for Tvl1Config {
    fn default() -> Self {
        Self {
            tau: 0.25,
            lambda: 0.15,
            theta: 0.3,
            num_scales: 5,
            scale_step: 0.8,
            num_warps: 5,
            num_outer_iters: 10,
            num_inner_iters: 30,
            median_filtering: 5,
        }
    }
}

/// OpenCV stops building the TV-L1 pyramid before a scale narrower than this.
const TVL1_MIN_SCALE_SIZE: usize = 16;

/// Dense optical flow by the duality-based TV-L1 method (Zach, Pock, Bischof
/// 2007; Sánchez, Meinhardt-Llopis, Facciolo 2013), following OpenCV's
/// `DualTVL1OpticalFlow` (`tvl1flow.cpp`) with `gamma = 0`.
///
/// Coarse to fine over `num_scales` scales; at each, `num_warps` times: warp
/// `next` by the flow (bicubic) and linearise, then `num_outer_iters` x
/// (median filter, `num_inner_iters` x (thresholding + primal step, dual
/// step)). Differences from OpenCV: every iteration runs (OpenCV may stop a
/// warp early once the mean squared update is below `epsilon²`; reading that
/// back would stall the GPU every iteration), and the warp replicates the
/// border where OpenCV's `remap` reads zeros.
///
/// Intensities are used as given; `lambda` is calibrated for 0-255. OpenCV
/// multiplies floating-point frames by 255 before this, so frames in `[0, 1]`
/// need `lambda * 255` (or rescaling) to match.
///
/// Returns flow of shape `(2, h, w)`, planar: plane 0 is u (x), plane 1 is v
/// (y), with `prev(x, y) ≈ next(x + u, y + v)`.
pub fn tvl1_optical_flow(
    ctx: &GpuContext,
    prev: &Tensor<f32, GpuStorage<f32>>,
    next: &Tensor<f32, GpuStorage<f32>>,
    config: Tvl1Config,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (h, w) = check_flow_frames("tvl1_optical_flow", prev, next)?;
    let positive = |v: f32| v.is_finite() && v > 0.0;
    if !(positive(config.tau) && positive(config.lambda) && positive(config.theta)) {
        return Err(crate::Error::InvalidInput(format!(
            "tvl1_optical_flow: tau ({}), lambda ({}) and theta ({}) must be finite and > 0",
            config.tau, config.lambda, config.theta
        )));
    }
    if !(config.scale_step.is_finite() && config.scale_step > 0.0 && config.scale_step < 1.0) {
        return Err(crate::Error::InvalidInput(format!(
            "tvl1_optical_flow: scale_step must be in (0, 1), got {}",
            config.scale_step
        )));
    }
    if config.num_scales == 0
        || config.num_warps == 0
        || config.num_outer_iters == 0
        || config.num_inner_iters == 0
    {
        return Err(crate::Error::InvalidInput(
            "tvl1_optical_flow: num_scales, num_warps, num_outer_iters and \
             num_inner_iters must be at least 1"
                .into(),
        ));
    }
    let median_radius = match config.median_filtering {
        0 | 1 => 0,
        3 => 1,
        5 => 2,
        m => {
            return Err(crate::Error::InvalidInput(format!(
                "tvl1_optical_flow: median_filtering must be 0, 1, 3 or 5, got {m}"
            )));
        }
    };

    // OpenCV: chained INTER_LINEAR resizes, stopping before a scale under 16.
    let mut pyramid = vec![(prev.clone(), next.clone())];
    for _ in 1..config.num_scales {
        let (p, q) = pyramid.last().unwrap();
        let (ph, pw) = p.shape.hw();
        let sw = ((pw as f64) * config.scale_step as f64).round() as usize;
        let sh = ((ph as f64) * config.scale_step as f64).round() as usize;
        if sw < TVL1_MIN_SCALE_SIZE || sh < TVL1_MIN_SCALE_SIZE {
            break;
        }
        let p = crate::gpu_kernels::resize::resize(ctx, p, sw as u32, sh as u32)?;
        let q = crate::gpu_kernels::resize::resize(ctx, q, sw as u32, sh as u32)?;
        pyramid.push((p, q));
    }

    let warp_pipeline =
        ctx.create_compute_pipeline(include_str!("../../shaders/tvl1_warp.wgsl"), "main");
    let primal_dual_pipeline =
        ctx.create_compute_pipeline(include_str!("../../shaders/tvl1_primal_dual.wgsl"), "main");
    let median_pipeline =
        ctx.create_compute_pipeline(include_str!("../../shaders/tvl1_median.wgsl"), "main");

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TVL1"),
        });
    // The previous (coarser) scale's final state and its size.
    let mut coarse: Option<(wgpu::Buffer, u32, u32)> = None;
    for (i0, i1) in pyramid.iter().rev() {
        let (sh, sw) = i0.shape.hw();
        let (sw, sh) = (sw as u32, sh as u32);
        let n = (sw * sh) as usize;

        // Two states, (u1, u2, p11, p12, p21, p22) planar (see
        // tvl1_primal_dual.wgsl): every inner iteration and every median
        // filter reads one and writes the other. New buffers are zeroed, which
        // is the zero initial flow and dual variables.
        let states = [
            flow_storage(ctx, "TVL1 State A", 6 * n),
            flow_storage(ctx, "TVL1 State B", 6 * n),
        ];
        let warp = flow_storage(ctx, "TVL1 Warp (I1wx, I1wy, grad, rho_c)", 4 * n);
        let params = flow_uniform(
            ctx,
            "TVL1 Params",
            &Tvl1Params {
                width: sw,
                height: sh,
                l_t: config.lambda * config.theta,
                theta: config.theta,
                taut: config.tau / config.theta,
                median_radius,
                _pad0: 0,
                _pad1: 0,
            },
        );

        // [warp, primal-dual, median], reading state `c`.
        let bgs: Vec<[wgpu::BindGroup; 3]> = (0..2)
            .map(|c| {
                let (cur, other) = (&states[c], &states[1 - c]);
                [
                    flow_bind_group(
                        ctx,
                        &warp_pipeline,
                        "TVL1 Warp",
                        &[
                            i0.storage.buffer(),
                            i1.storage.buffer(),
                            cur,
                            &warp,
                            &params,
                        ],
                    ),
                    flow_bind_group(
                        ctx,
                        &primal_dual_pipeline,
                        "TVL1 Primal-Dual",
                        &[&warp, cur, other, &params],
                    ),
                    flow_bind_group(ctx, &median_pipeline, "TVL1 Median", &[cur, other, &params]),
                ]
            })
            .collect();

        // The coarsest scale starts from zero flow; finer ones from the
        // coarser estimate, scaled to this scale. The upsample reads and
        // writes only the flow planes, which come first in a state.
        if let Some((prev_state, cw, ch)) = &coarse {
            encode_flow_upsample(
                ctx,
                &mut encoder,
                prev_state,
                (*cw, *ch),
                &states[0],
                (sw, sh),
            );
        }
        let mut cur = 0usize;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for _ in 0..config.num_warps {
                pass.set_pipeline(&warp_pipeline);
                pass.set_bind_group(0, &bgs[cur][0], &[]);
                flow_dispatch(&mut pass, sw, sh);
                for _ in 0..config.num_outer_iters {
                    if median_radius > 0 {
                        pass.set_pipeline(&median_pipeline);
                        pass.set_bind_group(0, &bgs[cur][2], &[]);
                        flow_dispatch(&mut pass, sw, sh);
                        cur = 1 - cur;
                    }
                    pass.set_pipeline(&primal_dual_pipeline);
                    for _ in 0..config.num_inner_iters {
                        pass.set_bind_group(0, &bgs[cur][1], &[]);
                        flow_dispatch(&mut pass, sw, sh);
                        cur = 1 - cur;
                    }
                }
            }
        }
        let [a, b] = states;
        coarse = Some((if cur == 0 { a } else { b }, sw, sh));
    }

    let (state, _, _) =
        coarse.ok_or_else(|| crate::Error::RuntimeError("No flow computed".into()))?;
    // The flow is the first two planes of the final state.
    let flow_bytes = (2 * w * h * std::mem::size_of::<f32>()) as u64;
    let flow = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TVL1 Flow"),
        size: flow_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&state, 0, &flow, 0, flow_bytes);
    ctx.submit(encoder);
    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(flow), 2 * w * h),
        shape: cv_core::TensorShape::new(2, h, w),
        dtype: prev.dtype,
        _phantom: std::marker::PhantomData,
    })
}

#[cfg(test)]
fn poly_gauss(x: f32, sigma: f32) -> f32 {
    let two = 2.0 * sigma * sigma;
    if !(two.is_finite() && two > 0.0) {
        return 1.0;
    }
    let w = (-(x * x) / two).exp();
    if w.is_finite() { w } else { 1.0 }
}

/// Weighted least-squares quadratic at one pixel. Same fit as
/// `farneback_poly_expand.wgsl`: basis `[1, x, y, x², xy, y²]` over the
/// `(2 poly_n + 1)²` window, border pixels replicated.
#[cfg(test)]
pub(crate) fn farneback_poly_pixel(
    img: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    poly_n: usize,
    sigma: f32,
) -> [f32; 6] {
    let half = poly_n as i32;
    let mut a = [0.0f32; 36];
    let mut b = [0.0f32; 6];
    for dy in -half..=half {
        for dx in -half..=half {
            let px = (x as i32 + dx).clamp(0, w as i32 - 1);
            let py = (y as i32 + dy).clamp(0, h as i32 - 1);
            let val = img[py as usize * w + px as usize];
            let xf = dx as f32;
            let yf = dy as f32;
            let weight = poly_gauss(xf, sigma) * poly_gauss(yf, sigma);
            let basis = [1.0, xf, yf, xf * xf, xf * yf, yf * yf];
            for i in 0..6 {
                for j in 0..6 {
                    a[i * 6 + j] += basis[i] * basis[j] * weight;
                }
                b[i] += basis[i] * val * weight;
            }
        }
    }
    solve6(&mut a, &mut b)
}

#[cfg(test)]
fn solve6(a: &mut [f32; 36], b: &mut [f32; 6]) -> [f32; 6] {
    for col in 0..6 {
        let mut max_row = col;
        let mut max_val = a[col * 6 + col].abs();
        for row in (col + 1)..6 {
            let v = a[row * 6 + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            for row in 0..6 {
                a[row * 6 + col] = 0.0;
            }
            continue;
        }
        if max_row != col {
            for j in 0..6 {
                a.swap(col * 6 + j, max_row * 6 + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * 6 + col];
        for row in (col + 1)..6 {
            let factor = a[row * 6 + col] / pivot;
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in col..6 {
                let sub = factor * a[col * 6 + j];
                a[row * 6 + j] -= sub;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f32; 6];
    for i in (0..6).rev() {
        if a[i * 6 + i].abs() < 1e-12 {
            continue;
        }
        let mut sum = b[i];
        for j in (i + 1)..6 {
            sum -= a[i * 6 + j] * x[j];
        }
        let xi = sum / a[i * 6 + i];
        x[i] = if xi.is_finite() { xi } else { 0.0 };
    }
    x
}

#[cfg(test)]
mod flow_fix_tests {
    use super::*;

    #[test]
    fn farneback_uniform_puts_poly_n_and_sigma_where_the_shader_reads_them() {
        assert_eq!(std::mem::offset_of!(FarnebackParams, poly_n), 20);
        assert_eq!(std::mem::offset_of!(FarnebackParams, poly_sigma), 24);
        let params = FarnebackParams {
            width: 8,
            height: 8,
            stride: 8,
            flow_stride: 16,
            window_size: 15,
            poly_n: 5,
            poly_sigma: 1.2,
            num_iters: 1,
        };
        let bytes = bytemuck::bytes_of(&params);
        let poly_n = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let sigma = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(poly_n, 5);
        assert!((sigma - 1.2).abs() < 1e-6);
        // The old shader read byte 12 as poly_n: an 8-pixel level's flow
        // stride, which is 16.
        let old_poly_n = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(old_poly_n, 16);
    }

    #[test]
    fn constant_image_poly_has_no_nan_and_zero_quadratic() {
        let value = 7.5f32;
        let (w, h) = (8usize, 8usize);
        let img = vec![value; w * h];
        for y in 0..h {
            for x in 0..w {
                let c = farneback_poly_pixel(&img, w, h, x, y, 5, 1.2);
                assert!(c.iter().all(|v| v.is_finite()), "NaN at {x},{y}: {c:?}");
                assert!(
                    (c[0] - value).abs() < 1e-3,
                    "constant term at {x},{y} is {}, want {value}",
                    c[0]
                );
                for (k, name) in [(3, "x^2"), (4, "xy"), (5, "y^2")] {
                    assert!(c[k].abs() < 1e-3, "{name} at {x},{y} is {}", c[k]);
                }
            }
        }
    }
}
