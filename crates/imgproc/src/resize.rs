use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::ResourceGroup;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::kernels::{cubic_kernel, lanczos_kernel};

/// Resampling filter.
///
/// `Linear` is half-pixel bilinear, the same mapping as OpenCV `INTER_LINEAR`
/// (see `resize_sampling_convention`). It point-samples, so a 2× shrink
/// discards three of every four source pixels and aliases. That is OpenCV's
/// choice for `INTER_LINEAR` too. [`Interpolation::Area`] averages the source
/// rectangle each output pixel covers, which is the decimation filter.
/// Changing `Linear` would alter every caller that upscales with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Linear,
    Cubic,
    Lanczos,
    /// Pixel-area average. On an integer shrink this is a box filter, so a
    /// checkerboard does not survive as stripes. On a zoom each output pixel
    /// falls inside one source pixel, which is what OpenCV's `INTER_AREA` does
    /// (nearest, not bilinear).
    Area,
}

pub fn resize(src: &GrayImage, width: u32, height: u32, interpolation: Interpolation) -> GrayImage {
    // Attempt to get a default group, fallback to CPU only if scheduler fails (unlikely in initialized app)
    // For now, we can create a temporary CPU-only group or panic if init failed.
    // Better: use scheduler if available.
    if let Ok(s) = cv_runtime::orchestrator::scheduler() {
        if let Ok(group) = s.get_default_group() {
            return resize_ctx(src, width, height, interpolation, &group);
        }
    }
    // Fallback: minimal CPU run
    resize_cpu(src, width, height, interpolation)
}

/// CPU dispatch for the requested interpolation.
///
/// Every variant is honoured: `Nearest`/`Linear` replicate/clamp sample the
/// source, `Cubic` uses OpenCV's cubic kernel (Keys a = -0.75) and `Lanczos`
/// OpenCV's Lanczos-4 window - see `kernels.rs` for why those are the choices.
fn resize_cpu(src: &GrayImage, width: u32, height: u32, interpolation: Interpolation) -> GrayImage {
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    if src.width() == 0 || src.height() == 0 {
        return GrayImage::new(width, height);
    }
    match interpolation {
        Interpolation::Nearest => resize_nearest(src, width, height),
        // Bilinear is owned by the HAL CPU backend (see `resize_cpu_hal`); there
        // is exactly one bilinear implementation per backend, shared with the
        // GPU resize shader.
        Interpolation::Linear => match cv_hal::cpu::CpuBackend::new() {
            Some(cpu) => resize_cpu_hal(&cpu, src, width, height)
                .unwrap_or_else(|_| GrayImage::new(width, height)),
            None => GrayImage::new(width, height),
        },
        Interpolation::Cubic => resize_sampled(src, width, height, 2, cubic_weights),
        Interpolation::Lanczos => resize_sampled(src, width, height, 4, lanczos_weights),
        Interpolation::Area => resize_area(src, width, height),
    }
}

pub fn resize_ctx(
    src: &GrayImage,
    width: u32,
    height: u32,
    interpolation: Interpolation,
    group: &ResourceGroup,
) -> GrayImage {
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    if src.width() == 0 || src.height() == 0 {
        return GrayImage::new(width, height);
    }

    // Bilinear resizing is a single implementation per backend, both routed
    // through the HAL: the CPU `ComputeContext::resize` and the GPU resize
    // shader share the same align-corners bilinear mapping. Nearest/Cubic/
    // Lanczos have no HAL equivalent and stay CPU-only below.
    if interpolation == Interpolation::Linear {
        if let Ok(device) = group.device() {
            match device {
                ComputeDevice::Gpu(gpu) => {
                    if let Ok(result) = resize_gpu(gpu, src, width, height) {
                        return result;
                    }
                }
                ComputeDevice::Cpu(cpu) => {
                    if let Ok(result) = resize_cpu_hal(cpu, src, width, height) {
                        return result;
                    }
                }
                ComputeDevice::Mlx(_) => {}
            }
        }
    }

    group.run(|| resize_cpu(src, width, height, interpolation))
}

/// Bilinear resize on the HAL CPU backend (the single CPU implementation).
///
/// The HAL computes the sample in `f32` and returns `f32`; the conversion back
/// to `u8` rounds to nearest (`+ 0.5`) to match the historical CPU resampler.
fn resize_cpu_hal(
    cpu: &cv_hal::cpu::CpuBackend,
    src: &GrayImage,
    width: u32,
    height: u32,
) -> cv_hal::Result<GrayImage> {
    use cv_core::storage::Storage;
    use cv_hal::context::ComputeContext;

    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

    let output = cpu.resize(&input, (width as usize, height as usize))?;
    let data: Vec<u8> = output
        .storage
        .as_slice()
        .ok_or_else(|| cv_hal::Error::MemoryError("Output not on CPU".into()))?
        .iter()
        .map(|&v: &f32| (v + 0.5).clamp(0.0, 255.0) as u8)
        .collect();

    GrayImage::from_raw(width, height, data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

fn resize_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    width: u32,
    height: u32,
) -> cv_hal::Result<GrayImage> {
    use cv_core::{storage::Storage, Tensor};
    use cv_hal::context::ComputeContext;
    use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let output_gpu = gpu.resize(&input_gpu, (width as usize, height as usize))?;
    let output_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = output_gpu.to_cpu_ctx(gpu)?;

    let data: Vec<u8> = output_cpu
        .storage
        .as_slice()
        .ok_or_else(|| cv_hal::Error::MemoryError("Download failed".into()))?
        .iter()
        .map(|&v| v.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(width, height, data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

/// Average of the source rectangle each output pixel covers.
///
/// An output pixel `(x, y)` owns `[x, x+1) * src_w/dst_w` by
/// `[y, y+1) * src_h/dst_h` in source coordinates. Every source pixel that
/// overlaps that rectangle contributes its value times the overlap area.
/// On a 2× shrink that is exactly the four source pixels, so a checkerboard
/// becomes the mean instead of one of the two colours.
fn resize_area(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let sw = src.width();
    let sh = src.height();
    let src_raw = src.as_raw();
    let scale_x = sw as f64 / width as f64;
    let scale_y = sh as f64 / height as f64;
    let mut dst = vec![0u8; (width * height) as usize];
    dst.par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = y as f64 * scale_y;
            let y1 = (y as f64 + 1.0) * scale_y;
            for x in 0..width {
                row[x as usize] = area_sample(src_raw, sw, sh, 1, scale_x, y0, y1, x as f64)[0];
            }
        });
    GrayImage::from_raw(width, height, dst).unwrap_or_else(|| GrayImage::new(width, height))
}

fn resize_rgb_area(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let sw = src.width();
    let sh = src.height();
    let src_raw = src.as_raw();
    let scale_x = sw as f64 / width as f64;
    let scale_y = sh as f64 / height as f64;
    let mut dst = vec![0u8; (width * height * 3) as usize];
    dst.par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = y as f64 * scale_y;
            let y1 = (y as f64 + 1.0) * scale_y;
            for x in 0..width {
                let px = area_sample(src_raw, sw, sh, 3, scale_x, y0, y1, x as f64);
                let o = x as usize * 3;
                row[o] = px[0];
                row[o + 1] = px[1];
                row[o + 2] = px[2];
            }
        });
    RgbImage::from_raw(width, height, dst).unwrap_or_else(|| RgbImage::new(width, height))
}

/// One output sample. `channels` is 1 or 3, packed in `src`.
fn area_sample(
    src: &[u8],
    sw: u32,
    sh: u32,
    channels: usize,
    scale_x: f64,
    y0: f64,
    y1: f64,
    x: f64,
) -> [u8; 3] {
    let x0 = x * scale_x;
    let x1 = (x + 1.0) * scale_x;
    let ix0 = x0.floor() as u32;
    let ix1 = (x1.ceil() as u32).min(sw);
    let iy0 = y0.floor() as u32;
    let iy1 = (y1.ceil() as u32).min(sh);
    let mut acc = [0.0f64; 3];
    let mut wsum = 0.0f64;
    for sy in iy0..iy1 {
        let wy = (((sy + 1) as f64).min(y1) - (sy as f64).max(y0)).max(0.0);
        if wy == 0.0 {
            continue;
        }
        for sx in ix0..ix1 {
            let wx = (((sx + 1) as f64).min(x1) - (sx as f64).max(x0)).max(0.0);
            if wx == 0.0 {
                continue;
            }
            let w = wx * wy;
            let base = ((sy * sw + sx) as usize) * channels;
            for c in 0..channels {
                acc[c] += src[base + c] as f64 * w;
            }
            wsum += w;
        }
    }
    let mut out = [0u8; 3];
    if wsum > 0.0 {
        for c in 0..channels {
            out[c] = (acc[c] / wsum).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn resize_nearest(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut dst = GrayImage::new(width, height);
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;
    let dst_width = width as f32;
    let dst_height = height as f32;

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                // OpenCV's INTER_NEAREST floors `x / (nw / w)` - the scale as a
                // reciprocal, in f32. Measured against cv2 over 293 samples of
                // 12 (w, nw) pairs, this is the only form that matches every
                // one: the multiply forms `(x*w)/nw`, `x*(w/nw)` (f32 and f64)
                // all give 27.0 at the 36->40 y=30 tie where cv2 picks 26, and
                // the half-pixel form is wrong in 101 of 293. Nearest is NOT
                // half-pixel; only the interpolating modes are.
                let sx = ((x as f32 / (dst_width / src_width)).floor() as u32).min(src.width() - 1);
                let sy =
                    ((y as f32 / (dst_height / src_height)).floor() as u32).min(src.height() - 1);
                let val = src.get_pixel(sx, sy)[0];
                row[x as usize] = val;
            }
        });
    dst
}

pub fn resize_rgb(
    src: &RgbImage,
    width: u32,
    height: u32,
    interpolation: Interpolation,
) -> RgbImage {
    if width == 0 || height == 0 {
        return RgbImage::new(width, height);
    }
    if src.width() == 0 || src.height() == 0 {
        return RgbImage::new(width, height);
    }

    match interpolation {
        Interpolation::Nearest => resize_rgb_nearest(src, width, height),
        Interpolation::Linear => resize_rgb_linear(src, width, height),
        Interpolation::Cubic => resize_rgb_sampled(src, width, height, 2, cubic_weights),
        Interpolation::Lanczos => resize_rgb_sampled(src, width, height, 4, lanczos_weights),
        Interpolation::Area => resize_rgb_area(src, width, height),
    }
}

/// Nearest-neighbour resampling of an RGB image (`resize_nearest` for gray).
fn resize_rgb_nearest(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let mut dst = RgbImage::new(width, height);
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;
    let dst_width = width as f32;
    let dst_height = height as f32;

    dst.as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                // OpenCV's INTER_NEAREST floors `x / (nw / w)` - the scale as a
                // reciprocal, in f32. Measured against cv2 over 293 samples of
                // 12 (w, nw) pairs, this is the only form that matches every
                // one: the multiply forms `(x*w)/nw`, `x*(w/nw)` (f32 and f64)
                // all give 27.0 at the 36->40 y=30 tie where cv2 picks 26, and
                // the half-pixel form is wrong in 101 of 293. Nearest is NOT
                // half-pixel; only the interpolating modes are.
                let sx = ((x as f32 / (dst_width / src_width)).floor() as u32).min(src.width() - 1);
                let sy =
                    ((y as f32 / (dst_height / src_height)).floor() as u32).min(src.height() - 1);
                let pixel = src.get_pixel(sx, sy);
                row[x as usize * 3] = pixel[0];
                row[x as usize * 3 + 1] = pixel[1];
                row[x as usize * 3 + 2] = pixel[2];
            }
        });

    dst
}

fn resize_rgb_linear(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let mut dst = RgbImage::new(width, height);
    let src_width = src.width() as f32;
    let src_height = src.height() as f32;
    let dst_width = width.max(1) as f32;
    let dst_height = height.max(1) as f32;

    // Only an empty source has nothing to sample; single-row/column sources
    // are replicated by the mapping below (see `resize_cpu_hal`).
    if src.width() == 0 || src.height() == 0 {
        return dst;
    }

    dst.as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let fx = ((x as f32 + 0.5) * src_width / dst_width - 0.5).max(0.0);
                let fy = ((y as f32 + 0.5) * src_height / dst_height - 0.5).max(0.0);

                let x0 = fx as u32;
                let y0 = fy as u32;
                let x1 = (x0 + 1).min(src.width() - 1);
                let y1 = (y0 + 1).min(src.height() - 1);

                let dx = fx - x0 as f32;
                let dy = fy - y0 as f32;

                for c in 0..3 {
                    let v00 = src.get_pixel(x0, y0)[c] as f32;
                    let v10 = src.get_pixel(x1, y0)[c] as f32;
                    let v01 = src.get_pixel(x0, y1)[c] as f32;
                    let v11 = src.get_pixel(x1, y1)[c] as f32;

                    let v0 = v00 * (1.0 - dx) + v10 * dx;
                    let v1 = v01 * (1.0 - dx) + v11 * dx;
                    let v = v0 * (1.0 - dy) + v1 * dy;

                    row[x as usize * 3 + c] = v.round_ties_even().clamp(0.0, 255.0) as u8;
                }
            }
        });

    dst
}

/// Cubic weights for the taps `floor(f) - 1 ..= floor(f) + 2` (sums to 1).
fn cubic_weights(f: f32) -> Vec<f32> {
    let t = f - f.floor();
    vec![
        cubic_kernel(t + 1.0),
        cubic_kernel(t),
        cubic_kernel(1.0 - t),
        cubic_kernel(2.0 - t),
    ]
}

/// Lanczos-4 weights for the taps `floor(f) - 3 ..= floor(f) + 4`
/// (OpenCV's `INTER_LANCZOS4` window), normalized so that they sum to 1.
fn lanczos_weights(f: f32) -> Vec<f32> {
    let t = f - f.floor();
    let mut weights: Vec<f32> = (-3..=4).map(|i| lanczos_kernel(i as f32 - t)).collect();
    let sum: f32 = weights.iter().sum();
    if sum != 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    weights
}

/// Separable resampling of a `channels`-channel interleaved u8 buffer.
///
/// The coordinate mapping matches the HAL bilinear resampler (`resize_cpu_hal`):
/// half-pixel, `(x + 0.5) * w / nw - 0.5` (OpenCV `resize` semantics), so
/// `Cubic` and `Lanczos` sample the same geometry as the bilinear path.
/// Taps outside the source replicate the nearest edge sample.
fn resample_kernel(
    src: &[u8],
    src_size: (u32, u32),
    dst_size: (u32, u32),
    channels: usize,
    radius: i32,
    weights: fn(f32) -> Vec<f32>,
) -> Vec<u8> {
    debug_assert!(channels <= 4);
    let (src_w, src_h) = src_size;
    let (dst_w, dst_h) = dst_size;
    let mut dst = vec![0u8; (dst_w * dst_h) as usize * channels];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return dst;
    }

    let src_width = src_w as f32;
    let src_height = src_h as f32;
    let dst_width = dst_w.max(1) as f32;
    let dst_height = dst_h.max(1) as f32;
    let row_len = dst_w as usize * channels;

    dst.par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = (y as f32 + 0.5) * src_height / dst_height - 0.5;
            let y0 = fy.floor() as i32;
            let wy = weights(fy);

            for x in 0..dst_w as usize {
                let fx = (x as f32 + 0.5) * src_width / dst_width - 0.5;
                let x0 = fx.floor() as i32;
                let wx = weights(fx);

                let mut acc = [0.0f32; 4];
                for (i, &wxi) in wx.iter().enumerate() {
                    let sx = (x0 - radius + 1 + i as i32).clamp(0, src_w as i32 - 1) as usize;
                    for (j, &wyj) in wy.iter().enumerate() {
                        let sy = (y0 - radius + 1 + j as i32).clamp(0, src_h as i32 - 1) as usize;
                        let w = wxi * wyj;
                        let base = (sy * src_w as usize + sx) * channels;
                        for c in 0..channels {
                            acc[c] += src[base + c] as f32 * w;
                        }
                    }
                }
                for c in 0..channels {
                    row[x * channels + c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });

    dst
}

fn resize_sampled(
    src: &GrayImage,
    width: u32,
    height: u32,
    radius: i32,
    weights: fn(f32) -> Vec<f32>,
) -> GrayImage {
    let data = resample_kernel(
        src.as_raw(),
        (src.width(), src.height()),
        (width, height),
        1,
        radius,
        weights,
    );
    GrayImage::from_raw(width, height, data).unwrap_or_else(|| GrayImage::new(width, height))
}

fn resize_rgb_sampled(
    src: &RgbImage,
    width: u32,
    height: u32,
    radius: i32,
    weights: fn(f32) -> Vec<f32>,
) -> RgbImage {
    let data = resample_kernel(
        src.as_raw(),
        (src.width(), src.height()),
        (width, height),
        3,
        radius,
        weights,
    );
    RgbImage::from_raw(width, height, data).unwrap_or_else(|| RgbImage::new(width, height))
}

/// OpenCV `borderInterpolate(p, len, BORDER_REFLECT_101)`.
fn reflect101(p: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let n = len as isize;
    let mut p = p;
    while p < 0 || p >= n {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * n - 2 - p;
        }
    }
    p as usize
}

/// Gaussian pyramid down-step, `cv2.pyrDown(src)`.
///
/// Output is `((w + 1) / 2, (h + 1) / 2)`. Each output pixel `(x, y)` is the
/// 5x5 `[1 4 6 4 1]^T [1 4 6 4 1] / 256` filter at source `(2x, 2y)`,
/// BORDER_REFLECT_101, integer arithmetic rounded half up as in OpenCV.
pub fn pyr_down(src: &GrayImage) -> GrayImage {
    use rayon::prelude::*;
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (dw, dh) = (sw.div_ceil(2), sh.div_ceil(2));
    if dw == 0 || dh == 0 {
        return GrayImage::new(dw as u32, dh as u32);
    }
    const K: [u32; 5] = [1, 4, 6, 4, 1];
    let data = src.as_raw();
    // Column taps for every output x, reflected once.
    let xtab: Vec<[usize; 5]> = (0..dw)
        .map(|x| std::array::from_fn(|k| reflect101(2 * x as isize + k as isize - 2, sw)))
        .collect();
    // Horizontal pass on the source rows (only every row is needed once).
    let mut rows = vec![0u32; sh * dw];
    rows.par_chunks_mut(dw).enumerate().for_each(|(y, out)| {
        let r = &data[y * sw..(y + 1) * sw];
        for (x, o) in out.iter_mut().enumerate() {
            *o = xtab[x].iter().zip(K).map(|(&i, k)| r[i] as u32 * k).sum();
        }
    });
    let mut dst = GrayImage::new(dw as u32, dh as u32);
    dst.as_mut()
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(y, out)| {
            let ys: [usize; 5] =
                std::array::from_fn(|k| reflect101(2 * y as isize + k as isize - 2, sh));
            for (x, o) in out.iter_mut().enumerate() {
                let s: u32 = ys.iter().zip(K).map(|(&yy, k)| rows[yy * dw + x] * k).sum();
                *o = ((s + 128) >> 8) as u8;
            }
        });
    dst
}

/// Gaussian pyramid up-step, `cv2.pyrUp(src)`.
///
/// Output is `(2w, 2h)`: zero-insertion upsampling filtered with
/// `[1 4 6 4 1] / 16` per axis (scaled by 4), i.e. even outputs
/// `(s[i-1] + 6 s[i] + s[i+1]) / 8` and odd outputs `(s[i] + s[i+1]) / 2`
/// per axis, rounded half up. As in OpenCV, the missing neighbour is
/// reflected at the start (`s[-1] = s[1]`) and repeated at the end
/// (`s[n] = s[n-1]`).
pub fn pyr_up(src: &GrayImage) -> GrayImage {
    use rayon::prelude::*;
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (dw, dh) = (sw * 2, sh * 2);
    if dw == 0 || dh == 0 {
        return GrayImage::new(dw as u32, dh as u32);
    }
    // Source index of neighbour i - 1 / i + 1, OpenCV style:
    // borderInterpolate(2i, 2n, REFLECT_101) / 2.
    let nb = |i: isize, n: usize| reflect101(2 * i, 2 * n) / 2;
    let data = src.as_raw();
    // Horizontal pass: each source row -> 2w values (x8).
    let mut rows = vec![0u32; sh * dw];
    rows.par_chunks_mut(dw).enumerate().for_each(|(y, out)| {
        let r = &data[y * sw..(y + 1) * sw];
        for x in 0..sw {
            let l = r[nb(x as isize - 1, sw)] as u32;
            let c = r[x] as u32;
            let rr = r[nb(x as isize + 1, sw)] as u32;
            out[2 * x] = l + 6 * c + rr;
            out[2 * x + 1] = (c + rr) * 4;
        }
    });
    let mut dst = GrayImage::new(dw as u32, dh as u32);
    dst.as_mut()
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(dy, out)| {
            let y = dy / 2;
            let up = nb(y as isize - 1, sh);
            let dn = nb(y as isize + 1, sh);
            for (x, o) in out.iter_mut().enumerate() {
                let c = rows[y * dw + x];
                let d = rows[dn * dw + x];
                let s = if dy % 2 == 0 {
                    rows[up * dw + x] + 6 * c + d
                } else {
                    (c + d) * 4
                };
                *o = ((s + 32) >> 6) as u8;
            }
        });
    dst
}

/// Gaussian pyramid: `levels` images, each `pyr_down` of the previous,
/// stopping early once a level is 1 pixel wide or tall.
pub fn build_pyramid(src: &GrayImage, levels: u32) -> Vec<GrayImage> {
    let mut pyramid = vec![src.clone()];

    for _ in 1..levels {
        let prev = pyramid.last().unwrap();
        if prev.width() < 2 || prev.height() < 2 {
            break;
        }
        pyramid.push(pyr_down(prev));
    }

    pyramid
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    #[test]
    fn kernels_are_normalized_and_interpolating() {
        for f in [0.0f32, 0.25, 0.5, 0.75, 1.5, 2.25] {
            let c = cubic_weights(f);
            assert_eq!(c.len(), 4);
            assert!((c.iter().sum::<f32>() - 1.0).abs() < 1e-5, "cubic @ {f}");

            let l = lanczos_weights(f);
            assert_eq!(l.len(), 8);
            assert!((l.iter().sum::<f32>() - 1.0).abs() < 1e-5, "lanczos @ {f}");
        }

        // At integer coordinates both kernels collapse onto the sample.
        for (i, &w) in cubic_weights(3.0).iter().enumerate() {
            let expected = if i == 1 { 1.0 } else { 0.0 };
            assert!((w - expected).abs() < 1e-5);
        }
        for (i, &w) in lanczos_weights(3.0).iter().enumerate() {
            let expected = if i == 3 { 1.0 } else { 0.0 };
            assert!((w - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn resize_dispatches_nearest() {
        // 2x1 image: nearest must keep the exact source levels, while the
        // bilinear path mixes them. Previously every variant returned the
        // bilinear result.
        let mut img = GrayImage::new(2, 1);
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(1, 0, Luma([255]));

        let nearest = resize(&img, 4, 1, Interpolation::Nearest);
        assert_eq!(nearest.as_raw(), &[0, 0, 255, 255]);

        let linear = resize(&img, 4, 1, Interpolation::Linear);
        assert_ne!(nearest.as_raw(), linear.as_raw());
    }

    #[test]
    fn resize_rgb_dispatches_nearest() {
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([0, 10, 20]));
        img.put_pixel(1, 0, Rgb([200, 210, 220]));

        let nearest = resize_rgb(&img, 4, 1, Interpolation::Nearest);
        assert_eq!(
            nearest.as_raw().as_slice(),
            &[0, 10, 20, 0, 10, 20, 200, 210, 220, 200, 210, 220]
        );

        let linear = resize_rgb(&img, 4, 1, Interpolation::Linear);
        assert_ne!(nearest.as_raw(), linear.as_raw());
    }

    /// A one-pixel spike must be averaged across the pixels it shares an
    /// output with. Bilinear samples the spike and reports 255.
    #[test]
    fn area_downscale_averages_what_linear_aliases() {
        let mut img = GrayImage::new(3, 1);
        img.put_pixel(0, 0, Luma([0]));
        img.put_pixel(1, 0, Luma([255]));
        img.put_pixel(2, 0, Luma([0]));
        assert_eq!(resize(&img, 1, 1, Interpolation::Area).as_raw(), &[85]);
        assert_eq!(resize(&img, 1, 1, Interpolation::Linear).as_raw(), &[255]);

        let mut block = GrayImage::new(2, 2);
        block.put_pixel(0, 0, Luma([10]));
        block.put_pixel(1, 0, Luma([20]));
        block.put_pixel(0, 1, Luma([30]));
        block.put_pixel(1, 1, Luma([40]));
        assert_eq!(resize(&block, 1, 1, Interpolation::Area).as_raw(), &[25]);
        // A zoom puts each output pixel inside one source pixel.
        assert_eq!(
            resize(&block, 4, 4, Interpolation::Area).as_raw(),
            resize(&block, 4, 4, Interpolation::Nearest).as_raw()
        );

        let mut rgb = RgbImage::new(2, 2);
        rgb.put_pixel(0, 0, Rgb([0, 0, 0]));
        rgb.put_pixel(1, 0, Rgb([10, 20, 30]));
        rgb.put_pixel(0, 1, Rgb([40, 50, 60]));
        rgb.put_pixel(1, 1, Rgb([70, 80, 90]));
        assert_eq!(
            resize_rgb(&rgb, 1, 1, Interpolation::Area).as_raw(),
            &[30, 38, 45]
        );
    }

    #[test]
    fn cubic_and_lanczos_reproduce_a_linear_ramp() {
        // Both kernels are interpolating, so a ramp resampled with them must
        // stay within the rounding error of the bilinear result.
        let ramp = GrayImage::from_fn(6, 6, |x, _| Luma([(x * 25) as u8]));
        let linear = resize(&ramp, 15, 15, Interpolation::Linear);

        for interp in [Interpolation::Cubic, Interpolation::Lanczos] {
            let out = resize(&ramp, 15, 15, interp);
            assert_eq!(out.dimensions(), (15, 15));
            for (a, b) in out.as_raw().iter().zip(linear.as_raw()) {
                assert!(
                    (*a as i32 - *b as i32).abs() <= 3,
                    "{interp:?} deviates from the ramp: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn single_pixel_source_is_replicated() {
        // Regression: a 1x1 source hit the `src_width <= 0.0` guard and came
        // back as an all-zero (black) image.
        let img = GrayImage::from_pixel(1, 1, Luma([7]));
        for interp in [
            Interpolation::Nearest,
            Interpolation::Linear,
            Interpolation::Cubic,
            Interpolation::Lanczos,
        ] {
            let out = resize(&img, 4, 3, interp);
            assert_eq!(out.dimensions(), (4, 3));
            assert!(
                out.as_raw().iter().all(|&p| p == 7),
                "{interp:?} did not replicate the single sample"
            );
        }

        let rgb = RgbImage::from_pixel(1, 1, Rgb([9, 8, 7]));
        let out_rgb = resize_rgb(&rgb, 3, 3, Interpolation::Linear);
        assert!(out_rgb.pixels().all(|p| p.0 == [9, 8, 7]));
    }

    #[test]
    fn single_row_source_is_replicated_along_that_axis() {
        // A 1xN source has src_height == 0: the destination must reuse the
        // only row instead of returning black.
        let img = GrayImage::from_fn(4, 1, |x, _| Luma([(x * 20) as u8]));
        let up = resize(&img, 4, 3, Interpolation::Linear);
        assert_eq!(up.dimensions(), (4, 3));
        for y in 0..3 {
            assert_eq!(up.get_pixel(3, y)[0], img.get_pixel(3, 0)[0]);
        }
    }

    /// Independent reference resampler for the HAL-routed bilinear path, in the
    /// same half-pixel convention `resize_cpu_hal` documents.
    fn reference_resize_linear(src: &GrayImage, width: u32, height: u32) -> GrayImage {
        let mut dst = GrayImage::new(width, height);
        let src_width = src.width() as f32;
        let src_height = src.height() as f32;
        let dst_width = width.max(1) as f32;
        let dst_height = height.max(1) as f32;
        if src.width() == 0 || src.height() == 0 {
            return dst;
        }
        for y in 0..height {
            for x in 0..width {
                let fx = ((x as f32 + 0.5) * src_width / dst_width - 0.5).max(0.0);
                let fy = ((y as f32 + 0.5) * src_height / dst_height - 0.5).max(0.0);
                let x0 = fx as u32;
                let y0 = fy as u32;
                let x1 = (x0 + 1).min(src.width() - 1);
                let y1 = (y0 + 1).min(src.height() - 1);
                let dx = fx - x0 as f32;
                let dy = fy - y0 as f32;
                let v00 = src.get_pixel(x0, y0)[0] as f32;
                let v10 = src.get_pixel(x1, y0)[0] as f32;
                let v01 = src.get_pixel(x0, y1)[0] as f32;
                let v11 = src.get_pixel(x1, y1)[0] as f32;
                let v0 = v00 * (1.0 - dx) + v10 * dx;
                let v1 = v01 * (1.0 - dx) + v11 * dx;
                let v = v0 * (1.0 - dy) + v1 * dy;
                dst.put_pixel(x, y, Luma([(v + 0.5).clamp(0.0, 255.0) as u8]));
            }
        }
        dst
    }

    #[test]
    fn hal_linear_resize_matches_reference() {
        // Deterministic textured source.
        let src = GrayImage::from_fn(37, 29, |x, y| Luma([((x * 7 + y * 13) % 256) as u8]));

        let mut max_diff = 0i32;
        for &(w, h) in &[
            (37u32, 29u32),
            (74, 15),
            (19, 58),
            (64, 64),
            (1, 1),
            (5, 3),
            (100, 7),
        ] {
            let new = resize(&src, w, h, Interpolation::Linear);
            let reference = reference_resize_linear(&src, w, h);
            assert_eq!(new.dimensions(), reference.dimensions());
            for (a, b) in new.as_raw().iter().zip(reference.as_raw()) {
                max_diff = max_diff.max((*a as i32 - *b as i32).abs());
            }
        }
        assert_eq!(
            max_diff, 0,
            "HAL-routed bilinear differs from the reference by up to {max_diff}"
        );
    }
}
