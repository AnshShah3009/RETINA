use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::ResourceGroup;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

use crate::kernels::{cubic_kernel, lanczos_kernel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Linear,
    Cubic,
    Lanczos,
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
/// source, `Cubic` uses the Catmull-Rom kernel and `Lanczos` the Lanczos-3
/// kernel (both separable and edge-clamped).
fn resize_cpu(src: &GrayImage, width: u32, height: u32, interpolation: Interpolation) -> GrayImage {
    if width == 0 || height == 0 {
        return GrayImage::new(0, 0);
    }
    if src.width() == 0 || src.height() == 0 {
        return GrayImage::new(width, height);
    }
    match interpolation {
        Interpolation::Nearest => resize_nearest(src, width, height),
        Interpolation::Linear => resize_linear(src, width, height),
        Interpolation::Cubic => resize_sampled(src, width, height, 2, cubic_weights),
        Interpolation::Lanczos => resize_sampled(src, width, height, 3, lanczos_weights),
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
        return GrayImage::new(0, 0);
    }
    if src.width() == 0 || src.height() == 0 {
        return GrayImage::new(width, height);
    }

    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if interpolation == Interpolation::Linear {
            if let Ok(result) = resize_gpu(gpu, src, width, height) {
                return result;
            }
        }
    }

    group.run(|| resize_cpu(src, width, height, interpolation))
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
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(width, height, data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
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
                let sx = ((x as f32 * src_width / dst_width).floor() as u32).min(src.width() - 1);
                let sy =
                    ((y as f32 * src_height / dst_height).floor() as u32).min(src.height() - 1);
                let val = src.get_pixel(sx, sy)[0];
                row[x as usize] = val;
            }
        });
    dst
}

fn resize_linear(src: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut dst = GrayImage::new(width, height);
    let src_width = src.width() as f32 - 1.0;
    let src_height = src.height() as f32 - 1.0;
    // Guard against 1-pixel destinations: (n-1) would be zero and the
    // coordinate mapping below would produce NaN (black output).
    let dst_width = (width.max(2) - 1) as f32;
    let dst_height = (height.max(2) - 1) as f32;

    // Only an empty source has nothing to sample. A source with a single
    // row/column (src_width or src_height == 0) is handled by the mapping
    // below, which collapses that axis onto the single sample and therefore
    // replicates it; a previous revision bailed out here and returned an
    // all-zero image for 1-pixel sources.
    if src.width() == 0 || src.height() == 0 {
        return dst;
    }

    dst.as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let fx = (x as f32 / dst_width) * src_width;
                let fy = (y as f32 / dst_height) * src_height;

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

                row[x as usize] = (v + 0.5).clamp(0.0, 255.0) as u8;
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
        return RgbImage::new(0, 0);
    }
    if src.width() == 0 || src.height() == 0 {
        return RgbImage::new(width, height);
    }

    match interpolation {
        Interpolation::Nearest => resize_rgb_nearest(src, width, height),
        Interpolation::Linear => resize_rgb_linear(src, width, height),
        Interpolation::Cubic => resize_rgb_sampled(src, width, height, 2, cubic_weights),
        Interpolation::Lanczos => resize_rgb_sampled(src, width, height, 3, lanczos_weights),
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
                let sx = ((x as f32 * src_width / dst_width).floor() as u32).min(src.width() - 1);
                let sy =
                    ((y as f32 * src_height / dst_height).floor() as u32).min(src.height() - 1);
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
    let src_width = src.width() as f32 - 1.0;
    let src_height = src.height() as f32 - 1.0;
    // Guard against 1-pixel destinations (see resize_linear).
    let dst_width = (width.max(2) - 1) as f32;
    let dst_height = (height.max(2) - 1) as f32;

    // Only an empty source has nothing to sample; single-row/column sources
    // are replicated by the mapping below (see resize_linear).
    if src.width() == 0 || src.height() == 0 {
        return dst;
    }

    dst.as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let fx = (x as f32 / dst_width) * src_width;
                let fy = (y as f32 / dst_height) * src_height;

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

                    row[x as usize * 3 + c] = v.clamp(0.0, 255.0) as u8;
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

/// Lanczos-3 weights for the taps `floor(f) - 2 ..= floor(f) + 3`,
/// normalized so that they sum to 1.
fn lanczos_weights(f: f32) -> Vec<f32> {
    let t = f - f.floor();
    let mut weights: Vec<f32> = (-2..=3).map(|i| lanczos_kernel(i as f32 - t)).collect();
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
/// The coordinate mapping matches `resize_linear`: the source spans
/// `0 ..= len - 1` across the destination's `0 ..= len - 1`, so `Cubic` and
/// `Lanczos` degrade gracefully to the same geometry as the bilinear path.
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

    let src_width = src_w as f32 - 1.0;
    let src_height = src_h as f32 - 1.0;
    let dst_width = (dst_w.max(2) - 1) as f32;
    let dst_height = (dst_h.max(2) - 1) as f32;
    let row_len = dst_w as usize * channels;

    dst.par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = (y as f32 / dst_height) * src_height;
            let y0 = fy.floor() as i32;
            let wy = weights(fy);

            for x in 0..dst_w as usize {
                let fx = (x as f32 / dst_width) * src_width;
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

pub fn pyr_down(src: &GrayImage) -> GrayImage {
    let new_width = src.width() / 2;
    let new_height = src.height() / 2;
    if new_width == 0 || new_height == 0 {
        return GrayImage::new(1, 1);
    }
    resize(src, new_width, new_height, Interpolation::Linear)
}

pub fn pyr_up(src: &GrayImage) -> GrayImage {
    let new_width = src.width() * 2;
    let new_height = src.height() * 2;
    resize(src, new_width, new_height, Interpolation::Linear)
}

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
            assert_eq!(l.len(), 6);
            assert!((l.iter().sum::<f32>() - 1.0).abs() < 1e-5, "lanczos @ {f}");
        }

        // At integer coordinates both kernels collapse onto the sample.
        for (i, &w) in cubic_weights(3.0).iter().enumerate() {
            let expected = if i == 1 { 1.0 } else { 0.0 };
            assert!((w - expected).abs() < 1e-5);
        }
        for (i, &w) in lanczos_weights(3.0).iter().enumerate() {
            let expected = if i == 2 { 1.0 } else { 0.0 };
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
}
