use cv_core::{storage::Storage, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;
use wide::*;

use crate::convolve::{convolve_with_border_into_ctx, gaussian_blur_ctx, BorderMode, Kernel};

fn sobel_kernels_1d(ksize: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    match ksize {
        3 => Some((vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])),
        5 => Some((
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        )),
        7 => Some((
            vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
            vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        )),
        _ => None,
    }
}

fn scharr_kernels_1d() -> (Vec<f32>, Vec<f32>) {
    (vec![-1.0, 0.0, 1.0], vec![3.0, 10.0, 3.0])
}

fn kernel_from_1d(kx: &[f32], ky: &[f32]) -> Kernel {
    let mut data = vec![0.0f32; kx.len() * ky.len()];
    for (y, &vy) in ky.iter().enumerate() {
        for (x, &vx) in kx.iter().enumerate() {
            data[y * kx.len() + x] = vx * vy;
        }
    }
    Kernel::new(data, kx.len(), ky.len())
}

fn apply_linear_transform(mut img: GrayImage, scale: f32, delta: f32) -> GrayImage {
    img.as_mut().par_iter_mut().for_each(|px| {
        let v = (*px as f32) * scale + delta;
        *px = v.clamp(0.0, 255.0) as u8;
    });
    img
}

pub fn sobel_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> GrayImage {
    match cv_runtime::best_runner() {
        Ok(runner) => sobel_ex_ctx(src, dx, dy, ksize, scale, delta, border, &runner),
        Err(_) => {
            // Fallback: compute on CPU
            sobel_ex_ctx(
                src,
                dx,
                dy,
                ksize,
                scale,
                delta,
                border,
                &cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)),
            )
        }
    }
}

pub fn sobel_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if dx == 1 && dy == 1 && ksize == 3 {
            if let Ok(result) = sobel_gpu(gpu, src, ksize) {
                let (gx, gy) = result;
                let target = if dx > 0 { gx } else { gy };
                return apply_linear_transform(target, scale, delta);
            }
        }
    }

    let mut out = GrayImage::new(src.width(), src.height());
    let (deriv, smooth) =
        sobel_kernels_1d(ksize).unwrap_or_else(|| (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]));

    let kx = if dx > 0 {
        deriv.as_slice()
    } else {
        smooth.as_slice()
    };
    let ky = if dy > 0 {
        deriv.as_slice()
    } else {
        smooth.as_slice()
    };

    // Use separable convolution for performance
    crate::convolve::separable_convolve_into_ctx(src, &mut out, kx, ky, border, group);
    apply_linear_transform(out, scale, delta)
}

fn sobel_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    ksize: usize,
) -> cv_hal::Result<(GrayImage, GrayImage)> {
    use cv_hal::context::ComputeContext;
    let input_tensor = cv_core::CpuTensor::from_vec(
        src.as_raw().to_vec(),
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let (gx_gpu, gy_gpu) = gpu.sobel(&input_gpu, 1, 1, ksize)?;

    let gx_cpu: Tensor<u8, cv_core::CpuStorage<u8>> = gx_gpu.to_cpu()?;
    let gy_cpu: Tensor<u8, cv_core::CpuStorage<u8>> = gy_gpu.to_cpu()?;

    let gx_data = gx_cpu.storage.as_slice().unwrap().to_vec();
    let gy_data = gy_cpu.storage.as_slice().unwrap().to_vec();

    let gx = GrayImage::from_raw(src.width(), src.height(), gx_data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))?;
    let gy = GrayImage::from_raw(src.width(), src.height(), gy_data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))?;

    Ok((gx, gy))
}

pub fn scharr_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> GrayImage {
    match cv_runtime::best_runner() {
        Ok(runner) => scharr_ex_ctx(src, dx, dy, scale, delta, border, &runner),
        Err(_) => {
            // Fallback: compute on CPU
            scharr_ex_ctx(
                src,
                dx,
                dy,
                scale,
                delta,
                border,
                &cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)),
            )
        }
    }
}

pub fn scharr_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    let mut out = GrayImage::new(src.width(), src.height());
    let (deriv, smooth) = scharr_kernels_1d();
    let kx = if dx > 0 {
        deriv.as_slice()
    } else {
        smooth.as_slice()
    };
    let ky = if dy > 0 {
        deriv.as_slice()
    } else {
        smooth.as_slice()
    };
    let kernel = kernel_from_1d(kx, ky);

    convolve_with_border_into_ctx(src, &mut out, &kernel, border, group);
    apply_linear_transform(out, scale, delta)
}

pub fn sobel_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    let gx = sobel_ex_ctx(src, 1, 0, 3, 1.0, 0.0, border, &runner);
    let gy = sobel_ex_ctx(src, 0, 1, 3, 1.0, 0.0, border, &runner);
    (gx, gy)
}

pub fn sobel(src: &GrayImage) -> (GrayImage, GrayImage) {
    sobel_with_border(src, BorderMode::Replicate)
}

pub fn scharr_with_border(src: &GrayImage, border: BorderMode) -> (GrayImage, GrayImage) {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    let gx = scharr_ex_ctx(src, 1, 0, 1.0, 0.0, border, &runner);
    let gy = scharr_ex_ctx(src, 0, 1, 1.0, 0.0, border, &runner);
    (gx, gy)
}

pub fn scharr(src: &GrayImage) -> (GrayImage, GrayImage) {
    scharr_with_border(src, BorderMode::Replicate)
}

pub fn sobel_magnitude(gx: &GrayImage, gy: &GrayImage) -> GrayImage {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    sobel_magnitude_ctx(gx, gy, &runner)
}

pub fn sobel_magnitude_ctx(gx: &GrayImage, gy: &GrayImage, group: &RuntimeRunner) -> GrayImage {
    group.run(|| {
        let width = gx.width();
        let height = gx.height();
        let count = (width * height) as usize;
        let mut output = vec![0u8; count];

        output
            .par_iter_mut()
            .zip(gx.as_raw().par_iter())
            .zip(gy.as_raw().par_iter())
            .for_each(|((out, &gx_val), &gy_val)| {
                let gx_f = gx_val as f32;
                let gy_f = gy_val as f32;
                let mag = (gx_f * gx_f + gy_f * gy_f).sqrt();
                *out = mag.min(255.0) as u8;
            });

        GrayImage::from_raw(width, height, output).unwrap_or_else(|| GrayImage::new(width, height))
    })
}

pub fn laplacian(src: &GrayImage) -> GrayImage {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    laplacian_ctx(src, &runner)
}

pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> GrayImage {
    let kernel = Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
    let mut out = GrayImage::new(src.width(), src.height());
    convolve_with_border_into_ctx(src, &mut out, &kernel, BorderMode::Replicate, group);
    out
}

fn gradients_and_directions(src: &GrayImage) -> (Vec<f32>, Vec<u8>) {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let data = src.as_raw();
    let mut magnitude = vec![0.0f32; width * height];
    let mut direction = vec![0u8; width * height];

    magnitude
        .par_chunks_mut(width)
        .zip(direction.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (mag_row, dir_row))| {
            if y == 0 || y >= height - 1 {
                return;
            }
            let r0_idx = (y - 1) * width;
            let r1_idx = y * width;
            let r2_idx = (y + 1) * width;
            let r0 = &data[r0_idx..r0_idx + width];
            let r1 = &data[r1_idx..r1_idx + width];
            let r2 = &data[r2_idx..r2_idx + width];

            let mut x = 1;

            // SIMD Loop
            while x + 8 < width - 1 {
                // Load 3x10 block to cover 8 outputs (needs x-1 to x+8+1)
                // We load 8-wide vectors shifted by -1, 0, 1
                // Actually easier to load unaligned? f32x8 requires alignment? No, but load is from u8.
                // We load u8 slices and convert.

                let load_f32x8 = |slice: &[u8]| -> f32x8 {
                    let mut arr = [0.0f32; 8];
                    for i in 0..8 {
                        arr[i] = slice[i] as f32;
                    }
                    f32x8::from(arr)
                };

                let p00 = load_f32x8(&r0[x - 1..]);
                let p01 = load_f32x8(&r0[x..]);
                let p02 = load_f32x8(&r0[x + 1..]);

                let p10 = load_f32x8(&r1[x - 1..]);
                // p11 unused for gradients
                let p12 = load_f32x8(&r1[x + 1..]);

                let p20 = load_f32x8(&r2[x - 1..]);
                let p21 = load_f32x8(&r2[x..]);
                let p22 = load_f32x8(&r2[x + 1..]);

                // Sobel
                let gx = p02 - p00 + (p12 - p10) * 2.0 + p22 - p20;
                let gy = p20 - p00 + (p21 - p01) * 2.0 + p22 - p02;

                // Magnitude
                let mag = (gx * gx + gy * gy).sqrt();
                let mag_arr: [f32; 8] = mag.into();
                mag_row[x..x + 8].copy_from_slice(&mag_arr);

                // Direction
                // 0: |gy| <= |gx| * tan(22.5) => 0.4142
                // 90: |gx| <= |gy| * 0.4142
                // 45: sign(gx) == sign(gy)
                // 135: else

                let abs_gx = gx.abs();
                let abs_gy = gy.abs();
                let tan_22_5 = 0.41421356;

                let is_0 = abs_gy.cmp_lt(abs_gx * tan_22_5);
                let is_90 = abs_gx.cmp_lt(abs_gy * tan_22_5);

                // For diagonal check:
                // sign bit check. f32x8 doesn't have direct sign access efficiently?
                // gx * gy > 0 means same sign.
                let same_sign = (gx * gy).cmp_gt(f32x8::ZERO);

                // Blend chain
                // Default 135 (3)
                // if same_sign -> 45 (1)
                // if is_90 -> 90 (2)
                // if is_0 -> 0 (0)
                // Order matters? 0 and 90 take precedence over diagonals in ambiguous near-center cases?
                // Logic:
                // Region 0: [-22.5, 22.5] -> |y| < |x| * tan
                // Region 90: [67.5, 112.5] -> |x| < |y| * cot(67.5) = |y| * tan(22.5)
                // So is_0 and is_90 checks are correct and exclusive.

                // Values: 0=0, 1=45, 2=90, 3=135
                let val_diag = same_sign.blend(f32x8::splat(1.0), f32x8::splat(3.0));
                let val_90 = is_90.blend(f32x8::splat(2.0), val_diag);
                let val_0 = is_0.blend(f32x8::splat(0.0), val_90);

                let dir_arr: [f32; 8] = val_0.into();
                for i in 0..8 {
                    dir_row[x + i] = dir_arr[i] as u8;
                }

                x += 8;
            }

            // Scalar tail
            for cx in x..width - 1 {
                let p00 = r0[cx - 1] as f32;
                let p02 = r0[cx + 1] as f32;
                let p10 = r1[cx - 1] as f32;
                let p12 = r1[cx + 1] as f32;
                let p20 = r2[cx - 1] as f32;
                let p22 = r2[cx + 1] as f32;
                let p01 = r0[cx] as f32;
                let p21 = r2[cx] as f32;

                let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
                let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

                mag_row[cx] = (gx * gx + gy * gy).sqrt();

                // Scalar direction logic (consistent with SIMD)
                let abs_gx = gx.abs();
                let abs_gy = gy.abs();
                let tan_22_5 = 0.41421356;

                if abs_gy <= abs_gx * tan_22_5 {
                    dir_row[cx] = 0;
                } else if abs_gx <= abs_gy * tan_22_5 {
                    dir_row[cx] = 2;
                } else if gx * gy > 0.0 {
                    dir_row[cx] = 1;
                } else {
                    dir_row[cx] = 3;
                }
            }
        });

    (magnitude, direction)
}

fn non_max_suppression(width: usize, height: usize, mag: &[f32], dir: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            if y == 0 || y >= height - 1 {
                return;
            }
            let r0_idx = (y - 1) * width;
            let r1_idx = y * width;
            let r2_idx = (y + 1) * width;

            for x in 1..width - 1 {
                let m = mag[r1_idx + x];
                let (m1, m2) = match dir[r1_idx + x] {
                    0 => (mag[r1_idx + x - 1], mag[r1_idx + x + 1]),
                    1 => (mag[r0_idx + x + 1], mag[r2_idx + x - 1]),
                    2 => (mag[r0_idx + x], mag[r2_idx + x]),
                    _ => (mag[r0_idx + x - 1], mag[r2_idx + x + 1]),
                };

                if m >= m1 && m >= m2 {
                    out_row[x] = m;
                }
            }
        });

    out
}

fn hysteresis(width: usize, height: usize, nms: &[f32], low: f32, high: f32) -> GrayImage {
    const STRONG: u8 = 255;
    const WEAK: u8 = 75;

    let mut state = vec![0u8; width * height];
    let mut stack = Vec::new();

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

    let mut out = GrayImage::new(width as u32, height as u32);
    out.as_mut().par_iter_mut().enumerate().for_each(|(i, px)| {
        *px = if state[i] == STRONG { 255 } else { 0 };
    });
    out
}

pub fn canny(src: &GrayImage, low_threshold: u8, high_threshold: u8) -> GrayImage {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    canny_ctx(src, low_threshold, high_threshold, &runner)
}

pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    group: &RuntimeRunner,
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(res) = canny_gpu(gpu, src, low_threshold as f32, high_threshold as f32) {
            return res;
        }
    }

    let blurred = gaussian_blur_ctx(src, 1.0, BorderMode::Reflect101, group);
    let width = blurred.width() as usize;
    let height = blurred.height() as usize;

    group.run(|| {
        let (mag, dir) = gradients_and_directions(&blurred);
        let nms = non_max_suppression(width, height, &mag, &dir);
        let low = low_threshold as f32;
        let high = high_threshold.max(low_threshold) as f32;
        hysteresis(width, height, &nms, low, high)
    })
}

fn canny_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    low: f32,
    high: f32,
) -> cv_hal::Result<GrayImage> {
    use cv_hal::context::ComputeContext;
    let input_tensor = cv_core::CpuTensor::from_vec(
        src.as_raw().to_vec(),
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let res_gpu = gpu.canny(&input_gpu, low, high)?;

    let res_cpu: Tensor<u8, cv_core::CpuStorage<u8>> = res_gpu.to_cpu()?;
    let res_data = res_cpu.storage.as_slice().unwrap().to_vec();

    GrayImage::from_raw(src.width(), src.height(), res_data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn sobel_constant_image_is_zero() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Luma([100]));
            }
        }
        let (gx, gy) = sobel_with_border(&img, BorderMode::Reflect101);
        assert!(gx.as_raw().iter().all(|&v| v == 0));
        assert!(gy.as_raw().iter().all(|&v| v == 0));
    }

    #[test]
    fn sobel_vertical_edge() {
        let mut img = GrayImage::new(16, 16);
        // Left half: black, right half: white
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        let (gx, _gy) = sobel(&img);
        // X-gradient should be high at vertical edges
        assert!(gx.get_pixel(8, 8)[0] > 100);
    }

    #[test]
    fn sobel_horizontal_edge() {
        let mut img = GrayImage::new(16, 16);
        // Top half: black, bottom half: white
        for y in 0..16 {
            for x in 0..16 {
                let val = if y < 8 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        let (_gx, gy) = sobel(&img);
        // Y-gradient should be high at horizontal edges
        assert!(gy.get_pixel(8, 8)[0] > 100);
    }

    #[test]
    fn sobel_magnitude_dimensions() {
        let mut img = GrayImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.put_pixel(x, y, Luma([((x + y) % 256) as u8]));
            }
        }
        let (gx, gy) = sobel(&img);
        let mag = sobel_magnitude(&gx, &gy);

        assert_eq!(mag.width(), 32);
        assert_eq!(mag.height(), 32);
    }

    #[test]
    fn sobel_magnitude_positive_values() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Luma([100]));
            }
        }
        let (gx, gy) = sobel(&img);
        let mag = sobel_magnitude(&gx, &gy);

        // Constant image should have near-zero magnitude
        assert!(mag.as_raw().iter().all(|&v| v < 10));
    }

    #[test]
    fn laplacian_constant_image() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Luma([128]));
            }
        }
        let lap = laplacian(&img);

        // Laplacian of constant image should be zero
        assert!(lap.as_raw().iter().all(|&v| v < 10));
    }

    #[test]
    fn laplacian_edge_detection() {
        let mut img = GrayImage::new(20, 20);
        // Create a white square in the center
        for y in 5..15 {
            for x in 5..15 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let lap = laplacian(&img);

        // Laplacian should respond to edges
        // Should detect some variation at the boundary
        assert!(lap.as_raw().iter().any(|&v| v > 0));
    }

    #[test]
    fn scharr_output_dimensions() {
        let mut img = GrayImage::new(32, 24);
        for y in 0..24 {
            for x in 0..32 {
                img.put_pixel(x, y, Luma([128]));
            }
        }
        let (gx, gy) = scharr(&img);

        assert_eq!(gx.width(), 32);
        assert_eq!(gx.height(), 24);
        assert_eq!(gy.width(), 32);
        assert_eq!(gy.height(), 24);
    }

    #[test]
    fn canny_basic() {
        let mut img = GrayImage::new(32, 32);
        // Create a white square
        for y in 8..24 {
            for x in 8..24 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let edges = canny(&img, 50u8, 150u8);

        assert_eq!(edges.width(), 32);
        assert_eq!(edges.height(), 32);
        // Should detect some edges
        assert!(edges.as_raw().iter().any(|&v| v > 0));
    }

    #[test]
    fn canny_uniform_image() {
        let img = GrayImage::new(32, 32);
        let edges = canny(&img, 50u8, 150u8);

        // Uniform image should have no edges
        assert!(edges.as_raw().iter().all(|&v| v == 0));
    }

    #[test]
    fn canny_threshold_effect() {
        let mut img = GrayImage::new(32, 32);
        for y in 8..24 {
            for x in 8..24 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let edges_low = canny(&img, 10u8, 50u8);
        let edges_high = canny(&img, 100u8, 200u8);

        // Lower threshold should detect more edges
        let low_count = edges_low.as_raw().iter().filter(|&&v| v > 0).count();
        let high_count = edges_high.as_raw().iter().filter(|&&v| v > 0).count();
        assert!(low_count >= high_count);
    }

    #[test]
    fn canny_various_sizes() {
        for size in &[16, 32, 48, 64] {
            let mut img = GrayImage::new(*size, *size);
            for y in 0..*size {
                for x in 0..*size {
                    let val = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
                    img.put_pixel(x, y, Luma([val]));
                }
            }
            let edges = canny(&img, 50u8, 150u8);

            assert_eq!(edges.width(), *size);
            assert_eq!(edges.height(), *size);
        }
    }
}
