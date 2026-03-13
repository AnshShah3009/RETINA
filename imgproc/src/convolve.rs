use crate::simd::convolve_row_1d;
use cv_core::{Tensor, TensorShape};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::BorderMode as HalBorderMode;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::{scheduler, RuntimeRunner};
use image::GrayImage;
use rayon::prelude::*;
use wide::*;

#[derive(Debug, Clone)]
pub struct Kernel {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Kernel {
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
        }
    }

    pub fn from_slice(data: &[f32], width: usize, height: usize) -> Self {
        Self::new(data.to_vec(), width, height)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn center(&self) -> (isize, isize) {
        ((self.width / 2) as isize, (self.height / 2) as isize)
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub fn normalize(&mut self) {
        let sum: f32 = self.data.iter().sum();
        if sum != 0.0 {
            for v in &mut self.data {
                *v /= sum;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    Constant(u8),
    Replicate,
    Reflect,
    Reflect101,
    Wrap,
}

pub fn box_kernel(size: usize) -> Kernel {
    let value = 1.0 / (size * size) as f32;
    Kernel::new(vec![value; size * size], size, size)
}

pub fn gaussian_kernel(sigma: f32, size: usize) -> Kernel {
    let mut data = Vec::with_capacity(size * size);
    let center = size / 2;
    let sigma2 = sigma * sigma;
    let mut sum = 0.0f32;

    for y in 0..size {
        for x in 0..size {
            let dx = (x as isize - center as isize).abs() as f32;
            let dy = (y as isize - center as isize).abs() as f32;
            let v = (-(dx * dx + dy * dy) / (2.0 * sigma2)).exp();
            data.push(v);
            sum += v;
        }
    }

    for v in &mut data {
        *v /= sum;
    }

    Kernel::new(data, size, size)
}

pub fn gaussian_kernel_1d(sigma: f32, size: usize) -> Vec<f32> {
    assert!(size % 2 == 1, "gaussian kernel size must be odd");
    let mut kernel = Vec::with_capacity(size);
    let center = (size / 2) as isize;
    let sigma2 = sigma * sigma;
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = (i as isize - center) as f32;
        let v = (-(x * x) / (2.0 * sigma2)).exp();
        kernel.push(v);
        sum += v;
    }

    if sum != 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }

    kernel
}

pub fn sobel_kernel() -> (Kernel, Kernel) {
    let gx = Kernel::from_slice(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3, 3);
    let gy = Kernel::from_slice(&[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3, 3);
    (gx, gy)
}

pub fn laplacian_kernel() -> Kernel {
    Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3)
}

fn map_coord(coord: isize, len: usize, mode: BorderMode) -> Option<usize> {
    let n = len as isize;
    if n <= 0 {
        return None;
    }

    match mode {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= n {
                None
            } else {
                Some(coord as usize)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, n - 1) as usize),
        BorderMode::Wrap => {
            let mut c = coord % n;
            if c < 0 {
                c += n;
            }
            Some(c as usize)
        }
        BorderMode::Reflect => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c - 1;
            }
            Some(c as usize)
        }
        BorderMode::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c;
            }
            Some(c as usize)
        }
    }
}

pub fn convolve(image: &GrayImage, kernel: &Kernel) -> GrayImage {
    convolve_with_border(image, kernel, BorderMode::Replicate)
}

pub fn convolve_with_border(image: &GrayImage, kernel: &Kernel, border: BorderMode) -> GrayImage {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    convolve_ctx(image, kernel, border, &runner)
}

pub fn convolve_ctx(
    image: &GrayImage,
    kernel: &Kernel,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    let mut output = GrayImage::new(image.width(), image.height());
    convolve_into_ctx(image, &mut output, kernel, border, group);
    output
}

pub fn convolve_into_ctx(
    image: &GrayImage,
    output: &mut GrayImage,
    kernel: &Kernel,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(result) = convolve_gpu(gpu, image, kernel, border) {
            output.copy_from_slice(result.as_raw());
            return;
        }
    }

    convolve_with_border_into_ctx(image, output, kernel, border, group);
}

pub fn convolve_with_border_into(
    image: &GrayImage,
    output: &mut GrayImage,
    kernel: &Kernel,
    border: BorderMode,
) {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    convolve_into_ctx(image, output, kernel, border, &runner);
}

pub fn gaussian_blur_with_border(image: &GrayImage, sigma: f32, border: BorderMode) -> GrayImage {
    match cv_runtime::best_runner() {
        Ok(runner) => gaussian_blur_ctx(image, sigma, border, &runner),
        Err(_) => {
            // Fallback: compute on CPU
            gaussian_blur_ctx(
                image,
                sigma,
                border,
                &cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)),
            )
        }
    }
}

pub fn gaussian_blur_ctx(
    image: &GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(result) = gaussian_blur_gpu(gpu, image, sigma, border) {
            return result;
        }
    }
    gaussian_blur_with_border_into_ctx(image, sigma, border, group)
}

fn gaussian_blur_gpu(
    gpu: &GpuContext,
    image: &GrayImage,
    sigma: f32,
    border: BorderMode,
) -> cv_hal::Result<GrayImage> {
    use cv_core::storage::Storage;
    use cv_hal::context::ComputeContext;

    let size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel = gaussian_kernel(sigma, size);

    let src_f32: Vec<f32> = image.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let kernel_tensor = cv_core::CpuTensor::from_vec(
        kernel.data,
        TensorShape::new(1, kernel.height, kernel.width),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let kernel_gpu = kernel_tensor.to_gpu_ctx(gpu)?;

    let hal_border = match border {
        BorderMode::Constant(v) => HalBorderMode::Constant(v as f32),
        BorderMode::Replicate => HalBorderMode::Replicate,
        BorderMode::Reflect => HalBorderMode::Reflect,
        BorderMode::Wrap => HalBorderMode::Wrap,
        _ => HalBorderMode::Replicate,
    };

    let output_gpu = gpu.convolve_2d(&input_gpu, &kernel_gpu, hal_border)?;
    let output_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = output_gpu.to_cpu()?;

    let data: Vec<u8> = output_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(image.width(), image.height(), data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

fn convolve_gpu(
    gpu: &GpuContext,
    image: &GrayImage,
    kernel: &Kernel,
    border: BorderMode,
) -> cv_hal::Result<GrayImage> {
    use cv_core::storage::Storage;
    use cv_hal::context::ComputeContext;

    let src_f32: Vec<f32> = image.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let kernel_tensor = cv_core::CpuTensor::from_vec(
        kernel.data.clone(),
        TensorShape::new(1, kernel.height, kernel.width),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let kernel_gpu = kernel_tensor.to_gpu_ctx(gpu)?;

    let hal_border = match border {
        BorderMode::Constant(v) => HalBorderMode::Constant(v as f32),
        BorderMode::Replicate => HalBorderMode::Replicate,
        BorderMode::Reflect => HalBorderMode::Reflect,
        BorderMode::Reflect101 => HalBorderMode::Reflect,
        BorderMode::Wrap => HalBorderMode::Wrap,
    };

    let output_gpu = gpu.convolve_2d(&input_gpu, &kernel_gpu, hal_border)?;
    let output_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = output_gpu.to_cpu()?;

    let data: Vec<u8> = output_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(image.width(), image.height(), data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

pub fn convolve_with_border_into_ctx(
    image: &GrayImage,
    output: &mut GrayImage,
    kernel: &Kernel,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    let (kx_center, ky_center) = kernel.center();
    if output.width() != image.width() || output.height() != image.height() {
        *output = GrayImage::new(image.width(), image.height());
    }
    let width = image.width() as usize;
    let height = image.height() as usize;
    let input_data = image.as_raw();

    group.run(|| {
        output
            .as_mut()
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let mut sum = 0.0f32;

                    for ky in 0..kernel.height {
                        for kx in 0..kernel.width {
                            let src_x = x as isize + kx as isize - kx_center;
                            let src_y = y as isize + ky as isize - ky_center;

                            let pixel_val = match (
                                map_coord(src_x, width, border),
                                map_coord(src_y, height, border),
                            ) {
                                (Some(ix), Some(iy)) => input_data[iy * width + ix] as f32,
                                _ => match border {
                                    BorderMode::Constant(v) => v as f32,
                                    _ => 0.0,
                                },
                            };

                            let kernel_val = kernel.get(kx, ky);
                            sum += pixel_val * kernel_val;
                        }
                    }

                    row[x] = sum.clamp(0.0, 255.0) as u8;
                }
            });
    });
}

pub fn separable_convolve(image: &GrayImage, kernel_1d: &[f32], border: BorderMode) -> GrayImage {
    let mut out = GrayImage::new(image.width(), image.height());
    separable_convolve_into(image, &mut out, kernel_1d, border);
    out
}

pub fn separable_convolve_into(
    image: &GrayImage,
    out: &mut GrayImage,
    kernel_1d: &[f32],
    border: BorderMode,
) {
    match cv_runtime::default_runner() {
        Ok(runner) => {
            separable_convolve_into_ctx(image, out, kernel_1d, kernel_1d, border, &runner)
        }
        Err(_) => {
            // Fallback: compute on CPU
            separable_convolve_into_ctx(
                image,
                out,
                kernel_1d,
                kernel_1d,
                border,
                &cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)),
            );
        }
    }
}

pub fn separable_convolve_into_ctx(
    image: &GrayImage,
    out: &mut GrayImage,
    kx: &[f32],
    ky: &[f32],
    border: BorderMode,
    group: &RuntimeRunner,
) {
    assert!(kx.len() % 2 == 1, "kx size must be odd");
    assert!(ky.len() % 2 == 1, "ky size must be odd");

    if out.width() != image.width() || out.height() != image.height() {
        *out = GrayImage::new(image.width(), image.height());
    }

    let width = image.width() as usize;
    let height = image.height() as usize;
    let rx = kx.len() / 2;
    let ry = ky.len() / 2;
    let src = image.as_raw();

    let mut tmp: Vec<f32> = vec![0.0f32; width * height];

    // Horizontal Pass (using kx)
    group.run(|| {
        tmp.par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_offset = y * width;
                let padded_width = width + 2 * rx;
                let mut padded_row = vec![0.0f32; padded_width];
                for i in 0..padded_width {
                    let src_x = (i as isize) - (rx as isize);
                    padded_row[i] = match map_coord(src_x, width, border) {
                        Some(ix) => src[row_offset + ix] as f32,
                        None => match border {
                            BorderMode::Constant(v) => v as f32,
                            _ => 0.0,
                        },
                    };
                }
                convolve_row_1d(&padded_row, row_out, kx, rx);
            });
    });

    // Vertical Pass (using ky)
    group.run(|| {
        out.as_mut()
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row_out)| {
                for x in (0..width).step_by(8) {
                    if x + 8 <= width {
                        let mut sum_v = f32x8::ZERO;
                        for k in 0..ky.len() {
                            let w_v = f32x8::splat(ky[k]);
                            let sy_base = (y as isize) + (k as isize) - (ry as isize);
                            let target_y = map_coord(sy_base, height, border);
                            let mut vals = [0.0f32; 8];
                            if let Some(iy) = target_y {
                                let idx = iy * width + x;
                                vals.copy_from_slice(&tmp[idx..idx + 8]);
                            } else if let BorderMode::Constant(v) = border {
                                vals = [v as f32; 8];
                            }
                            sum_v += f32x8::from(vals) * w_v;
                        }
                        let res: [f32; 8] = sum_v.into();
                        for i in 0..8 {
                            row_out[x + i] = res[i].clamp(0.0, 255.0) as u8;
                        }
                    } else {
                        for cx in x..width {
                            let mut sum = 0.0;
                            for k in 0..ky.len() {
                                let sy = (y as isize) + (k as isize) - (ry as isize);
                                let val = match map_coord(sy, height, border) {
                                    Some(iy) => tmp[iy * width + cx],
                                    None => match border {
                                        BorderMode::Constant(v) => v as f32,
                                        _ => 0.0,
                                    },
                                };
                                sum += val * ky[k];
                            }
                            row_out[cx] = sum.clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            });
    });
}

pub fn gaussian_blur_with_border_into_ctx(
    image: &GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    let mut out = GrayImage::new(image.width(), image.height());
    gaussian_blur_with_border_into_ctx_into(image, &mut out, sigma, border, group);
    out
}

pub fn gaussian_blur_with_border_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
) {
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            let runner = RuntimeRunner::Group(group);
            gaussian_blur_ctx_into(image, out, sigma, border, &runner);
        }
    }
}

pub fn gaussian_blur_ctx_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    let size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel_1d = gaussian_kernel_1d(sigma, size);
    separable_convolve_into_ctx(image, out, &kernel_1d, &kernel_1d, border, group);
}

fn gaussian_blur_with_border_into_ctx_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    let size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel_1d = gaussian_kernel_1d(sigma, size);
    separable_convolve_into_ctx(image, out, &kernel_1d, &kernel_1d, border, group);
}

pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    gaussian_blur_with_border(image, sigma, BorderMode::Replicate)
}

pub fn box_blur(image: &GrayImage, size: usize) -> GrayImage {
    let kernel = box_kernel(size);
    convolve(image, &kernel)
}

pub fn sharpen(image: &GrayImage, amount: f32) -> GrayImage {
    let mut kernel = laplacian_kernel();
    for v in &mut kernel.data {
        *v *= -amount;
    }
    let center = kernel.width / 2;
    kernel.data[center * kernel.width + center] += 1.0 + amount;
    convolve(image, &kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn gaussian_kernel_1d_is_normalized() {
        let k = gaussian_kernel_1d(1.2, 7);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gaussian_blur_preserves_size() {
        let mut img = GrayImage::new(32, 24);
        img.put_pixel(10, 10, Luma([255]));

        let out = gaussian_blur_with_border(&img, 1.0, BorderMode::Reflect101);
        assert_eq!(out.width(), img.width());
        assert_eq!(out.height(), img.height());
    }
}
