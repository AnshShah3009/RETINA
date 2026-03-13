use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::ResourceGroup;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

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
    resize_linear(src, width, height)
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

    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if interpolation == Interpolation::Linear {
            if let Ok(result) = resize_gpu(gpu, src, width, height) {
                return result;
            }
        }
    }

    group.run(|| resize_linear(src, width, height))
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

#[allow(dead_code)]
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
    let dst_width = (width - 1) as f32;
    let dst_height = (height - 1) as f32;

    if src_width <= 0.0 || src_height <= 0.0 {
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
    _interpolation: Interpolation,
) -> RgbImage {
    if width == 0 || height == 0 {
        return RgbImage::new(0, 0);
    }

    let mut dst = RgbImage::new(width, height);
    let src_width = src.width() as f32 - 1.0;
    let src_height = src.height() as f32 - 1.0;
    let dst_width = (width - 1) as f32;
    let dst_height = (height - 1) as f32;

    if src_width <= 0.0 || src_height <= 0.0 {
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
