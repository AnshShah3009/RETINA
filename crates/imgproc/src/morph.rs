use crate::convolve::BorderMode;
use cv_core::{storage::Storage, TensorShape};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::MorphologyType as HalMorphType;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::{scheduler, ResourceGroup};
use image::GrayImage;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShape {
    Rectangle,
    Ellipse,
    Cross,
}

pub fn create_morph_kernel(shape: MorphShape, width: u32, height: u32) -> Vec<(i32, i32)> {
    let mut kernel = Vec::new();
    let cx = width as i32 / 2;
    let cy = height as i32 / 2;

    match shape {
        MorphShape::Rectangle => {
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    kernel.push((x - cx, y - cy));
                }
            }
        }
        MorphShape::Ellipse => {
            let rx = width as f32 / 2.0;
            let ry = height as f32 / 2.0;

            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let dx = (x - cx) as f32;
                    let dy = (y - cy) as f32;
                    if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0 {
                        kernel.push((x - cx, y - cy));
                    }
                }
            }
        }
        MorphShape::Cross => {
            for i in -(width as i32 / 2)..=(width as i32 / 2) {
                kernel.push((i, 0));
            }
            for i in -(height as i32 / 2)..=(height as i32 / 2) {
                kernel.push((0, i));
            }
        }
    }

    kernel
}

fn map_coord(coord: i32, len: i32, border: BorderMode) -> Option<i32> {
    if len <= 0 {
        return None;
    }

    match border {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= len {
                None
            } else {
                Some(coord)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, len - 1)),
        BorderMode::Wrap => {
            let mut c = coord % len;
            if c < 0 {
                c += len;
            }
            Some(c)
        }
        BorderMode::Reflect => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= len {
                c = period - c - 1;
            }
            Some(c)
        }
        BorderMode::Reflect101 => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= len {
                c = period - c;
            }
            Some(c)
        }
    }
}

fn dilate_once_into(
    src: &GrayImage,
    output: &mut GrayImage,
    kernel: &[(i32, i32)],
    border: BorderMode,
) {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let src_data = src.as_raw();
    if output.width() != src.width() || output.height() != src.height() {
        *output = GrayImage::new(src.width(), src.height());
    }

    output
        .as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width {
                let mut max_val = 0u8;
                for &(kx, ky) in kernel {
                    let px = x + kx;
                    let py = y + ky;

                    let val = match (map_coord(px, width, border), map_coord(py, height, border)) {
                        (Some(ix), Some(iy)) => src_data[(iy * width + ix) as usize],
                        _ => match border {
                            BorderMode::Constant(v) => v,
                            _ => 0,
                        },
                    };
                    max_val = max_val.max(val);
                }
                row[x as usize] = max_val;
            }
        });
}

fn erode_once_into(
    src: &GrayImage,
    output: &mut GrayImage,
    kernel: &[(i32, i32)],
    border: BorderMode,
) {
    let width = src.width() as i32;
    let height = src.height() as i32;
    let src_data = src.as_raw();
    if output.width() != src.width() || output.height() != src.height() {
        *output = GrayImage::new(src.width(), src.height());
    }

    output
        .as_mut()
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width {
                let mut min_val = 255u8;
                for &(kx, ky) in kernel {
                    let px = x + kx;
                    let py = y + ky;

                    let val = match (map_coord(px, width, border), map_coord(py, height, border)) {
                        (Some(ix), Some(iy)) => src_data[(iy * width + ix) as usize],
                        _ => match border {
                            BorderMode::Constant(v) => v,
                            _ => 255,
                        },
                    };
                    min_val = min_val.min(val);
                }
                row[x as usize] = min_val;
            }
        });
}

pub fn dilate_with_border(
    src: &GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) -> GrayImage {
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            return morphology_ctx(
                src,
                HalMorphType::Dilate,
                kernel,
                iterations,
                border,
                &group,
            );
        }
    }
    // Fallback to basic parallel CPU if scheduler fails
    let mut out = GrayImage::new(src.width(), src.height());
    dilate_with_border_into(src, &mut out, kernel, iterations, border);
    out
}

pub fn erode_with_border(
    src: &GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) -> GrayImage {
    if let Ok(s) = scheduler() {
        if let Ok(group) = s.get_default_group() {
            return morphology_ctx(src, HalMorphType::Erode, kernel, iterations, border, &group);
        }
    }
    // Fallback
    let mut out = GrayImage::new(src.width(), src.height());
    erode_with_border_into(src, &mut out, kernel, iterations, border);
    out
}

pub fn morphology_ctx(
    src: &GrayImage,
    typ: HalMorphType,
    kernel: &[(i32, i32)],
    iterations: u32,
    _border: BorderMode,
    group: &ResourceGroup,
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        if let Ok(result) = morphology_gpu(gpu, src, typ, kernel, iterations) {
            return result;
        }
    }

    // CPU fallback
    let mut out = GrayImage::new(src.width(), src.height());
    match typ {
        HalMorphType::Dilate => dilate_with_border_into(src, &mut out, kernel, iterations, _border),
        HalMorphType::Erode => erode_with_border_into(src, &mut out, kernel, iterations, _border),
        _ => {}
    }
    out
}

fn morphology_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    typ: HalMorphType,
    kernel: &[(i32, i32)],
    iterations: u32,
) -> cv_hal::Result<GrayImage> {
    use cv_hal::context::ComputeContext;

    let input_tensor = cv_core::CpuTensor::from_vec(
        src.as_raw().to_vec(),
        TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    // Convert (i32, i32) kernel to Tensor<u8> mask
    let (min_kx, max_kx, min_ky, max_ky) = kernel
        .iter()
        .fold((0, 0, 0, 0), |(mix, max, miy, may), &(x, y)| {
            (mix.min(x), max.max(x), miy.min(y), may.max(y))
        });
    let kw = (max_kx - min_kx + 1).max(1) as usize;
    let kh = (max_ky - min_ky + 1).max(1) as usize;
    let mut k_data = vec![0u8; kw * kh];
    let cx = -min_kx as usize;
    let cy = -min_ky as usize;
    for &(x, y) in kernel {
        let kx = (x as isize + cx as isize) as usize;
        let ky = (y as isize + cy as isize) as usize;
        k_data[ky * kw + kx] = 1;
    }
    let kernel_tensor = cv_core::CpuTensor::from_vec(k_data, TensorShape::new(1, kh, kw))
        .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let kernel_gpu = kernel_tensor.to_gpu_ctx(gpu)?;

    let output_gpu = gpu.morphology(&input_gpu, typ, &kernel_gpu, iterations)?;
    let output_cpu = output_gpu.to_cpu_ctx(gpu)?;

    let data = output_cpu.storage.as_slice().unwrap().to_vec();
    GrayImage::from_raw(src.width(), src.height(), data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

pub fn dilate_with_border_into(
    src: &GrayImage,
    dst: &mut GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) {
    if iterations == 0 {
        if dst.width() != src.width() || dst.height() != src.height() {
            *dst = src.clone();
        } else {
            dst.as_mut().copy_from_slice(src.as_raw());
        }
        return;
    }
    let mut cur = src.clone();
    let mut tmp = GrayImage::new(src.width(), src.height());
    for _ in 0..iterations {
        dilate_once_into(&cur, &mut tmp, kernel, border);
        std::mem::swap(&mut cur, &mut tmp);
    }
    if dst.width() != cur.width() || dst.height() != cur.height() {
        *dst = cur;
    } else {
        dst.as_mut().copy_from_slice(cur.as_raw());
    }
}

pub fn erode_with_border_into(
    src: &GrayImage,
    dst: &mut GrayImage,
    kernel: &[(i32, i32)],
    iterations: u32,
    border: BorderMode,
) {
    if iterations == 0 {
        if dst.width() != src.width() || dst.height() != src.height() {
            *dst = src.clone();
        } else {
            dst.as_mut().copy_from_slice(src.as_raw());
        }
        return;
    }
    let mut cur = src.clone();
    let mut tmp = GrayImage::new(src.width(), src.height());
    for _ in 0..iterations {
        erode_once_into(&cur, &mut tmp, kernel, border);
        std::mem::swap(&mut cur, &mut tmp);
    }
    if dst.width() != cur.width() || dst.height() != cur.height() {
        *dst = cur;
    } else {
        dst.as_mut().copy_from_slice(cur.as_raw());
    }
}

pub fn dilate(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    dilate_with_border(src, kernel, iterations, BorderMode::Replicate)
}

pub fn erode(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    erode_with_border(src, kernel, iterations, BorderMode::Replicate)
}

pub fn opening(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let eroded = erode(src, kernel, iterations);
    dilate(&eroded, kernel, iterations)
}

pub fn closing(src: &GrayImage, kernel: &[(i32, i32)], iterations: u32) -> GrayImage {
    let dilated = dilate(src, kernel, iterations);
    erode(&dilated, kernel, iterations)
}

pub fn morphological_gradient(src: &GrayImage, kernel_size: u32) -> GrayImage {
    let kernel = create_morph_kernel(MorphShape::Ellipse, kernel_size, kernel_size);
    let dilated = dilate(src, &kernel, 1);
    let eroded = erode(src, &kernel, 1);

    let mut output = GrayImage::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            let d = dilated.get_pixel(x, y)[0];
            let e = eroded.get_pixel(x, y)[0];
            output.put_pixel(x, y, image::Luma([d.saturating_sub(e)]));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn iterations_zero_is_noop() {
        let mut img = GrayImage::new(8, 8);
        img.put_pixel(4, 4, Luma([255]));
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = dilate(&img, &k, 0);
        assert_eq!(out.as_raw(), img.as_raw());
    }

    #[test]
    fn create_rectangle_kernel() {
        let kernel = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        assert_eq!(kernel.len(), 9); // 3x3 = 9
    }

    #[test]
    fn create_ellipse_kernel() {
        let kernel = create_morph_kernel(MorphShape::Ellipse, 3, 3);
        // Ellipse with 3x3 should have multiple points
        assert!(kernel.len() > 0);
        assert!(kernel.len() <= 9);
    }

    #[test]
    fn create_cross_kernel() {
        let kernel = create_morph_kernel(MorphShape::Cross, 3, 3);
        // Cross pattern: horizontal + vertical lines
        assert!(kernel.len() > 0);
        // Should have some cross-shaped points
        assert!(kernel.len() <= 9);
    }

    #[test]
    fn dilate_white_point() {
        let mut img = GrayImage::new(8, 8);
        img.put_pixel(4, 4, Luma([255]));
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = dilate(&img, &k, 1);

        // Dilate should expand the white region
        assert!(out.get_pixel(3, 4)[0] > 0); // Should expand left
        assert!(out.get_pixel(5, 4)[0] > 0); // Should expand right
        assert!(out.get_pixel(4, 3)[0] > 0); // Should expand up
        assert!(out.get_pixel(4, 5)[0] > 0); // Should expand down
    }

    #[test]
    fn erode_white_region() {
        let mut img = GrayImage::new(16, 16);
        // Create a white square
        for y in 5..11 {
            for x in 5..11 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = erode(&img, &k, 1);

        // Erode should shrink the white region
        assert!(out.get_pixel(5, 8)[0] == 0); // Boundary should be eroded
        assert!(out.get_pixel(8, 8)[0] > 0); // Center should remain
    }

    #[test]
    fn opening_removes_noise() {
        let mut img = GrayImage::new(12, 12);
        // Create white square with noise
        for y in 3..9 {
            for x in 3..9 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        // Add noise (isolated white pixels)
        img.put_pixel(1, 1, Luma([255]));
        img.put_pixel(10, 10, Luma([255]));

        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = opening(&img, &k, 1);

        // Opening should remove isolated pixels
        assert!(out.get_pixel(1, 1)[0] == 0);
        assert!(out.get_pixel(10, 10)[0] == 0);
    }

    #[test]
    fn closing_fills_holes() {
        let mut img = GrayImage::new(12, 12);
        // Create white region with a hole
        for y in 3..9 {
            for x in 3..9 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        img.put_pixel(6, 6, Luma([0])); // Create a hole

        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = closing(&img, &k, 1);

        // Closing should fill the hole
        assert!(out.get_pixel(6, 6)[0] > 0);
    }

    #[test]
    fn multiple_iterations() {
        let mut img = GrayImage::new(16, 16);
        img.put_pixel(8, 8, Luma([255]));
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);

        let iter1 = dilate(&img, &k, 1);
        let iter2 = dilate(&img, &k, 2);

        // More iterations should expand more
        let count1 = iter1.as_raw().iter().filter(|&&v| v > 0).count();
        let count2 = iter2.as_raw().iter().filter(|&&v| v > 0).count();
        assert!(count2 > count1);
    }

    #[test]
    fn morphological_gradient_basic() {
        let mut img = GrayImage::new(16, 16);
        // Create a white square
        for y in 5..11 {
            for x in 5..11 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        let grad = morphological_gradient(&img, 3);

        assert_eq!(grad.width(), 16);
        assert_eq!(grad.height(), 16);
        // Gradient should be non-zero at boundaries
        assert!(grad.as_raw().iter().any(|&v| v > 0));
    }

    #[test]
    fn dilate_output_dimensions() {
        let img = GrayImage::new(32, 24);
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = dilate(&img, &k, 1);

        assert_eq!(out.width(), 32);
        assert_eq!(out.height(), 24);
    }

    #[test]
    fn erode_output_dimensions() {
        let img = GrayImage::new(32, 24);
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = erode(&img, &k, 1);

        assert_eq!(out.width(), 32);
        assert_eq!(out.height(), 24);
    }

    #[test]
    fn opening_then_closing() {
        let mut img = GrayImage::new(16, 16);
        // Create pattern with noise and holes
        for y in 4..12 {
            for x in 4..12 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        // Add noise
        img.put_pixel(2, 2, Luma([255]));
        // Add hole
        img.put_pixel(8, 8, Luma([0]));

        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = opening(&img, &k, 1);
        let final_out = closing(&out, &k, 1);

        assert_eq!(final_out.width(), 16);
        assert_eq!(final_out.height(), 16);
        // Should remove noise and fill hole
        assert!(final_out.get_pixel(2, 2)[0] == 0); // Noise removed
        assert!(final_out.get_pixel(8, 8)[0] > 0); // Hole filled
    }

    #[test]
    fn different_kernel_shapes() {
        let mut img = GrayImage::new(16, 16);
        img.put_pixel(8, 8, Luma([255]));

        let rect_k = create_morph_kernel(MorphShape::Rectangle, 5, 5);
        let ellipse_k = create_morph_kernel(MorphShape::Ellipse, 5, 5);
        let cross_k = create_morph_kernel(MorphShape::Cross, 5, 5);

        let out_rect = dilate(&img, &rect_k, 1);
        let out_ellipse = dilate(&img, &ellipse_k, 1);
        let out_cross = dilate(&img, &cross_k, 1);

        // Different kernels should produce different results
        let rect_count = out_rect.as_raw().iter().filter(|&&v| v > 0).count();
        let ellipse_count = out_ellipse.as_raw().iter().filter(|&&v| v > 0).count();
        let cross_count = out_cross.as_raw().iter().filter(|&&v| v > 0).count();

        assert_ne!(rect_count, ellipse_count);
        assert_ne!(rect_count, cross_count);
    }

    #[test]
    fn dilate_preserves_range() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let val = ((x + y) % 256) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
        let out = dilate(&img, &k, 1);

        // Output values are always valid byte range (u8 is 0..=255 by definition)
        assert!(!out.as_raw().is_empty());
    }
}
