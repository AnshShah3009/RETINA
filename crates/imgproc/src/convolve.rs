use crate::simd::convolve_row_1d;
use cv_core::{Tensor, TensorShape};
use cv_hal::compute::ComputeDevice;
use cv_hal::context::BorderMode as HalBorderMode;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::{RuntimeRunner, scheduler};
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

/// OpenCV's fixed kernels for `getGaussianKernel(size, sigma <= 0)` with
/// `size <= 9` (`small_gaussian_tab` in `smooth.dispatch.cpp`).
fn small_gaussian_table(size: usize) -> Option<&'static [f32]> {
    match size {
        1 => Some(&[1.0]),
        3 => Some(&[0.25, 0.5, 0.25]),
        5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
        7 => Some(&[
            0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
        ]),
        9 => Some(&[
            0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875,
            0.05078125, 0.015625,
        ]),
        _ => None,
    }
}

/// OpenCV's sigma for a kernel `size` when the given sigma is not positive:
/// `0.3 * ((size - 1) * 0.5 - 1) + 0.8`.
fn sigma_from_size(size: usize) -> f32 {
    (0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8) as f32
}

/// 2-D Gaussian kernel of `size x size` taps, normalised to sum to 1.
///
/// A `sigma` that is not positive (zero, negative or NaN) is derived from
/// `size` as `cv::getGaussianKernel` does: OpenCV's fixed tables for sizes
/// 1-9 (the kernel is then their outer product), otherwise
/// `0.3 * ((size - 1) * 0.5 - 1) + 0.8`. It used to divide by `sigma² = 0`
/// and return NaN taps.
pub fn gaussian_kernel(sigma: f32, size: usize) -> Kernel {
    if !(sigma > 0.0) {
        if let Some(t) = small_gaussian_table(size) {
            let data = t
                .iter()
                .flat_map(|&a| t.iter().map(move |&b| a * b))
                .collect();
            return Kernel::new(data, size, size);
        }
        return gaussian_kernel(sigma_from_size(size), size);
    }
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

    // Normalise, then force the f32 sum to be exactly 1.
    //
    // Dividing by `sum` alone is not enough: the f32 *taps* are accurate to ~6e-9
    // (measured, summed in f64), but summing 169 of them in f32 accumulates
    // rounding and lands off 1. Measured for a 13-tap sigma-2.0 kernel:
    //
    // f64 sum of the f32 taps : 0.999999994139216142   (5.9e-9 - the taps)
    // f32 sum of the f32 taps : 1.000000357627868650   (3.6e-7 - the summation)
    //
    // The consequence is not academic. A **normalised blur of a constant image must
    // return that constant**, and at sigma 2.0 this returned `127` where OpenCV
    // returns `128` - for **100% of pixels**, because `128 * 0.99999964` truncates to
    // `127`. Any detector thresholding a blurred image sees a uniform brightness
    // shift. Verified against `cv2.filter2D` on a constant 128 image:
    //
    // cv2  reflect101 / replicate : 128   (kernel sums to exactly 1)
    // rust                       : 127   (kernel sums to 0.99999964)
    //
    // Correcting the **last** tap costs one subtraction and lands on the smallest
    // tap in the kernel, so the filter shape is perturbed negligibly: at sigma 2.0
    // / size 13 that is the 4.5e-6 corner, against a peak near 1. Accumulating the
    // normalisation in f64 first is *also* not sufficient - measured, the error is
    // the same 3.6e-7, because it is the summation of the narrowed taps, not the
    // accumulation of the weights, that carries it.
    //
    // After this the sum is exactly 1.0 for every sigma and size measured
    // (0.75/1.5/2.0/3.0 at size 5, and size 13).
    let inv = 1.0f32 / sum;
    for v in &mut data {
        *v *= inv;
    }
    let last = data.len() - 1;
    let head: f32 = data[..last].iter().sum();
    data[last] = 1.0 - head;

    Kernel::new(data, size, size)
}

/// 1-D Gaussian kernel of `size` (odd) taps, normalised to sum to 1.
///
/// A `sigma` that is not positive (zero, negative or NaN) is derived from
/// `size` exactly as `cv::getGaussianKernel(size, sigma)` does: OpenCV's fixed
/// tables for sizes 1, 3, 5, 7 and 9, otherwise
/// `sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8`. This is what
/// `cv2.GaussianBlur(img, (size, size), 0)` blurs with. It used to divide by
/// `sigma² = 0` and return NaN taps.
pub fn gaussian_kernel_1d(sigma: f32, size: usize) -> Vec<f32> {
    assert!(size % 2 == 1, "gaussian kernel size must be odd");
    if !(sigma > 0.0) {
        if let Some(t) = small_gaussian_table(size) {
            return t.to_vec();
        }
        return gaussian_kernel_1d(sigma_from_size(size), size);
    }
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
        // Same correction as the 2-D case above, and for the same reason: dividing
        // by `sum` leaves the f32 taps summing to 1 +/- a few ulp, which a constant
        // image does not survive after truncation. Measured on a 13-tap sigma-2.0
        // 1-D kernel: `1 - sum = 5.960e-8` after division, `0` after this.
        let inv = 1.0f32 / sum;
        for v in &mut kernel {
            *v *= inv;
        }
        let last = kernel.len() - 1;
        let head: f32 = kernel[..last].iter().sum();
        kernel[last] = 1.0 - head;
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

/// 2-D correlation with `kernel` (OpenCV's `filter2D`), padding with
/// OpenCV's default border, [`BorderMode::Reflect101`].
///
/// This used to pad with `Replicate`, so every pixel within the kernel radius
/// of an edge differed from `cv2.filter2D`. Use [`convolve_with_border`] for
/// any other border.
pub fn convolve(image: &GrayImage, kernel: &Kernel) -> GrayImage {
    convolve_with_border(image, kernel, BorderMode::Reflect101)
}

pub fn convolve_with_border(image: &GrayImage, kernel: &Kernel, border: BorderMode) -> GrayImage {
    let runner = cv_runtime::best_runner()
        .map(|(r, _)| r)
        .unwrap_or_else(|_| {
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
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device()
        && let Ok(result) = convolve_gpu(gpu, image, kernel, border)
    {
        output.copy_from_slice(result.as_raw());
        return;
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
    match cv_runtime::best_runner().map(|(r, _)| r) {
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
    if !(sigma > 0.0) {
        // See `gaussian_blur`: no kernel size to derive one from.
        return image.clone();
    }
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device()
        && let Ok(result) = gaussian_blur_gpu(gpu, image, sigma, border)
    {
        return result;
    }
    gaussian_blur_with_border_into_ctx(image, sigma, border, group)
}

fn gaussian_blur_gpu(
    gpu: &GpuContext,
    image: &GrayImage,
    sigma: f32,
    _border: BorderMode,
) -> cv_hal::Result<GrayImage> {
    use cv_core::storage::Storage;
    use cv_hal::context::ComputeContext;

    let size = ((sigma * 6.0).ceil() as usize) | 1;

    let src_f32: Vec<f32> = image.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, image.height() as usize, image.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    // Use the HAL's separable gaussian_blur shader for ~3.5x speedup over convolve_2d
    let output_gpu = gpu.gaussian_blur(&input_gpu, sigma, size)?;
    let output_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = output_gpu.to_cpu()?;

    let data: Vec<u8> = output_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.round_ties_even().clamp(0.0, 255.0) as u8)
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
        BorderMode::Reflect101 => HalBorderMode::Reflect101,
        BorderMode::Wrap => HalBorderMode::Wrap,
    };

    let output_gpu = gpu.convolve_2d(&input_gpu, &kernel_gpu, hal_border)?;
    let output_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = output_gpu.to_cpu()?;

    let data: Vec<u8> = output_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(image.width(), image.height(), data)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from tensor".into()))
}

#[allow(clippy::needless_range_loop)]
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
    if width == 0 || height == 0 {
        return;
    }
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

                    row[x] = sum.round_ties_even().clamp(0.0, 255.0) as u8;
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

#[allow(clippy::needless_range_loop)]
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
    if width == 0 || height == 0 {
        return;
    }
    let rx = kx.len() / 2;
    let ry = ky.len() / 2;
    let src = image.as_raw();

    // Value the *whole* row takes when it lies outside the image under a
    // `Constant` border. The vertical pass convolves rows of the horizontal
    // result, so an out-of-range row must be the horizontal convolution of a
    // row filled with `v`, i.e. `v * Σkx` - not `v`. Substituting the raw
    // constant made the separable path disagree with the 2-D
    // `convolve_with_border` by a factor of `Σkx`: zero for a derivative
    // kernel (`[-1, 0, 1]`), larger for an unnormalised smoothing kernel
    // (`[1, 2, 1]` -> `4v`). It agreed only for kernels that sum to exactly 1.
    let constant_row_value = match border {
        BorderMode::Constant(v) => v as f32 * kx.iter().sum::<f32>(),
        _ => 0.0,
    };

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
                            } else {
                                vals = [constant_row_value; 8];
                            }
                            sum_v += f32x8::from(vals) * w_v;
                        }
                        let res: [f32; 8] = sum_v.into();
                        for i in 0..8 {
                            row_out[x + i] = res[i].round_ties_even().clamp(0.0, 255.0) as u8;
                        }
                    } else {
                        for cx in x..width {
                            let mut sum = 0.0;
                            for k in 0..ky.len() {
                                let sy = (y as isize) + (k as isize) - (ry as isize);
                                let val = match map_coord(sy, height, border) {
                                    Some(iy) => tmp[iy * width + cx],
                                    None => constant_row_value,
                                };
                                sum += val * ky[k];
                            }
                            row_out[cx] = sum.round_ties_even().clamp(0.0, 255.0) as u8;
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
    if let Ok(s) = scheduler()
        && let Ok(group) = s.get_default_group()
    {
        let runner = RuntimeRunner::Group(group);
        gaussian_blur_ctx_into(image, out, sigma, border, &runner);
    }
}

pub fn gaussian_blur_ctx_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    gaussian_blur_with_border_into_ctx_into(image, out, sigma, border, group);
}

fn gaussian_blur_with_border_into_ctx_into(
    image: &GrayImage,
    out: &mut GrayImage,
    sigma: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) {
    if !(sigma > 0.0) {
        // See `gaussian_blur`: no kernel size to derive one from.
        if out.dimensions() != image.dimensions() {
            *out = image.clone();
        } else {
            out.copy_from_slice(image.as_raw());
        }
        return;
    }
    let size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel_1d = gaussian_kernel_1d(sigma, size);
    separable_convolve_into_ctx(image, out, &kernel_1d, &kernel_1d, border, group);
}

/// Gaussian blur with OpenCV's default border, [`BorderMode::Reflect101`]
/// (it used to be `Replicate`). Use [`gaussian_blur_with_border`] for any
/// other border.
///
/// The kernel has `ceil(6 sigma) | 1` taps. A `sigma` that is not positive
/// (zero, negative or NaN) returns the image unchanged: with no kernel size
/// to derive a sigma from, `cv2.GaussianBlur(img, (0, 0), 0)` raises, and the
/// unchanged image is what OpenCV returns for the 1x1 kernel. For OpenCV's
/// sigma-from-size rule, build the kernel with
/// `gaussian_kernel_1d(0.0, size)`.
pub fn gaussian_blur(image: &GrayImage, sigma: f32) -> GrayImage {
    gaussian_blur_with_border(image, sigma, BorderMode::Reflect101)
}

/// Normalised `size x size` box filter (`cv2.blur`) with OpenCV's default
/// border, [`BorderMode::Reflect101`] (it used to be `Replicate`). Use
/// [`box_blur_with_border`] for any other border.
pub fn box_blur(image: &GrayImage, size: usize) -> GrayImage {
    box_blur_with_border(image, size, BorderMode::Reflect101)
}

/// [`box_blur`] with an explicit border.
pub fn box_blur_with_border(image: &GrayImage, size: usize, border: BorderMode) -> GrayImage {
    let runner = cv_runtime::best_runner()
        .map(|(r, _)| r)
        .unwrap_or_else(|_| {
            cv_runtime::registry()
                .ok()
                .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
                .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
        });
    box_blur_with_border_ctx(image, size, border, &runner)
}

/// `box_blur` on an explicit runner, with OpenCV's default border
/// ([`BorderMode::Reflect101`]).
pub fn box_blur_ctx(image: &GrayImage, size: usize, group: &RuntimeRunner) -> GrayImage {
    box_blur_with_border_ctx(image, size, BorderMode::Reflect101, group)
}

/// `box_blur_with_border` on an explicit runner.
///
/// GPU runners take the generic `convolve_2d` shader. The CPU path returns
/// the same bytes as
/// `convolve_with_border_into_ctx(image, out, &box_kernel(size), border, _)`
/// (the dense O(size^2)-per-pixel loop) but runs in O(1) per pixel: see
/// [`box_blur_cpu_with_border_into`].
pub fn box_blur_with_border_ctx(
    image: &GrayImage,
    size: usize,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    if let Ok(ComputeDevice::Gpu(_)) = group.device() {
        return convolve_ctx(image, &box_kernel(size), border, group);
    }
    let mut out = GrayImage::new(image.width(), image.height());
    group.run(|| box_blur_cpu_with_border_into(image, &mut out, size, border));
    out
}

/// Largest box side the running-sum path handles. The rounding argument in
/// [`box_blur_cpu_into`] needs `size^2 * 2^-24` well below 1 (the classic
/// `gamma_n` bound only holds for `n * u < 1`); 1024 keeps it under 1/16, and
/// also keeps `size^2` exact in `f32` and `255 * size^2` inside `u32`.
const BOX_BLUR_MAX_FAST_SIZE: usize = 1024;

/// Sentinel in the window-sum lookup table: replay the dense loop.
const BOX_BLUR_REPLAY: u16 = u16::MAX;

/// Replicate-border box filter, byte-identical to the dense convolution with
/// [`box_kernel`] but O(1) per pixel. Kept with its original border; see
/// [`box_blur_cpu_with_border_into`] for the general form.
pub fn box_blur_cpu_into(image: &GrayImage, out: &mut GrayImage, size: usize) {
    box_blur_cpu_with_border_into(image, out, size, BorderMode::Replicate);
}

/// Box filter with any [`BorderMode`], byte-identical to the dense
/// convolution with [`box_kernel`] but O(1) per pixel.
///
/// The dense loop computes `round_ties_even(sum_f32(p_i * w))` with
/// `w = 1 / (size*size) as f32`, accumulating `n = size^2` products in kernel
/// order. That float sum is within `(n + 3) * 2^-24 * m` of the exact mean
/// `m = S / n`, `S` the integer window sum (each product and each of the `n`
/// additions rounds once, and `w` itself is `1/n` to half an ulp, all terms
/// non-negative). Whenever `m` lies farther than twice that bound from the
/// nearest `j + 0.5`, both the float sum and `m` round to the same integer, so
/// the exact integer `S` (from separable running sums) decides the byte. The
/// rest - exact ties for even `n`, and the rare near-ties of large kernels -
/// replay the dense loop for that pixel, so every output byte is the one the
/// dense loop produced.
pub fn box_blur_cpu_with_border_into(
    image: &GrayImage,
    out: &mut GrayImage,
    size: usize,
    border: BorderMode,
) {
    let width = image.width() as usize;
    let height = image.height() as usize;
    if out.width() != image.width() || out.height() != image.height() {
        *out = GrayImage::new(image.width(), image.height());
    }
    if width == 0 || height == 0 {
        return;
    }
    if size == 0 || size > BOX_BLUR_MAX_FAST_SIZE {
        // size 0: the dense loop sums nothing and writes 0 everywhere; huge
        // kernels: keep the dense loop rather than widen the accumulators.
        let runner = RuntimeRunner::Sync(cv_hal::DeviceId(0));
        convolve_with_border_into_ctx(image, out, &box_kernel(size), border, &runner);
        return;
    }

    let src = image.as_raw();
    let lo = -((size / 2) as isize);
    let hi = lo + size as isize - 1;
    // A `Constant` border contributes `v` per out-of-image tap; every other
    // mode maps the tap back into the image.
    let constant = match border {
        BorderMode::Constant(v) => v as u32,
        _ => 0,
    };
    let tap = |row: &[u8], c: isize| -> u32 {
        match map_coord(c, row.len(), border) {
            Some(i) => row[i] as u32,
            None => constant,
        }
    };

    // Horizontal window sums.
    let mut hsum = vec![0u32; width * height];
    hsum.par_chunks_mut(width)
        .zip(src.par_chunks(width))
        .for_each(|(hrow, row)| {
            let mut s: u32 = (lo..=hi).map(|k| tap(row, k)).sum();
            hrow[0] = s;
            for x in 1..width {
                s += tap(row, x as isize + hi);
                s -= tap(row, x as isize - 1 + lo);
                hrow[x] = s;
            }
        });
    // Window sum of an out-of-image row under a `Constant` border.
    let constant_row = vec![constant * size as u32; width];
    let hrow_at = |r: isize| -> &[u32] {
        match map_coord(r, height, border) {
            Some(iy) => &hsum[iy * width..][..width],
            None => &constant_row,
        }
    };

    let n = size * size;
    let w = 1.0 / n as f32;
    // Twice the float-sum error bound, relative to the mean (see doc comment).
    let rel_tol = 2.0 * (n as f64 + 3.0) * f64::from(f32::EPSILON) * 0.5 * 1.01;
    // Output byte for window sum `S`, or `BOX_BLUR_REPLAY` where the float
    // loop's rounding must be replayed.
    let decide = |s: u32| -> u16 {
        let m = s as f64 / n as f64;
        let dist_to_half = (m - (m.floor() + 0.5)).abs();
        if dist_to_half > m * rel_tol {
            m.round() as u16
        } else {
            BOX_BLUR_REPLAY
        }
    };
    // Tabulate every possible `S` in `0..=255 n` when that is small (boxes up
    // to 64x64, ~1 M entries); larger boxes decide per pixel.
    let lut: Option<Vec<u16>> = (n <= 64 * 64).then(|| (0..=255 * n as u32).map(decide).collect());

    // Vertical running sums over bands of rows.
    const BAND: usize = 32;
    out.as_mut()
        .par_chunks_mut(width * BAND)
        .enumerate()
        .for_each(|(band, out_band)| {
            let y0 = band * BAND;
            let rows = out_band.len() / width;
            let mut col: Vec<u32> = vec![0; width];
            for k in lo..=hi {
                let r = hrow_at(y0 as isize + k);
                for (c, &v) in col.iter_mut().zip(r) {
                    *c += v;
                }
            }
            for dy in 0..rows {
                let y = y0 + dy;
                if dy > 0 {
                    let add = hrow_at(y as isize + hi);
                    let sub = hrow_at(y as isize - 1 + lo);
                    for ((c, &a), &s) in col.iter_mut().zip(add).zip(sub) {
                        *c = *c + a - s;
                    }
                }
                let out_row = &mut out_band[dy * width..(dy + 1) * width];
                for (x, (o, &s)) in out_row.iter_mut().zip(&col).enumerate() {
                    let v = match &lut {
                        Some(lut) => lut[s as usize],
                        None => decide(s),
                    };
                    *o = if v != BOX_BLUR_REPLAY {
                        v as u8
                    } else {
                        box_blur_dense_pixel(src, width, height, x, y, lo, hi, w, border)
                    };
                }
            }
        });
}

/// One pixel of the dense box convolution, in the dense loop's exact
/// accumulation order (rows outer, columns inner) and border handling.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn box_blur_dense_pixel(
    src: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    lo: isize,
    hi: isize,
    w: f32,
    border: BorderMode,
) -> u8 {
    let constant = match border {
        BorderMode::Constant(v) => v as f32,
        _ => 0.0,
    };
    let mut sum = 0.0f32;
    for ky in lo..=hi {
        let iy = map_coord(y as isize + ky, height, border);
        for kx in lo..=hi {
            let ix = map_coord(x as isize + kx, width, border);
            let v = match (ix, iy) {
                (Some(ix), Some(iy)) => src[iy * width + ix] as f32,
                _ => constant,
            };
            sum += v * w;
        }
    }
    sum.round_ties_even().clamp(0.0, 255.0) as u8
}

pub fn sharpen(image: &GrayImage, amount: f32) -> GrayImage {
    let mut kernel = laplacian_kernel();
    for v in &mut kernel.data {
        *v *= -amount;
    }
    let center = kernel.width / 2;
    // The scaled Laplacian already contributes `4 * amount` at the centre, so
    // adding 1.0 yields the standard unsharp kernel (centre `1 + 4 * amount`,
    // neighbours `-amount`) with unit DC gain. Adding `1.0 + amount` instead
    // made the DC gain `1 + amount`, brightening the whole image.
    kernel.data[center * kernel.width + center] += 1.0;
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

    #[test]
    fn sharpen_has_unit_dc_gain() {
        // Regression: the kernel centre carried an extra `amount`, giving DC
        // gain `1 + amount` and brightening flat regions.
        let img = GrayImage::from_pixel(6, 6, Luma([100]));
        for amount in [0.0f32, 0.5, 1.0, 2.0] {
            let out = sharpen(&img, amount);
            assert!(
                out.as_raw().iter().all(|&p| p == 100),
                "sharpen(amount = {amount}) brightened a flat image"
            );
        }

        // Sharpening an edge must still amplify the local contrast.
        let mut edge = GrayImage::new(6, 6);
        for y in 0..6 {
            for x in 0..6 {
                edge.put_pixel(x, y, Luma([if x < 3 { 100 } else { 120 }]));
            }
        }
        let sharpened = sharpen(&edge, 1.0);
        let dark_side = sharpened.get_pixel(2, 3)[0];
        let bright_side = sharpened.get_pixel(3, 3)[0];
        assert!(dark_side < 100, "expected undershoot beside the edge");
        assert!(bright_side > 120, "expected overshoot beside the edge");
    }
}
