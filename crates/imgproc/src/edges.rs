use cv_core::{Tensor, storage::Storage};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::convolve::{
    BorderMode, Kernel, convolve_with_border_into_ctx, gaussian_blur_with_border_into_ctx,
};

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

fn apply_linear_transform(mut img: GrayImage, scale: f32, delta: f32) -> GrayImage {
    img.as_mut().par_iter_mut().for_each(|px| {
        let v = (*px as f32) * scale + delta;
        *px = v.round_ties_even().clamp(0.0, 255.0) as u8;
    });
    img
}

/// The `ksize` values [`sobel_ex`] can build separable Sobel kernels for.
pub const SOBEL_SUPPORTED_KSIZES: &[usize] = &[3, 5, 7];

/// Validate a Sobel kernel size.
///
/// Returns `Err` for any size that has no separable 1-D Sobel derivative /
/// smoothing kernel pair. Previously an unsupported size silently fell back to
/// the 3x3 pair, so `ksize = 4` produced bit-identical output to `ksize = 3`
/// with no error and no warning.
pub fn validate_sobel_ksize(ksize: usize) -> crate::Result<()> {
    if SOBEL_SUPPORTED_KSIZES.contains(&ksize) {
        Ok(())
    } else {
        Err(crate::ImgprocError::InvalidInput(format!(
            "sobel_ex: unsupported ksize = {}; supported sizes are {:?}. \
             An unsupported size is no longer silently treated as 3",
            ksize, SOBEL_SUPPORTED_KSIZES
        )))
    }
}

/// Compute a Sobel derivative image with an explicit kernel size.
///
/// `ksize` must be one of [`SOBEL_SUPPORTED_KSIZES`] (3, 5, 7).
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] if `ksize` is not supported,
/// instead of silently substituting the 3x3 kernel.
#[allow(clippy::too_many_arguments)]
pub fn sobel_ex(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
) -> crate::Result<GrayImage> {
    validate_sobel_ksize(ksize)?;
    match cv_runtime::best_runner().map(|(r, _)| r) {
        Ok(runner) => sobel_ex_ctx(src, dx, dy, ksize, scale, delta, border, &runner),
        Err(_) => {
            warn!("GPU not available for sobel, falling back to CPU");
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

/// Compute a Sobel derivative image with an explicit kernel size and an
/// explicit execution group.
///
/// # Errors
/// Returns [`crate::ImgprocError::InvalidInput`] if `ksize` is not one of
/// [`SOBEL_SUPPORTED_KSIZES`].
#[allow(clippy::too_many_arguments)]
pub fn sobel_ex_ctx(
    src: &GrayImage,
    dx: i32,
    dy: i32,
    ksize: usize,
    scale: f32,
    delta: f32,
    border: BorderMode,
    group: &RuntimeRunner,
) -> crate::Result<GrayImage> {
    validate_sobel_ksize(ksize)?;
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        // NOTE: GPU Sobel shader (sobel.wgsl) expects packed u8-in-u32 input but the Rust
        // host side uploads f32 tensors. The GPU path only works for ksize==3 single-axis
        // Sobel because `sobel_gpu()` converts to f32 before dispatch and converts back
        // after readback.  A full fix requires either (a) rewriting the WGSL shader to
        // accept f32 storage buffers, or (b) packing u8 pixels into u32 on the host before
        // upload.  Until then, all other cases fall through to the CPU separable path.
        if ksize == 3
            && ((dx == 1 && dy == 0) || (dx == 0 && dy == 1))
            && let Ok(result) = sobel_gpu(gpu, src, ksize)
        {
            info!(
                "GPU sobel_ex: executed on GPU ({}x{})",
                src.width(),
                src.height()
            );
            let (gx, gy) = result;
            let target = if dx == 1 { gx } else { gy };
            return Ok(apply_linear_transform(target, scale, delta));
        }
    }

    let mut out = GrayImage::new(src.width(), src.height());
    // `ksize` was validated above, so this cannot take the fallback branch.
    let (deriv, smooth) = sobel_kernels_1d(ksize).ok_or_else(|| {
        crate::ImgprocError::InvalidInput(format!(
            "sobel_ex: no separable Sobel kernel for ksize = {}",
            ksize
        ))
    })?;

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
    Ok(apply_linear_transform(out, scale, delta))
}

fn sobel_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    ksize: usize,
) -> cv_hal::Result<(GrayImage, GrayImage)> {
    use cv_hal::context::ComputeContext;
    let src_f32: Vec<f32> = src.as_raw().iter().map(|&p| p as f32).collect();
    let input_tensor = cv_core::CpuTensor::from_vec(
        src_f32,
        cv_core::TensorShape::new(1, src.height() as usize, src.width() as usize),
    )
    .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
    let input_gpu = input_tensor.to_gpu_ctx(gpu)?;

    let (gx_gpu, gy_gpu) = gpu.sobel(&input_gpu, 1, 1, ksize)?;

    let gx_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = gx_gpu.to_cpu()?;
    let gy_cpu: Tensor<f32, cv_core::CpuStorage<f32>> = gy_gpu.to_cpu()?;

    let gx_data: Vec<u8> = gx_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();
    let gy_data: Vec<u8> = gy_cpu
        .storage
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();

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
    match cv_runtime::best_runner().map(|(r, _)| r) {
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

    // Use separable convolution for performance (Scharr is separable: [-1,0,1] x [3,10,3])
    crate::convolve::separable_convolve_into_ctx(src, &mut out, kx, ky, border, group);
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
    let gx = sobel_ex_ctx(src, 1, 0, 3, 1.0, 0.0, border, &runner).expect("ksize=3 is supported");
    let gy = sobel_ex_ctx(src, 0, 1, 3, 1.0, 0.0, border, &runner).expect("ksize=3 is supported");
    (gx, gy)
}

/// 3x3 Sobel `(dx, dy)` pair with OpenCV's default border,
/// [`BorderMode::Reflect101`] (it used to be `Replicate`). Use
/// [`sobel_with_border`] for any other border.
pub fn sobel(src: &GrayImage) -> (GrayImage, GrayImage) {
    sobel_with_border(src, BorderMode::Reflect101)
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

/// Scharr `(dx, dy)` pair with OpenCV's default border,
/// [`BorderMode::Reflect101`] (it used to be `Replicate`). Use
/// [`scharr_with_border`] for any other border.
pub fn scharr(src: &GrayImage) -> (GrayImage, GrayImage) {
    scharr_with_border(src, BorderMode::Reflect101)
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
                *out = mag.round_ties_even().min(255.0) as u8;
            });

        GrayImage::from_raw(width, height, output).unwrap_or_else(|| GrayImage::new(width, height))
    })
}

/// `cv2.Laplacian(src, CV_8U, ksize=1)`: the 4-neighbour kernel
/// `[0 1 0; 1 -4 1; 0 1 0]`, saturated to `u8`, with OpenCV's default border
/// [`BorderMode::Reflect101`] (it used to be `Replicate`). Use
/// [`laplacian_with_border`] for any other border.
pub fn laplacian(src: &GrayImage) -> GrayImage {
    laplacian_with_border(src, BorderMode::Reflect101)
}

/// [`laplacian`] with an explicit border.
pub fn laplacian_with_border(src: &GrayImage, border: BorderMode) -> GrayImage {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback: use CPU registry if available
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    laplacian_with_border_ctx(src, border, &runner)
}

/// [`laplacian`] on an explicit runner (reflect-101 border).
pub fn laplacian_ctx(src: &GrayImage, group: &RuntimeRunner) -> GrayImage {
    laplacian_with_border_ctx(src, BorderMode::Reflect101, group)
}

/// [`laplacian_with_border`] on an explicit runner.
pub fn laplacian_with_border_ctx(
    src: &GrayImage,
    border: BorderMode,
    group: &RuntimeRunner,
) -> GrayImage {
    let kernel = Kernel::from_slice(&[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], 3, 3);
    let mut out = GrayImage::new(src.width(), src.height());
    convolve_with_border_into_ctx(src, &mut out, &kernel, border, group);
    out
}

/// Options for [`canny_with_options`].
///
/// The default reproduces `cv2.Canny(img, low, high)`: aperture 3,
/// `L2gradient = false`, no smoothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CannyOptions {
    /// Use `sqrt(dx² + dy²)` instead of `|dx| + |dy|` (OpenCV's
    /// `L2gradient`). Thresholds stay in gradient units either way.
    pub l2_gradient: bool,
    /// Gaussian pre-blur (σ, BORDER_REFLECT_101) before the gradient.
    /// OpenCV's Canny does not smooth; `None` matches it.
    pub blur_sigma: Option<f32>,
}

/// OpenCV's Canny on an 8-bit image: Sobel 3x3 (BORDER_REPLICATE), integer
/// L1 or squared-L2 magnitude, NMS strict towards left/up (`>`) and
/// non-strict towards right/down (`>=`), strict on both diagonal neighbours,
/// thresholds strict, 8-connected hysteresis. Pixels outside the image have
/// magnitude 0. The implementation lives in cv-hal
/// (`cv_hal::cpu::canny_opencv`) so the GPU kernel shares its thresholds and
/// hysteresis.
fn canny_cpu(src: &GrayImage, low: f64, high: f64, l2: bool) -> GrayImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let out = cv_hal::cpu::canny_opencv::canny_u8(src.as_raw(), w, h, low, high, l2);
    GrayImage::from_raw(src.width(), src.height(), out)
        .unwrap_or_else(|| GrayImage::new(src.width(), src.height()))
}

/// Canny edge detector, `cv2.Canny(src, low_threshold, high_threshold)`.
///
/// No smoothing is applied (OpenCV does not smooth either); blur the input
/// first, or use [`canny_with_options`] with `blur_sigma`, if it is noisy.
/// The gradient is a 3x3 Sobel and the magnitude is L1 (`|dx| + |dy|`).
/// Output is 0/255.
pub fn canny(src: &GrayImage, low_threshold: u8, high_threshold: u8) -> GrayImage {
    canny_with_options(
        src,
        low_threshold as f32,
        high_threshold as f32,
        &CannyOptions::default(),
    )
}

/// Canny with f32 thresholds (may exceed 255: the L1 Sobel magnitude goes up
/// to 2040) and [`CannyOptions`]. `CannyOptions::default()` matches
/// `cv2.Canny(src, low, high)`; `l2_gradient: true` matches
/// `cv2.Canny(src, low, high, L2gradient=True)`. `low > high` are swapped, as
/// in OpenCV.
pub fn canny_with_options(
    src: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    options: &CannyOptions,
) -> GrayImage {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        warn!("GPU not available for canny, falling back to CPU");
        cv_runtime::registry()
            .ok()
            .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
            .unwrap_or_else(|| cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0)))
    });
    canny_with_options_ctx(src, low_threshold, high_threshold, options, &runner)
}

/// [`canny`] on a given runner.
pub fn canny_ctx(
    src: &GrayImage,
    low_threshold: u8,
    high_threshold: u8,
    group: &RuntimeRunner,
) -> GrayImage {
    canny_with_options_ctx(
        src,
        low_threshold as f32,
        high_threshold as f32,
        &CannyOptions::default(),
        group,
    )
}

/// [`canny_with_options`] on a given runner.
///
/// On a GPU runner the `canny.wgsl` kernel runs Sobel, NMS and the double
/// threshold (integer arithmetic, bit-exact with the CPU path and with
/// OpenCV, L1 or L2); hysteresis runs on the host. The optional pre-blur
/// always uses the CPU Gaussian, because the GPU Gaussian ignores the border
/// mode (BORDER_REFLECT_101 here) and would change the edges. On any GPU
/// error the CPU path runs instead.
pub fn canny_with_options_ctx(
    src: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    options: &CannyOptions,
    group: &RuntimeRunner,
) -> GrayImage {
    if src.width() == 0 || src.height() == 0 {
        return GrayImage::new(src.width(), src.height());
    }

    let blurred;
    let input = match options.blur_sigma {
        Some(sigma) if sigma > 0.0 => {
            blurred = gaussian_blur_with_border_into_ctx(src, sigma, BorderMode::Reflect101, group);
            &blurred
        }
        _ => src,
    };

    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        match canny_gpu(
            gpu,
            input,
            low_threshold,
            high_threshold,
            options.l2_gradient,
        ) {
            Ok(res) => {
                info!(
                    "GPU canny: executed on GPU ({}x{})",
                    src.width(),
                    src.height()
                );
                return res;
            }
            Err(e) => warn!("GPU canny failed ({e}), falling back to CPU"),
        }
    }

    group.run(|| {
        canny_cpu(
            input,
            low_threshold as f64,
            high_threshold as f64,
            options.l2_gradient,
        )
    })
}

fn canny_gpu(
    gpu: &cv_hal::gpu::GpuContext,
    src: &GrayImage,
    low: f32,
    high: f32,
    l2: bool,
) -> cv_hal::Result<GrayImage> {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let out =
        cv_hal::gpu_kernels::canny::canny_u8(gpu, src.as_raw(), w, h, low as f64, high as f64, l2)?;
    GrayImage::from_raw(src.width(), src.height(), out)
        .ok_or_else(|| cv_hal::Error::MemoryError("Failed to create image from mask".into()))
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
    fn test_canny_finds_step_edge() {
        // 64x64 image: left half = 0, right half = 255
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }

        let edges = canny(&img, 50, 150);

        // Check that edges are detected near columns 31-32 (the boundary)
        // Count edge pixels in columns 30-33 vs. the rest of the image
        let mut edge_count_near_boundary = 0u32;
        let mut edge_count_elsewhere = 0u32;
        // Only check interior rows
        for y in 5..59 {
            for x in 0..64 {
                if edges.get_pixel(x, y)[0] > 0 {
                    if (30..=33).contains(&x) {
                        edge_count_near_boundary += 1;
                    } else {
                        edge_count_elsewhere += 1;
                    }
                }
            }
        }

        assert!(
            edge_count_near_boundary > 0,
            "Canny should detect edges near the step boundary (columns 31-32)"
        );
        // The vast majority of edge pixels should be at the boundary
        assert!(
            edge_count_near_boundary > edge_count_elsewhere,
            "Most edges should be at the boundary: boundary={}, elsewhere={}",
            edge_count_near_boundary,
            edge_count_elsewhere
        );
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
