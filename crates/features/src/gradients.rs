//! Shared 1-D Sobel / Gaussian gradient helpers for the corner detectors.
//!
//! `harris.rs` and `gftt.rs` previously carried byte-identical private copies of
//! these helpers; they now both use this module. `cv-imgproc/src/edges.rs` has a
//! similarly-shaped `sobel_kernels_1d`, but with a different scalar type (f32),
//! kernel-size type (usize) and a fallible `Option` return, so it is left as-is
//! rather than forced through an f64 signature.

use image::GrayImage;

/// Return 1-D Sobel kernel pair (derivative, smoothing) for the given aperture size.
fn sobel_kernels_1d(ksize: i32) -> (Vec<f64>, Vec<f64>) {
    match ksize {
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        5 => (
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        ),
        7 => (
            vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
            vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        ),
        _ => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
    }
}

/// Build a normalised 1-D Gaussian kernel of the given (odd) size and sigma.
pub(crate) fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64> {
    let center = (size / 2) as isize;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = (i as isize - center) as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
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

/// Clamp-border pixel fetch helper.
fn pixel_at(image: &GrayImage, x: i32, y: i32) -> f64 {
    let cx = x.clamp(0, image.width() as i32 - 1) as u32;
    let cy = y.clamp(0, image.height() as i32 - 1) as u32;
    image.get_pixel(cx, cy)[0] as f64
}

/// Compute Sobel gradients Ix, Iy for every pixel using separable 1-D kernels.
///
/// Returns two `Vec<f64>` of length width*height, stored in row-major order.
pub(crate) fn compute_sobel_gradients(image: &GrayImage, ksize: i32) -> (Vec<f64>, Vec<f64>) {
    let w = image.width() as i32;
    let h = image.height() as i32;
    let n = (w * h) as usize;

    let (deriv, smooth) = sobel_kernels_1d(ksize);
    let half = (deriv.len() / 2) as i32;

    // Ix: derivative in x, smoothing in y  => horiz pass uses deriv, vert pass uses smooth
    // Iy: smoothing in x, derivative in y  => horiz pass uses smooth, vert pass uses deriv

    // Horizontal pass for Ix (deriv in x)
    let mut tmp_ix = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                sum += pixel_at(image, x + k, y) * deriv[(k + half) as usize];
            }
            tmp_ix[(y * w + x) as usize] = sum;
        }
    }

    // Vertical pass for Ix (smooth in y)
    let mut ix = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                let sy = (y + k).clamp(0, h - 1);
                sum += tmp_ix[(sy * w + x) as usize] * smooth[(k + half) as usize];
            }
            ix[(y * w + x) as usize] = sum;
        }
    }

    // Horizontal pass for Iy (smooth in x)
    let mut tmp_iy = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                sum += pixel_at(image, x + k, y) * smooth[(k + half) as usize];
            }
            tmp_iy[(y * w + x) as usize] = sum;
        }
    }

    // Vertical pass for Iy (deriv in y)
    let mut iy = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                let sy = (y + k).clamp(0, h - 1);
                sum += tmp_iy[(sy * w + x) as usize] * deriv[(k + half) as usize];
            }
            iy[(y * w + x) as usize] = sum;
        }
    }

    (ix, iy)
}

/// Apply separable 1-D Gaussian blur to an f64 buffer (row-major, width x height).
pub(crate) fn gaussian_blur_f64(
    buf: &[f64],
    width: usize,
    height: usize,
    kernel: &[f64],
) -> Vec<f64> {
    let half = (kernel.len() / 2) as isize;
    let n = width * height;

    // Horizontal pass
    let mut tmp = vec![0.0f64; n];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for k in -half..=half {
                let sx = (x as isize + k).clamp(0, width as isize - 1) as usize;
                sum += buf[y * width + sx] * kernel[(k + half) as usize];
            }
            tmp[y * width + x] = sum;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f64; n];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for k in -half..=half {
                let sy = (y as isize + k).clamp(0, height as isize - 1) as usize;
                sum += tmp[sy * width + x] * kernel[(k + half) as usize];
            }
            out[y * width + x] = sum;
        }
    }

    out
}
