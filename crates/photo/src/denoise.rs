//! Image denoising algorithms
//!
//! Provides noise reduction methods for grayscale and color images.
//!
//! # Algorithms
//!
//! - **Non-Local Means (NLMeans)**: Patch-based denoising that finds and averages similar
//!   patches across the image. Preserves texture and detail better than linear filters.
//! - **NLMeans Colored**: NLMeans in perceptual (Lab-like) color space with separate
//!   strength parameters for luminance and chrominance.
//! - **Gaussian denoising**: Simple Gaussian blur for fast smoothing.

use cv_core::float::Float;
use cv_core::tensor::{CpuTensor, TensorShape};
use cv_core::Result;

/// Non-Local Means denoising for grayscale images.
///
/// For each pixel, searches a local window for patches with similar structure
/// and computes a weighted average, where weights decay exponentially with
/// patch dissimilarity.
///
/// # Arguments
/// * `image` - Single-channel image (1, H, W), values typically in [0, 1].
/// * `h` - Filter strength. Higher values remove more noise but also more detail.
/// * `template_window` - Size of the comparison patch (must be odd; default 7).
/// * `search_window` - Size of the search area (must be odd; default 21).
///
/// # Returns
/// Denoised single-channel image with the same shape.
pub fn fast_nl_means_denoising<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    h: T,
    template_window: usize,
    search_window: usize,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();

    if channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "fast_nl_means_denoising expects a single-channel image (1, H, W)".into(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(cv_core::Error::InvalidInput(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let tw = if template_window.is_multiple_of(2) {
        template_window + 1
    } else {
        template_window
    };
    let sw = if search_window.is_multiple_of(2) {
        search_window + 1
    } else {
        search_window
    };

    let t_half = tw / 2;
    let s_half = sw / 2;

    let h_f64 = Float::to_f64(h);
    let h_sq = h_f64 * h_f64;
    if h_sq < 1e-15 {
        return Ok(image.clone());
    }

    let data = image.as_slice()?;
    let src: Vec<f64> = data.iter().map(|v| Float::to_f64(*v)).collect();

    let n_pixels = height * width;
    let mut out = vec![0.0f64; n_pixels];

    // Process each pixel.
    for py in 0..height {
        for px in 0..width {
            let mut sum_weight = 0.0f64;
            let mut sum_val = 0.0f64;

            // Search window bounds.
            let sy_start = py.saturating_sub(s_half);
            let sy_end = (py + s_half + 1).min(height);
            let sx_start = px.saturating_sub(s_half);
            let sx_end = (px + s_half + 1).min(width);

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    // Compute squared distance between patches centered at (py,px) and (sy,sx).
                    let mut dist_sq = 0.0f64;
                    let mut count = 0;

                    let ty_start = t_half.min(py).min(sy);
                    let ty_end_p = (height - py).min(t_half + 1);
                    let ty_end_s = (height - sy).min(t_half + 1);
                    let ty_end = ty_end_p.min(ty_end_s);

                    let tx_start = t_half.min(px).min(sx);
                    let tx_end_p = (width - px).min(t_half + 1);
                    let tx_end_s = (width - sx).min(t_half + 1);
                    let tx_end = tx_end_p.min(tx_end_s);

                    for dy_offset in 0..(ty_start + ty_end) {
                        let dy = dy_offset as isize - ty_start as isize;
                        let p_row = (py as isize + dy) as usize;
                        let s_row = (sy as isize + dy) as usize;

                        for dx_offset in 0..(tx_start + tx_end) {
                            let dx = dx_offset as isize - tx_start as isize;
                            let p_col = (px as isize + dx) as usize;
                            let s_col = (sx as isize + dx) as usize;

                            let diff = src[p_row * width + p_col] - src[s_row * width + s_col];
                            dist_sq += diff * diff;
                            count += 1;
                        }
                    }

                    if count > 0 {
                        dist_sq /= count as f64;
                    }

                    let w = (-dist_sq / h_sq).exp();
                    sum_weight += w;
                    sum_val += w * src[sy * width + sx];
                }
            }

            out[py * width + px] = if sum_weight > 1e-15 {
                sum_val / sum_weight
            } else {
                src[py * width + px]
            };
        }
    }

    let result: Vec<T> = out.iter().map(|&v| T::from_f64(v)).collect();
    CpuTensor::<T>::from_vec(result, TensorShape::new(1, height, width))
}

/// Non-Local Means denoising for color images.
///
/// Separates the image into luminance and chrominance components (approximate
/// Lab-like space) and applies NLMeans with different filter strengths.
///
/// # Arguments
/// * `image` - 3-channel image (3, H, W), values in [0, 1].
/// * `h_luminance` - Filter strength for the luminance channel.
/// * `h_color` - Filter strength for the chrominance channels.
/// * `template_window` - Patch size (odd integer, default 7).
/// * `search_window` - Search area size (odd integer, default 21).
///
/// # Returns
/// Denoised 3-channel image with the same shape.
pub fn fast_nl_means_denoising_colored<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    h_luminance: T,
    h_color: T,
    template_window: usize,
    search_window: usize,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();

    if channels != 3 {
        return Err(cv_core::Error::InvalidInput(
            "fast_nl_means_denoising_colored expects a 3-channel image (3, H, W)".into(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(cv_core::Error::InvalidInput(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let data = image.as_slice()?;
    let n_pixels = height * width;

    // Convert RGB to approximate Lab-like space:
    //   L = 0.2126*R + 0.7152*G + 0.0722*B
    //   a = R - G  (crude opponent channel)
    //   b = 0.5*(R + G) - B  (crude opponent channel)
    let mut l_data = vec![T::ZERO; n_pixels];
    let mut a_data = vec![T::ZERO; n_pixels];
    let mut b_data = vec![T::ZERO; n_pixels];

    let c0_213 = T::from_f64(0.2126);
    let c0_715 = T::from_f64(0.7152);
    let c0_072 = T::from_f64(0.0722);
    let c0_5 = T::from_f64(0.5);

    for i in 0..n_pixels {
        let r = data[i];
        let g = data[n_pixels + i];
        let b_val = data[2 * n_pixels + i];

        l_data[i] = c0_213 * r + c0_715 * g + c0_072 * b_val;
        a_data[i] = r - g;
        b_data[i] = c0_5 * (r + g) - b_val;
    }

    // Denoise each component.
    let l_tensor = CpuTensor::<T>::from_vec(l_data, TensorShape::new(1, height, width))?;
    let a_tensor = CpuTensor::<T>::from_vec(a_data, TensorShape::new(1, height, width))?;
    let b_tensor = CpuTensor::<T>::from_vec(b_data, TensorShape::new(1, height, width))?;

    let l_denoised =
        fast_nl_means_denoising(&l_tensor, h_luminance, template_window, search_window)?;
    let a_denoised = fast_nl_means_denoising(&a_tensor, h_color, template_window, search_window)?;
    let b_denoised = fast_nl_means_denoising(&b_tensor, h_color, template_window, search_window)?;

    // Convert back to RGB.
    let l_out = l_denoised.as_slice()?;
    let a_out = a_denoised.as_slice()?;
    let b_out = b_denoised.as_slice()?;

    let mut rgb_data = vec![T::ZERO; 3 * n_pixels];

    // Inverse transform (approximate):
    //   R = L + 0.7152 * a + 0.0722 * b_ch  (reconstructed from the opponent channels)
    //   G = L - 0.2126 * a + 0.0722 * b_ch
    //   B = L + offset - b_ch
    // Since our forward transform is simple, we invert by solving:
    //   L = 0.2126*R + 0.7152*G + 0.0722*B
    //   a = R - G
    //   b_ch = 0.5*(R+G) - B
    // => R = L + a*(0.7152 + 0.0722)/(1) ... actually just solve the 3x3 system.
    // Forward: [L; a; b] = M * [R; G; B]
    //   M = [0.2126, 0.7152, 0.0722;
    //        1,      -1,      0;
    //        0.5,     0.5,   -1]
    // Inverse: [R; G; B] = M^{-1} * [L; a; b]
    // M^{-1} (precomputed):
    //   R = L + 0.7874*a + 0.0722*b_ch   (approximately)
    //   But let's just solve it properly.

    // det(M) = 0.2126*(-1*(-1) - 0*0.5) - 0.7152*(1*(-1) - 0*0.5) + 0.0722*(1*0.5 - (-1)*0.5)
    //        = 0.2126*(1) - 0.7152*(-1) + 0.0722*(1)
    //        = 0.2126 + 0.7152 + 0.0722 = 1.0

    // Cofactor matrix (transposed = inverse since det=1):
    //   C00 = (-1)(-1) - (0)(0.5)   = 1
    //   C01 = -((1)(-1) - (0)(0.5)) = 1
    //   C02 = (1)(0.5) - (-1)(0.5)  = 1
    //   C10 = -(0.7152*(-1) - 0.0722*0.5) = 0.7152 + 0.0361 = 0.7513
    //   C11 = 0.2126*(-1) - 0.0722*0.5    = -0.2126 - 0.0361 = -0.2487
    //   C12 = -(0.2126*0.5 - 0.7152*0.5)  = -(0.1063 - 0.3576) = 0.2513
    //   C20 = 0.7152*0 - 0.0722*(-1)      = 0.0722
    //   C21 = -(0.2126*0 - 0.0722*1)      = 0.0722
    //   C22 = 0.2126*(-1) - 0.7152*1      = -0.9278

    // Actually, M^{-1} = adj(M)/det(M). adj(M) = cofactor(M)^T.
    // Let me just use the direct inversion. Since det=1, M^{-1} = adj(M).
    // adj(M)_ij = cofactor(M)_ji
    // So M^{-1}:
    //   Row 0: C00, C10, C20 = 1, 0.7513, 0.0722
    //   Row 1: C01, C11, C21 = 1, -0.2487, 0.0722
    //   Row 2: C02, C12, C22 = 1, 0.2513, -0.9278

    let m00 = T::from_f64(1.0);
    let m01 = T::from_f64(0.7513);
    let m02 = T::from_f64(0.0722);
    let m10 = T::from_f64(1.0);
    let m11 = T::from_f64(-0.2487);
    let m12 = T::from_f64(0.0722);
    let m20 = T::from_f64(1.0);
    let m21 = T::from_f64(0.2513);
    let m22 = T::from_f64(-0.9278);

    for i in 0..n_pixels {
        let l = l_out[i];
        let a = a_out[i];
        let b = b_out[i];

        rgb_data[i] = m00 * l + m01 * a + m02 * b;
        rgb_data[n_pixels + i] = m10 * l + m11 * a + m12 * b;
        rgb_data[2 * n_pixels + i] = m20 * l + m21 * a + m22 * b;
    }

    CpuTensor::<T>::from_vec(rgb_data, TensorShape::new(3, height, width))
}

/// Gaussian denoising via Gaussian blur.
///
/// Applies a Gaussian kernel to smooth the image, reducing high-frequency noise.
/// This is a simple but fast denoising method.
///
/// # Arguments
/// * `image` - Input image (CHW layout).
/// * `sigma` - Standard deviation of the Gaussian kernel. Larger = more smoothing.
///
/// # Returns
/// Smoothed image with the same shape.
#[allow(clippy::needless_range_loop)]
pub fn denoise_gaussian<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    sigma: T,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();
    let n_pixels = height * width;

    if height == 0 || width == 0 {
        return Err(cv_core::Error::InvalidInput(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let sigma_f64 = Float::to_f64(sigma);
    if sigma_f64 < 1e-10 {
        return Ok(image.clone());
    }

    // Compute kernel radius (3*sigma covers 99.7%).
    let radius = (3.0 * sigma_f64).ceil() as usize;
    let ksize = 2 * radius + 1;

    // Build 1D Gaussian kernel.
    let mut kernel = vec![0.0f64; ksize];
    let coeff = -0.5 / (sigma_f64 * sigma_f64);
    let mut sum = 0.0f64;
    for i in 0..ksize {
        let d = i as f64 - radius as f64;
        kernel[i] = (d * d * coeff).exp();
        sum += kernel[i];
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    let data = image.as_slice()?;
    let src: Vec<f64> = data.iter().map(|v| Float::to_f64(*v)).collect();

    // Separable Gaussian: horizontal pass then vertical pass.
    let mut temp = vec![0.0f64; channels * n_pixels];
    let mut out = vec![0.0f64; channels * n_pixels];

    // Horizontal pass.
    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for y in 0..height {
            for x in 0..width {
                let mut val = 0.0;
                for ki in 0..ksize {
                    let sx = x as isize + ki as isize - radius as isize;
                    // Clamp to border (replicate edge).
                    let sx = sx.max(0).min(width as isize - 1) as usize;
                    val += kernel[ki] * src[ch_offset + y * width + sx];
                }
                temp[ch_offset + y * width + x] = val;
            }
        }
    }

    // Vertical pass.
    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for y in 0..height {
            for x in 0..width {
                let mut val = 0.0;
                for ki in 0..ksize {
                    let sy = y as isize + ki as isize - radius as isize;
                    let sy = sy.max(0).min(height as isize - 1) as usize;
                    val += kernel[ki] * temp[ch_offset + sy * width + x];
                }
                out[ch_offset + y * width + x] = val;
            }
        }
    }

    let result: Vec<T> = out.iter().map(|&v| T::from_f64(v)).collect();
    CpuTensor::<T>::from_vec(result, TensorShape::new(channels, height, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::tensor::TensorShape;

    fn make_test_image(h: usize, w: usize, value: f32) -> CpuTensor<f32> {
        let data = vec![value; h * w];
        CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap()
    }

    fn make_noisy_image(h: usize, w: usize, base: f32, noise_pattern: &[f32]) -> CpuTensor<f32> {
        let n = h * w;
        let mut data = vec![base; n];
        for (i, d) in data.iter_mut().enumerate() {
            *d += noise_pattern[i % noise_pattern.len()];
            *d = d.clamp(0.0, 1.0);
        }
        CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap()
    }

    // --- NLMeans grayscale tests ---

    #[test]
    fn test_nlmeans_preserves_uniform() {
        let img = make_test_image(8, 8, 0.5);
        let result = fast_nl_means_denoising(&img, 0.1f32, 7, 21).unwrap();
        let data = result.as_slice().unwrap();
        for &v in data {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "Uniform image should be preserved, got {}",
                v
            );
        }
    }

    #[test]
    fn test_nlmeans_reduces_noise() {
        let h = 16;
        let w = 16;
        let base = 0.5f32;

        // Deterministic "noise" pattern.
        let noise_pattern = [0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03, -0.03];
        let noisy = make_noisy_image(h, w, base, &noise_pattern);
        let clean = make_test_image(h, w, base);

        let denoised = fast_nl_means_denoising(&noisy, 0.15f32, 5, 11).unwrap();

        // Compute MSE of noisy vs clean and denoised vs clean.
        let clean_data = clean.as_slice().unwrap();
        let noisy_data = noisy.as_slice().unwrap();
        let denoised_data = denoised.as_slice().unwrap();

        let n = h * w;
        let mse_noisy: f64 = (0..n)
            .map(|i| {
                let d = (noisy_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let mse_denoised: f64 = (0..n)
            .map(|i| {
                let d = (denoised_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        // Denoised should have lower MSE than noisy.
        assert!(
            mse_denoised < mse_noisy,
            "Denoising should reduce MSE: noisy={:.6}, denoised={:.6}",
            mse_noisy,
            mse_denoised
        );
    }

    #[test]
    fn test_nlmeans_zero_h_returns_original() {
        let img = make_test_image(8, 8, 0.5);
        // With h=0 the function should return a clone.
        let result = fast_nl_means_denoising(&img, 0.0f32, 7, 21).unwrap();
        let a = img.as_slice().unwrap();
        let b = result.as_slice().unwrap();
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nlmeans_wrong_channels() {
        let data = vec![0.5f32; 3 * 8 * 8];
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(3, 8, 8)).unwrap();
        let result = fast_nl_means_denoising(&img, 0.1f32, 7, 21);
        assert!(result.is_err());
    }

    // --- NLMeans colored tests ---

    #[test]
    fn test_nlmeans_colored_preserves_uniform() {
        let n = 8 * 8;
        let data = vec![0.5f32; 3 * n];
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(3, 8, 8)).unwrap();

        let result = fast_nl_means_denoising_colored(&img, 0.1f32, 0.1f32, 5, 11).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, 8, 8));

        let out = result.as_slice().unwrap();
        for &v in out {
            assert!(
                (v - 0.5).abs() < 0.05,
                "Uniform color image should be approximately preserved, got {}",
                v
            );
        }
    }

    #[test]
    fn test_nlmeans_colored_wrong_channels() {
        let img = make_test_image(8, 8, 0.5);
        let result = fast_nl_means_denoising_colored(&img, 0.1f32, 0.1f32, 5, 11);
        assert!(result.is_err());
    }

    // --- Gaussian denoising tests ---

    #[test]
    fn test_gaussian_preserves_uniform() {
        let img = make_test_image(10, 10, 0.7);
        let result = denoise_gaussian(&img, 2.0f32).unwrap();
        let data = result.as_slice().unwrap();
        for &v in data {
            assert!(
                (v - 0.7).abs() < 1e-4,
                "Uniform image should be preserved by Gaussian, got {}",
                v
            );
        }
    }

    #[test]
    fn test_gaussian_smoothing_effect() {
        let h = 16;
        let w = 16;
        let base = 0.5f32;

        let noise_pattern = [0.15, -0.15, 0.1, -0.1, 0.12, -0.12];
        let noisy = make_noisy_image(h, w, base, &noise_pattern);
        let clean = make_test_image(h, w, base);

        let smoothed = denoise_gaussian(&noisy, 1.5f32).unwrap();

        let clean_data = clean.as_slice().unwrap();
        let noisy_data = noisy.as_slice().unwrap();
        let smoothed_data = smoothed.as_slice().unwrap();

        let n = h * w;
        let mse_noisy: f64 = (0..n)
            .map(|i| {
                let d = (noisy_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let mse_smoothed: f64 = (0..n)
            .map(|i| {
                let d = (smoothed_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        assert!(
            mse_smoothed < mse_noisy,
            "Gaussian smoothing should reduce MSE: noisy={:.6}, smoothed={:.6}",
            mse_noisy,
            mse_smoothed
        );
    }

    #[test]
    fn test_gaussian_zero_sigma_returns_original() {
        let img = make_test_image(8, 8, 0.3);
        let result = denoise_gaussian(&img, 0.0f32).unwrap();
        let a = img.as_slice().unwrap();
        let b = result.as_slice().unwrap();
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gaussian_multichannel() {
        let n = 10 * 10;
        let mut data = vec![0.0f32; 3 * n];
        for c in 0..3 {
            for i in 0..n {
                data[c * n + i] = (c as f32 + 1.0) * 0.2;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(3, 10, 10)).unwrap();
        let result = denoise_gaussian(&img, 1.0f32).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, 10, 10));

        let out = result.as_slice().unwrap();
        // Each channel should stay near its uniform value.
        for c in 0..3 {
            let expected = (c as f32 + 1.0) * 0.2;
            for i in 0..n {
                assert!(
                    (out[c * n + i] - expected).abs() < 1e-3,
                    "Channel {} pixel {} should be ~{}, got {}",
                    c,
                    i,
                    expected,
                    out[c * n + i]
                );
            }
        }
    }

    #[test]
    fn test_gaussian_psnr_improvement() {
        // Verify PSNR improves after denoising.
        let h = 20;
        let w = 20;
        let base = 0.5f32;
        let noise_pattern = [0.2, -0.2, 0.15, -0.15, 0.1, -0.1, 0.18, -0.18];
        let noisy = make_noisy_image(h, w, base, &noise_pattern);
        let clean = make_test_image(h, w, base);

        let denoised = denoise_gaussian(&noisy, 2.0f32).unwrap();

        let clean_data = clean.as_slice().unwrap();
        let noisy_data = noisy.as_slice().unwrap();
        let denoised_data = denoised.as_slice().unwrap();

        let n = h * w;
        let mse_noisy: f64 = (0..n)
            .map(|i| {
                let d = (noisy_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let mse_denoised: f64 = (0..n)
            .map(|i| {
                let d = (denoised_data[i] - clean_data[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let psnr_noisy = 10.0 * (1.0 / mse_noisy).log10();
        let psnr_denoised = 10.0 * (1.0 / mse_denoised).log10();

        assert!(
            psnr_denoised > psnr_noisy,
            "PSNR should improve: noisy={:.2}dB, denoised={:.2}dB",
            psnr_noisy,
            psnr_denoised
        );
    }
}
