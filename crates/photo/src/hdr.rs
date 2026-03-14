//! HDR processing and tone mapping
//!
//! Provides algorithms for merging multiple exposures into HDR images
//! and mapping HDR images back to displayable LDR range.
//!
//! # Algorithms
//!
//! - **Debevec merge**: Recovers camera response function and merges exposures into HDR radiance map.
//! - **Mertens fusion**: Exposure fusion without HDR — weights and blends LDR images directly.
//! - **Reinhard tone mapping**: Simple global operator `L / (1 + L)` with gamma correction.
//! - **Drago tone mapping**: Adaptive logarithmic mapping for natural appearance.
//! - **Mantiuk tone mapping**: Gradient-domain compression for local contrast preservation.

use cv_core::float::Float;
use cv_core::tensor::{CpuTensor, TensorShape};
use cv_core::Result;

// ---------------------------------------------------------------------------
// HDR Merge
// ---------------------------------------------------------------------------

/// Hat-shaped weighting function for pixel values in [0, 1].
///
/// Gives highest weight to mid-range values and low weight to near-black or
/// near-white pixels that are likely clipped.
#[inline]
fn hat_weight(z: f64) -> f64 {
    if z <= 0.5 {
        z + 0.01
    } else {
        1.01 - z
    }
}

/// Merge multiple exposures into an HDR image using Debevec's method.
///
/// Recovers the camera response function from a set of differently-exposed
/// photographs and uses it to compute an HDR radiance map.
///
/// # Arguments
/// * `images` - Slice of exposure images (all must have the same shape, CHW layout, values in [0,1]).
/// * `exposure_times` - Exposure time in seconds for each image.
/// * `samples` - Number of pixel locations to sample for response curve fitting (256 is typical).
///
/// # Returns
/// An HDR radiance map as a `CpuTensor<f64>` with the same spatial dimensions and channels.
#[allow(clippy::needless_range_loop)]
pub fn merge_debevec<T: Float + Default + 'static>(
    images: &[CpuTensor<T>],
    exposure_times: &[f64],
    samples: usize,
) -> Result<CpuTensor<f64>> {
    if images.is_empty() {
        return Err(cv_core::Error::InvalidInput(
            "At least one image is required".into(),
        ));
    }
    if images.len() != exposure_times.len() {
        return Err(cv_core::Error::InvalidInput(
            "Number of images must match number of exposure times".into(),
        ));
    }
    if images.len() < 2 {
        return Err(cv_core::Error::InvalidInput(
            "At least two exposures are required for Debevec merge".into(),
        ));
    }

    let (channels, height, width) = images[0].shape.chw();
    let n_pixels = height * width;

    // Validate all images have the same shape.
    for (i, img) in images.iter().enumerate() {
        if img.shape != images[0].shape {
            return Err(cv_core::Error::DimensionMismatch(format!(
                "Image {} shape {:?} differs from image 0 shape {:?}",
                i, img.shape, images[0].shape
            )));
        }
    }

    let n_exposures = images.len();
    let n_samples = samples.min(n_pixels);

    // Sample pixel indices evenly across the image.
    let step = if n_pixels > n_samples {
        n_pixels / n_samples
    } else {
        1
    };
    let sample_indices: Vec<usize> = (0..n_pixels).step_by(step).take(n_samples).collect();

    // Convert all image data to f64 for precision.
    let img_data: Vec<Vec<f64>> = images
        .iter()
        .map(|img| {
            img.as_slice()
                .unwrap()
                .iter()
                .map(|v| Float::to_f64(*v))
                .collect()
        })
        .collect();

    // For each channel, recover the response curve via a simplified Debevec approach,
    // then compute the HDR radiance.
    //
    // We quantize pixel values to 256 levels and solve for g(z) = ln(E) + ln(dt).
    // Using a least-squares fit with smoothness regularization.

    let n_levels: usize = 256;
    let lambda = 50.0; // smoothness weight

    let mut hdr_data = vec![0.0f64; channels * n_pixels];

    for c in 0..channels {
        let ch_offset = c * n_pixels;

        // Build the overdetermined system: A*x = b
        // x = [g(0)..g(255), lnE(sample_0)..lnE(sample_{n-1})]
        // Dimensions: rows = n_samples * n_exposures + (n_levels - 2) + 1, cols = n_levels + n_samples.
        let n_data_rows = n_samples * n_exposures;
        let n_smooth_rows = n_levels.saturating_sub(2);
        let n_rows = n_data_rows + n_smooth_rows + 1; // +1 for g(128) = 0 constraint
        let n_cols = n_levels + n_samples;

        // Use flat arrays for the system (dense solve).
        let mut a_flat = vec![0.0f64; n_rows * n_cols];
        let mut b = vec![0.0f64; n_rows];

        let mut row = 0;

        // Data equations: w(z) * [g(z) - lnE_j - ln(dt_p)] = 0
        for (j, &si) in sample_indices.iter().enumerate() {
            for p in 0..n_exposures {
                let z_val = img_data[p][ch_offset + si].clamp(0.0, 1.0);
                let z_idx = (z_val * (n_levels - 1) as f64).round() as usize;
                let z_idx = z_idx.min(n_levels - 1);
                let w = hat_weight(z_val);

                a_flat[row * n_cols + z_idx] = w;
                a_flat[row * n_cols + n_levels + j] = -w;
                b[row] = w * exposure_times[p].ln();
                row += 1;
            }
        }

        // Fix the curve: g(128) = 0
        a_flat[row * n_cols + 128.min(n_levels - 1)] = 1.0;
        b[row] = 0.0;
        row += 1;

        // Smoothness equations: lambda * w(z) * [g(z-1) - 2*g(z) + g(z+1)] = 0
        for z in 1..n_levels.saturating_sub(1) {
            let z_norm = z as f64 / (n_levels - 1) as f64;
            let w = lambda * hat_weight(z_norm);
            a_flat[row * n_cols + z - 1] = w;
            a_flat[row * n_cols + z] = -2.0 * w;
            a_flat[row * n_cols + z + 1] = w;
            b[row] = 0.0;
            row += 1;
        }

        // Solve via normal equations: A^T A x = A^T b
        let actual_rows = row;
        let mut ata = vec![0.0f64; n_cols * n_cols];
        let mut atb = vec![0.0f64; n_cols];

        for r in 0..actual_rows {
            let row_start = r * n_cols;
            for i in 0..n_cols {
                let ai = a_flat[row_start + i];
                if ai == 0.0 {
                    continue;
                }
                atb[i] += ai * b[r];
                for j in i..n_cols {
                    let aj = a_flat[row_start + j];
                    if aj != 0.0 {
                        ata[i * n_cols + j] += ai * aj;
                        if i != j {
                            ata[j * n_cols + i] += ai * aj;
                        }
                    }
                }
            }
        }

        // Add Tikhonov regularization for stability.
        for i in 0..n_cols {
            ata[i * n_cols + i] += 1e-6;
        }

        // Solve with Cholesky (or fall back to simple iterative Gauss-Seidel).
        let x = gauss_seidel_solve(&ata, &atb, n_cols, 200);

        // Extract response curve g(z).
        let g: Vec<f64> = x[..n_levels].to_vec();

        // Compute HDR radiance for every pixel.
        for pixel in 0..n_pixels {
            let mut sum_w = 0.0;
            let mut sum_log_e = 0.0;

            for p in 0..n_exposures {
                let z_val = img_data[p][ch_offset + pixel].clamp(0.0, 1.0);
                let z_idx = (z_val * (n_levels - 1) as f64)
                    .round()
                    .min((n_levels - 1) as f64) as usize;
                let w = hat_weight(z_val);

                if w > 1e-10 {
                    let log_e = g[z_idx] - exposure_times[p].ln();
                    sum_log_e += w * log_e;
                    sum_w += w;
                }
            }

            hdr_data[ch_offset + pixel] = if sum_w > 1e-10 {
                (sum_log_e / sum_w).exp()
            } else {
                0.0
            };
        }
    }

    CpuTensor::<f64>::from_vec(hdr_data, TensorShape::new(channels, height, width))
}

/// Simple Gauss-Seidel iterative solver for Ax = b.
fn gauss_seidel_solve(a: &[f64], b: &[f64], n: usize, iterations: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for _ in 0..iterations {
        for i in 0..n {
            let diag = a[i * n + i];
            if diag.abs() < 1e-15 {
                continue;
            }
            let mut sum = b[i];
            for j in 0..n {
                if j != i {
                    sum -= a[i * n + j] * x[j];
                }
            }
            x[i] = sum / diag;
        }
    }
    x
}

/// Merge multiple exposures using Mertens exposure fusion.
///
/// Blends LDR images directly without computing an HDR radiance map.
/// Each pixel's weight is the product of contrast, saturation, and
/// well-exposedness quality measures.
///
/// # Arguments
/// * `images` - Slice of LDR images (all same shape, CHW layout, values in [0,1]).
///
/// # Returns
/// A fused LDR image with the same shape and type.
#[allow(clippy::needless_range_loop)]
pub fn merge_mertens<T: Float + Default + 'static>(
    images: &[CpuTensor<T>],
) -> Result<CpuTensor<T>> {
    if images.is_empty() {
        return Err(cv_core::Error::InvalidInput(
            "At least one image is required".into(),
        ));
    }
    if images.len() == 1 {
        return Ok(images[0].clone());
    }

    let (channels, height, width) = images[0].shape.chw();
    let n_pixels = height * width;

    for (i, img) in images.iter().enumerate() {
        if img.shape != images[0].shape {
            return Err(cv_core::Error::DimensionMismatch(format!(
                "Image {} shape {:?} differs from image 0 shape {:?}",
                i, img.shape, images[0].shape
            )));
        }
    }

    let n_images = images.len();

    // Convert to f64 for computation.
    let img_data: Vec<Vec<f64>> = images
        .iter()
        .map(|img| {
            img.as_slice()
                .unwrap()
                .iter()
                .map(|v| Float::to_f64(*v))
                .collect()
        })
        .collect();

    // Compute quality weights for each image at each pixel.
    let mut weights = vec![vec![1.0f64; n_pixels]; n_images];

    for (k, img) in img_data.iter().enumerate() {
        for pixel in 0..n_pixels {
            // --- Contrast: Laplacian magnitude on grayscale ---
            let y = pixel / width;
            let x = pixel % width;

            // Compute grayscale value.
            let gray = if channels == 1 {
                img[pixel]
            } else {
                let mut sum = 0.0;
                for c in 0..channels {
                    sum += img[c * n_pixels + pixel];
                }
                sum / channels as f64
            };

            // Simple Laplacian (4-connected).
            let mut laplacian = 0.0;
            if y > 0 && y + 1 < height && x > 0 && x + 1 < width {
                let gray_at = |py: usize, px: usize| -> f64 {
                    if channels == 1 {
                        img[py * width + px]
                    } else {
                        let mut s = 0.0;
                        for c in 0..channels {
                            s += img[c * n_pixels + py * width + px];
                        }
                        s / channels as f64
                    }
                };
                laplacian =
                    (gray_at(y - 1, x) + gray_at(y + 1, x) + gray_at(y, x - 1) + gray_at(y, x + 1)
                        - 4.0 * gray)
                        .abs();
            }
            let contrast = laplacian + 1e-6;

            // --- Saturation: std dev across channels ---
            let saturation = if channels >= 3 {
                let mean = gray;
                let mut var = 0.0;
                for c in 0..channels {
                    let diff = img[c * n_pixels + pixel] - mean;
                    var += diff * diff;
                }
                (var / channels as f64).sqrt() + 1e-6
            } else {
                1.0
            };

            // --- Well-exposedness: Gaussian centered at 0.5 ---
            let sigma_exp = 0.2;
            let mut exposedness = 1.0;
            for c in 0..channels {
                let v = img[c * n_pixels + pixel];
                let diff = v - 0.5;
                exposedness *= (-0.5 * diff * diff / (sigma_exp * sigma_exp)).exp();
            }
            exposedness += 1e-6;

            weights[k][pixel] = contrast * saturation * exposedness;
        }
    }

    // Normalize weights across images at each pixel.
    for pixel in 0..n_pixels {
        let sum: f64 = weights.iter().map(|w| w[pixel]).sum();
        if sum > 1e-10 {
            for k in 0..n_images {
                weights[k][pixel] /= sum;
            }
        } else {
            // Equal weights as fallback.
            let eq = 1.0 / n_images as f64;
            for k in 0..n_images {
                weights[k][pixel] = eq;
            }
        }
    }

    // Weighted blend.
    let mut out_data = vec![T::ZERO; channels * n_pixels];
    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for pixel in 0..n_pixels {
            let mut val = 0.0;
            for k in 0..n_images {
                val += weights[k][pixel] * img_data[k][ch_offset + pixel];
            }
            out_data[ch_offset + pixel] = T::from_f64(val.clamp(0.0, 1.0));
        }
    }

    CpuTensor::<T>::from_vec(out_data, TensorShape::new(channels, height, width))
}

// ---------------------------------------------------------------------------
// Tone Mapping
// ---------------------------------------------------------------------------

/// Compute luminance from an HDR image (CHW layout).
/// Returns a per-pixel luminance vector.
fn compute_luminance(data: &[f64], channels: usize, n_pixels: usize) -> Vec<f64> {
    let mut lum = vec![0.0f64; n_pixels];
    if channels >= 3 {
        // Rec. 709 luminance.
        for pixel in 0..n_pixels {
            lum[pixel] = 0.2126 * data[pixel]
                + 0.7152 * data[n_pixels + pixel]
                + 0.0722 * data[2 * n_pixels + pixel];
        }
    } else {
        lum.copy_from_slice(&data[..n_pixels]);
    }
    // Clamp to positive.
    for v in lum.iter_mut() {
        if *v < 1e-10 {
            *v = 1e-10;
        }
    }
    lum
}

/// Reinhard global tone mapping operator.
///
/// Applies `L_out = L / (1 + L)` followed by gamma correction.
///
/// # Arguments
/// * `hdr` - HDR radiance map (CHW layout, f64).
/// * `gamma` - Gamma correction exponent (e.g. 2.2). Values > 0.
///
/// # Returns
/// An LDR image in [0, 1] as `CpuTensor<f32>`.
pub fn tonemap_reinhard(hdr: &CpuTensor<f64>, gamma: f64) -> Result<CpuTensor<f32>> {
    let (channels, height, width) = hdr.shape.chw();
    let n_pixels = height * width;
    let data = hdr.as_slice()?;

    let lum = compute_luminance(data, channels, n_pixels);

    // Compute log-average luminance for key scaling.
    let epsilon = 1e-10;
    let l_avg = {
        let sum_log: f64 = lum.iter().map(|&l| (l + epsilon).ln()).sum();
        (sum_log / n_pixels as f64).exp()
    };

    let inv_gamma = 1.0 / gamma.max(0.01);
    let mut out = vec![0.0f32; channels * n_pixels];

    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for pixel in 0..n_pixels {
            let l_in = lum[pixel];
            // Scale luminance by key value.
            let l_scaled = 0.18 / l_avg * l_in;
            let l_out = l_scaled / (1.0 + l_scaled);

            // Scale the color channel proportionally.
            let ratio = if l_in > 1e-10 { l_out / l_in } else { 0.0 };
            let mapped = (data[ch_offset + pixel].max(0.0) * ratio).powf(inv_gamma);
            out[ch_offset + pixel] = mapped.clamp(0.0, 1.0) as f32;
        }
    }

    CpuTensor::<f32>::from_vec(out, TensorShape::new(channels, height, width))
}

/// Drago logarithmic tone mapping operator.
///
/// Uses adaptive logarithmic mapping to compress the dynamic range while
/// preserving detail in both shadows and highlights.
///
/// # Arguments
/// * `hdr` - HDR radiance map (CHW layout, f64).
/// * `gamma` - Gamma correction exponent (e.g. 2.2).
/// * `saturation` - Color saturation adjustment (1.0 = preserve, <1 = desaturate).
///
/// # Returns
/// An LDR image in [0, 1] as `CpuTensor<f32>`.
pub fn tonemap_drago(hdr: &CpuTensor<f64>, gamma: f64, saturation: f64) -> Result<CpuTensor<f32>> {
    let (channels, height, width) = hdr.shape.chw();
    let n_pixels = height * width;
    let data = hdr.as_slice()?;

    let lum = compute_luminance(data, channels, n_pixels);

    // Log-average luminance.
    let log_avg: f64 = {
        let sum: f64 = lum.iter().map(|&l| l.ln()).sum();
        (sum / n_pixels as f64).exp()
    };

    let bias = 0.85; // Drago bias parameter
    let bias_log = bias.ln() / (0.5f64).ln();
    let inv_gamma = 1.0 / gamma.max(0.01);

    // Drago mapping: L_out = log(1 + L_scaled) / log(1 + L_max_scaled)
    let l_max = lum.iter().cloned().fold(0.0f64, f64::max);
    let l_max_scaled = l_max / log_avg;

    let mut out = vec![0.0f32; channels * n_pixels];

    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for pixel in 0..n_pixels {
            let l_in = lum[pixel];
            let l_scaled = l_in / log_avg;

            // Adaptive base for logarithm.
            let base = (2.0 + 8.0 * (l_scaled / l_max_scaled).powf(bias_log)).clamp(2.0, 10.0);
            let l_out = l_scaled.log(base) / l_max_scaled.log(base).max(1e-10);
            let l_out = l_out.max(0.0);

            // Apply saturation.
            let channel_val = data[ch_offset + pixel].max(0.0);
            let ratio = if l_in > 1e-10 {
                (channel_val / l_in).powf(saturation) * l_out
            } else {
                0.0
            };

            let mapped = ratio.powf(inv_gamma);
            out[ch_offset + pixel] = mapped.clamp(0.0, 1.0) as f32;
        }
    }

    // Normalize to [0, 1] range.
    let max_val = out.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 1e-10 {
        for v in out.iter_mut() {
            *v /= max_val;
        }
    }

    CpuTensor::<f32>::from_vec(out, TensorShape::new(channels, height, width))
}

/// Mantiuk tone mapping operator (gradient-domain compression).
///
/// Compresses the dynamic range by working in the gradient domain,
/// attenuating large gradients while preserving small (local contrast) gradients.
///
/// # Arguments
/// * `hdr` - HDR radiance map (CHW layout, f64).
/// * `gamma` - Gamma correction exponent.
/// * `scale` - Gradient attenuation scale (0.0 to 1.0; lower = more compression).
///
/// # Returns
/// An LDR image in [0, 1] as `CpuTensor<f32>`.
pub fn tonemap_mantiuk(hdr: &CpuTensor<f64>, gamma: f64, scale: f64) -> Result<CpuTensor<f32>> {
    let (channels, height, width) = hdr.shape.chw();
    let n_pixels = height * width;
    let data = hdr.as_slice()?;

    if height < 2 || width < 2 {
        return Err(cv_core::Error::InvalidInput(
            "Image must be at least 2x2 for Mantiuk tone mapping".into(),
        ));
    }

    let lum = compute_luminance(data, channels, n_pixels);

    // Work in log-luminance domain.
    let log_lum: Vec<f64> = lum.iter().map(|&l| l.ln()).collect();

    // Compute gradients.
    let mut gx = vec![0.0f64; n_pixels];
    let mut gy = vec![0.0f64; n_pixels];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x + 1 < width {
                gx[idx] = log_lum[idx + 1] - log_lum[idx];
            }
            if y + 1 < height {
                gy[idx] = log_lum[idx + width] - log_lum[idx];
            }
        }
    }

    // Attenuate gradients: compress large gradients, keep small ones.
    let attenuation_scale = scale.clamp(0.01, 1.0);
    for idx in 0..n_pixels {
        let mag = (gx[idx] * gx[idx] + gy[idx] * gy[idx]).sqrt();
        if mag > 1e-10 {
            let factor = attenuation_scale / (1.0 + mag / attenuation_scale);
            gx[idx] *= factor;
            gy[idx] *= factor;
        }
    }

    // Reconstruct log-luminance from attenuated gradients using Poisson solver
    // (Gauss-Seidel iteration on the divergence).
    let mut div = vec![0.0f64; n_pixels];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut d = 0.0;
            // d(gx)/dx
            d += gx[idx];
            if x > 0 {
                d -= gx[idx - 1];
            }
            // d(gy)/dy
            d += gy[idx];
            if y > 0 {
                d -= gy[idx - width];
            }
            div[idx] = d;
        }
    }

    // Gauss-Seidel Poisson solve.
    let mut recon = log_lum.clone();
    for _ in 0..100 {
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let mut sum = div[idx];
                let mut count = 0.0;

                if x > 0 {
                    sum += recon[idx - 1];
                    count += 1.0;
                }
                if x + 1 < width {
                    sum += recon[idx + 1];
                    count += 1.0;
                }
                if y > 0 {
                    sum += recon[idx - width];
                    count += 1.0;
                }
                if y + 1 < height {
                    sum += recon[idx + width];
                    count += 1.0;
                }

                if count > 0.0 {
                    recon[idx] = sum / count;
                }
            }
        }
    }

    // Convert back from log domain.
    let new_lum: Vec<f64> = recon.iter().map(|&v| v.exp()).collect();

    // Normalize to [0, 1].
    let max_lum = new_lum.iter().cloned().fold(0.0f64, f64::max);
    let inv_gamma = 1.0 / gamma.max(0.01);

    let mut out = vec![0.0f32; channels * n_pixels];

    for c in 0..channels {
        let ch_offset = c * n_pixels;
        for pixel in 0..n_pixels {
            let l_old = lum[pixel];
            let l_new = new_lum[pixel] / max_lum.max(1e-10);
            let ratio = if l_old > 1e-10 { l_new / l_old } else { 0.0 };

            let mapped = (data[ch_offset + pixel].max(0.0) * ratio).powf(inv_gamma);
            out[ch_offset + pixel] = mapped.clamp(0.0, 1.0) as f32;
        }
    }

    CpuTensor::<f32>::from_vec(out, TensorShape::new(channels, height, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::tensor::TensorShape;

    fn make_uniform_image(channels: usize, h: usize, w: usize, value: f32) -> CpuTensor<f32> {
        let data = vec![value; channels * h * w];
        CpuTensor::<f32>::from_vec(data, TensorShape::new(channels, h, w)).unwrap()
    }

    fn make_gradient_image(channels: usize, h: usize, w: usize, scale: f32) -> CpuTensor<f32> {
        let n = h * w;
        let mut data = vec![0.0f32; channels * n];
        for c in 0..channels {
            for y in 0..h {
                for x in 0..w {
                    let v = (x as f32 / (w - 1).max(1) as f32) * scale;
                    data[c * n + y * w + x] = v.clamp(0.0, 1.0);
                }
            }
        }
        CpuTensor::<f32>::from_vec(data, TensorShape::new(channels, h, w)).unwrap()
    }

    // --- Mertens fusion tests ---

    #[test]
    fn test_merge_mertens_three_exposures() {
        let h = 16;
        let w = 16;
        let dark = make_uniform_image(3, h, w, 0.15);
        let normal = make_uniform_image(3, h, w, 0.5);
        let bright = make_uniform_image(3, h, w, 0.85);

        let result = merge_mertens(&[dark, normal, bright]).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, h, w));

        let data = result.as_slice().unwrap();
        for &v in data {
            assert!(v >= 0.0 && v <= 1.0, "Output out of [0,1]: {}", v);
        }
    }

    #[test]
    fn test_merge_mertens_preserves_single_image() {
        let img = make_gradient_image(1, 8, 8, 1.0);
        let result = merge_mertens(&[img.clone()]).unwrap();
        let a = img.as_slice().unwrap();
        let b = result.as_slice().unwrap();
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-6);
        }
    }

    #[test]
    fn test_merge_mertens_gradient_images() {
        let h = 10;
        let w = 20;
        let dark = make_gradient_image(3, h, w, 0.3);
        let normal = make_gradient_image(3, h, w, 0.6);
        let bright = make_gradient_image(3, h, w, 1.0);

        let result = merge_mertens(&[dark, normal, bright]).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, h, w));

        let data = result.as_slice().unwrap();
        for &v in data {
            assert!(v >= 0.0 && v <= 1.0, "Output out of [0,1]: {}", v);
        }
    }

    #[test]
    fn test_merge_mertens_empty() {
        let result = merge_mertens::<f32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_mertens_shape_mismatch() {
        let img1 = make_uniform_image(1, 8, 8, 0.5);
        let img2 = make_uniform_image(1, 4, 4, 0.5);
        let result = merge_mertens(&[img1, img2]);
        assert!(result.is_err());
    }

    // --- Debevec merge tests ---

    #[test]
    fn test_merge_debevec_synthetic() {
        let h = 8;
        let w = 8;
        // Simulate 3 exposures of a scene with known radiance.
        let exposures = [0.01, 0.1, 1.0];
        let base_radiance = 5.0; // constant scene radiance

        let images: Vec<CpuTensor<f32>> = exposures
            .iter()
            .map(|&dt| {
                let pixel_val = (base_radiance * dt).min(1.0) as f32;
                make_uniform_image(1, h, w, pixel_val)
            })
            .collect();

        let result = merge_debevec(&images, &exposures, 32).unwrap();
        assert_eq!(result.shape, TensorShape::new(1, h, w));

        let data = result.as_slice().unwrap();
        // All pixels should be positive (HDR radiance).
        for &v in data {
            assert!(v > 0.0, "HDR value should be positive, got {}", v);
            assert!(v.is_finite(), "HDR value should be finite, got {}", v);
        }
    }

    #[test]
    fn test_merge_debevec_needs_two_exposures() {
        let img = make_uniform_image(1, 4, 4, 0.5);
        let result = merge_debevec(&[img], &[1.0], 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_debevec_mismatched_counts() {
        let img1 = make_uniform_image(1, 4, 4, 0.3);
        let img2 = make_uniform_image(1, 4, 4, 0.7);
        let result = merge_debevec(&[img1, img2], &[0.1, 0.5, 1.0], 32);
        assert!(result.is_err());
    }

    // --- Reinhard tone mapping tests ---

    #[test]
    fn test_tonemap_reinhard_output_range() {
        // Create an HDR image with values in [0.001, 100].
        let h = 10;
        let w = 10;
        let n = h * w;
        let mut data = vec![0.0f64; 3 * n];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data[c * n + y * w + x] = 0.001 + 99.999 * (x as f64 / (w - 1) as f64);
                }
            }
        }
        let hdr = CpuTensor::<f64>::from_vec(data, TensorShape::new(3, h, w)).unwrap();

        let result = tonemap_reinhard(&hdr, 2.2).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, h, w));

        let out = result.as_slice().unwrap();
        for &v in out {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Reinhard output should be in [0,1], got {}",
                v
            );
        }
    }

    #[test]
    fn test_tonemap_reinhard_monotonic() {
        // Brighter input should produce brighter (or equal) output.
        let h = 1;
        let w = 10;
        let mut data = vec![0.0f64; w];
        for x in 0..w {
            data[x] = (x + 1) as f64;
        }
        let hdr = CpuTensor::<f64>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        let result = tonemap_reinhard(&hdr, 1.0).unwrap();
        let out = result.as_slice().unwrap();
        for x in 1..w {
            assert!(
                out[x] >= out[x - 1] - 1e-6,
                "Reinhard should be monotonic: out[{}]={} < out[{}]={}",
                x,
                out[x],
                x - 1,
                out[x - 1]
            );
        }
    }

    // --- Drago tone mapping tests ---

    #[test]
    fn test_tonemap_drago_output_range() {
        let h = 8;
        let w = 8;
        let n = h * w;
        let mut data = vec![0.0f64; 3 * n];
        for c in 0..3 {
            for i in 0..n {
                data[c * n + i] = 0.01 + 50.0 * (i as f64 / n as f64);
            }
        }
        let hdr = CpuTensor::<f64>::from_vec(data, TensorShape::new(3, h, w)).unwrap();

        let result = tonemap_drago(&hdr, 2.2, 1.0).unwrap();
        assert_eq!(result.shape, TensorShape::new(3, h, w));

        let out = result.as_slice().unwrap();
        for &v in out {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Drago output should be in [0,1], got {}",
                v
            );
        }
    }

    // --- Mantiuk tone mapping tests ---

    #[test]
    fn test_tonemap_mantiuk_output_range() {
        let h = 8;
        let w = 8;
        let n = h * w;
        let mut data = vec![0.0f64; n];
        for i in 0..n {
            data[i] = 0.01 + 20.0 * (i as f64 / n as f64);
        }
        let hdr = CpuTensor::<f64>::from_vec(data, TensorShape::new(1, h, w)).unwrap();

        let result = tonemap_mantiuk(&hdr, 2.2, 0.5).unwrap();
        assert_eq!(result.shape, TensorShape::new(1, h, w));

        let out = result.as_slice().unwrap();
        for &v in out {
            assert!(
                v >= 0.0 && v <= 1.0,
                "Mantiuk output should be in [0,1], got {}",
                v
            );
        }
    }

    #[test]
    fn test_tonemap_mantiuk_too_small() {
        let data = vec![1.0f64];
        let hdr = CpuTensor::<f64>::from_vec(data, TensorShape::new(1, 1, 1)).unwrap();
        let result = tonemap_mantiuk(&hdr, 2.2, 0.5);
        assert!(result.is_err());
    }
}
