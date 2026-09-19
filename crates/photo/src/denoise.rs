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
//! - **BM3D**: Block-Matching and 3D filtering — state-of-the-art denoising using
//!   block matching, 3D DCT transform, and collaborative Wiener filtering.

use cv_core::float::Float;
use cv_core::storage::Storage;
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

// ── BM3D (Block-Matching and 3D Filtering) ──────────────────────────────────

/// BM3D denoising for grayscale images.
///
/// Two-stage algorithm:
/// 1. **Hard thresholding**: Group similar blocks, apply 3D DCT, hard-threshold,
///    inverse transform, and aggregate.
/// 2. **Wiener filtering**: Use stage 1 output as pilot estimate for adaptive
///    Wiener filtering on grouped blocks.
///
/// # Arguments
/// * `image` — Single-channel image (1, H, W), values in [0, 1].
/// * `sigma` — Noise standard deviation (e.g., 25.0/255.0 for sigma=25 on 0-255 scale).
/// * `block_size` — Block size (typically 8).
/// * `search_window` — Search window size (typically 39).
/// * `max_matches` — Maximum similar blocks per group (typically 16).
///
/// # Returns
/// Denoised single-channel image.
pub fn bm3d<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    sigma: T,
    block_size: usize,
    search_window: usize,
    max_matches: usize,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();
    if channels != 1 {
        return Err(cv_core::Error::InvalidInput(
            "BM3D expects single-channel image".into(),
        ));
    }
    if height < block_size || width < block_size {
        return Err(cv_core::Error::InvalidInput(
            "Image too small for block size".into(),
        ));
    }

    let src = image
        .storage
        .as_slice()
        .ok_or_else(|| cv_core::Error::InvalidInput("Cannot access image data".into()))?;

    let bs = block_size;
    let half_sw = search_window / 2;
    let sigma_f64 = sigma.to_f64();
    let threshold = 2.7 * sigma_f64; // hard threshold = lambda * sigma

    // Orthonormal DCT-II basis and reusable scratch (allocated once — a
    // previous revision heap-allocated per candidate block inside the
    // quadruple-nested search loops).
    let dct_b = dct_basis(bs);
    let mut scratch_a = vec![0.0f64; bs * bs];
    let mut scratch_b = vec![0.0f64; bs * bs];
    let mut coefs = vec![0.0f64; bs * bs];
    let mut noisy_coefs = vec![0.0f64; bs * bs];
    let mut pilot_coefs = vec![0.0f64; bs * bs];
    let mut filtered = vec![0.0f64; bs * bs];

    // ── Stage 1: Hard thresholding ──────────────────────────────────────

    let mut estimate = vec![0.0f64; height * width];
    let mut weights = vec![0.0f64; height * width];

    // Step through reference blocks
    let step = (bs / 2).max(1);
    for ry in (0..height - bs + 1).step_by(step) {
        for rx in (0..width - bs + 1).step_by(step) {
            // Extract reference block
            let ref_block = extract_block_f64(src, width, rx, ry, bs);

            // Find similar blocks in search window
            let mut matches: Vec<(usize, usize, f64)> = Vec::new();
            let sy0 = ry.saturating_sub(half_sw);
            let sy1 = (ry + half_sw + 1).min(height - bs + 1);
            let sx0 = rx.saturating_sub(half_sw);
            let sx1 = (rx + half_sw + 1).min(width - bs + 1);

            for my in sy0..sy1 {
                for mx in sx0..sx1 {
                    let cand = extract_block_f64(src, width, mx, my, bs);
                    let dist = block_distance(&ref_block, &cand, bs);
                    if dist < threshold * threshold * (bs * bs) as f64 {
                        matches.push((mx, my, dist));
                    }
                }
            }

            // Sort by distance and keep top matches
            matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            matches.truncate(max_matches);

            if matches.is_empty() {
                matches.push((rx, ry, 0.0));
            }

            // Stack matched blocks into 3D group
            let n_matches = matches.len();
            let mut group: Vec<f64> = Vec::with_capacity(n_matches * bs * bs);
            for &(mx, my, _) in &matches {
                let blk = extract_block_f64(src, width, mx, my, bs);
                group.extend_from_slice(&blk);
            }

            // Apply 2D DCT to each block, hard-threshold coefficients,
            // inverse-transform back (real transform-domain hard thresholding;
            // a previous revision only subtracted the block mean).
            for b in 0..n_matches {
                let offset = b * bs * bs;
                dct_2d(&group[offset..offset + bs * bs], bs, &dct_b, &mut scratch_a, &mut coefs);

                let mut nonzero = 0usize;
                for c in coefs.iter_mut() {
                    if c.abs() > threshold {
                        nonzero += 1;
                    } else {
                        *c = 0.0;
                    }
                }
                idct_2d(&coefs, bs, &dct_b, &mut scratch_a, &mut scratch_b);
                group[offset..offset + bs * bs].copy_from_slice(&scratch_b);
                // Weight based on number of surviving coefficients
                let w = if nonzero > 0 {
                    1.0 / nonzero as f64
                } else {
                    1.0
                };

                // Aggregate back into estimate
                let (mx, my) = (matches[b].0, matches[b].1);
                for by in 0..bs {
                    for bx in 0..bs {
                        let idx = (my + by) * width + (mx + bx);
                        estimate[idx] += group[offset + by * bs + bx] * w;
                        weights[idx] += w;
                    }
                }
            }
        }
    }

    // Normalize stage 1 estimate
    for i in 0..height * width {
        if weights[i] > 0.0 {
            estimate[i] /= weights[i];
        } else {
            estimate[i] = src[i].to_f64();
        }
    }

    // ── Stage 2: Wiener filtering using stage 1 as pilot ────────────────

    let mut final_est = vec![0.0f64; height * width];
    let mut final_w = vec![0.0f64; height * width];

    for ry in (0..height - bs + 1).step_by(step) {
        for rx in (0..width - bs + 1).step_by(step) {
            let ref_pilot = extract_block_f64_raw(&estimate, width, rx, ry, bs);

            let mut matches: Vec<(usize, usize, f64)> = Vec::new();
            let sy0 = ry.saturating_sub(half_sw);
            let sy1 = (ry + half_sw + 1).min(height - bs + 1);
            let sx0 = rx.saturating_sub(half_sw);
            let sx1 = (rx + half_sw + 1).min(width - bs + 1);

            for my in sy0..sy1 {
                for mx in sx0..sx1 {
                    let cand = extract_block_f64_raw(&estimate, width, mx, my, bs);
                    let dist = block_distance(&ref_pilot, &cand, bs);
                    if dist < threshold * threshold * (bs * bs) as f64 {
                        matches.push((mx, my, dist));
                    }
                }
            }
            matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            matches.truncate(max_matches);
            if matches.is_empty() {
                matches.push((rx, ry, 0.0));
            }

            let n_matches = matches.len();

            // Empirical Wiener filtering in the DCT domain: shrink each noisy
            // coefficient by pilot-energy/(pilot-energy + sigma^2).
            for &(mx, my, _) in &matches {
                let noisy_blk = extract_block_f64(src, width, mx, my, bs);
                let pilot_blk = extract_block_f64_raw(&estimate, width, mx, my, bs);

                dct_2d(&noisy_blk, bs, &dct_b, &mut scratch_a, &mut noisy_coefs);
                dct_2d(&pilot_blk, bs, &dct_b, &mut scratch_a, &mut pilot_coefs);

                for k in 0..bs * bs {
                    let energy = pilot_coefs[k] * pilot_coefs[k];
                    let w_shrink = energy / (energy + sigma_f64 * sigma_f64);
                    coefs[k] = noisy_coefs[k] * w_shrink;
                }
                idct_2d(&coefs, bs, &dct_b, &mut scratch_a, &mut filtered);

                let w = 1.0 / (1.0 + 1.0 / (n_matches as f64));
                for by in 0..bs {
                    for bx in 0..bs {
                        let idx = (my + by) * width + (mx + bx);
                        final_est[idx] += filtered[by * bs + bx] * w;
                        final_w[idx] += w;
                    }
                }
            }
        }
    }

    // Normalize
    let mut output = vec![T::zero(); height * width];
    for i in 0..height * width {
        let v = if final_w[i] > 0.0 {
            final_est[i] / final_w[i]
        } else {
            estimate[i]
        };
        output[i] = T::from_f64(v.clamp(0.0, 1.0));
    }

    CpuTensor::from_vec(output, TensorShape::new(1, height, width))
}

fn extract_block_f64<T: Float>(src: &[T], width: usize, x: usize, y: usize, bs: usize) -> Vec<f64> {
    let mut block = Vec::with_capacity(bs * bs);
    for by in 0..bs {
        for bx in 0..bs {
            block.push(src[(y + by) * width + (x + bx)].to_f64());
        }
    }
    block
}

/// Orthonormal DCT-II basis rows: row k holds basis_k(x) for x in 0..bs.
fn dct_basis(bs: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; bs * bs];
    let nf = bs as f64;
    for (k, row) in b.chunks_mut(bs).enumerate() {
        let ck = if k == 0 {
            (1.0 / nf).sqrt()
        } else {
            (2.0 / nf).sqrt()
        };
        for (x, v) in row.iter_mut().enumerate() {
            *v = ck
                * ((k as f64)
                    * std::f64::consts::PI
                    * (2.0 * x as f64 + 1.0)
                    / (2.0 * nf))
                    .cos();
        }
    }
    b
}

/// Forward 2D DCT via separable passes: out = B · block · Bᵀ.
/// `tmp` and `out` must be at least bs*bs long.
fn dct_2d(block: &[f64], bs: usize, basis: &[f64], tmp: &mut [f64], out: &mut [f64]) {
    // Pass 1 over rows: tmp[r][c] = Σ_x block[r][x] · B[c][x]
    for r in 0..bs {
        for c in 0..bs {
            let mut s = 0.0;
            for x in 0..bs {
                s += block[r * bs + x] * basis[c * bs + x];
            }
            tmp[r * bs + c] = s;
        }
    }
    // Pass 2 over columns: out[r][c] = Σ_y B[r][y] · tmp[y][c]
    for r in 0..bs {
        for c in 0..bs {
            let mut s = 0.0;
            for y in 0..bs {
                s += basis[r * bs + y] * tmp[y * bs + c];
            }
            out[r * bs + c] = s;
        }
    }
}

/// Inverse 2D DCT: out = Bᵀ · coef · B.
fn idct_2d(coef: &[f64], bs: usize, basis: &[f64], tmp: &mut [f64], out: &mut [f64]) {
    // Pass 1 over rows of coef: tmp[y][x] = Σ_c coef[y][c] · B[c][x]
    for y in 0..bs {
        for x in 0..bs {
            let mut s = 0.0;
            for c in 0..bs {
                s += coef[y * bs + c] * basis[c * bs + x];
            }
            tmp[y * bs + x] = s;
        }
    }
    // Pass 2 over columns: out[r][x] = Σ_v Bᵀ[r][v] · tmp[v][x]
    for r in 0..bs {
        for x in 0..bs {
            let mut s = 0.0;
            for v in 0..bs {
                s += basis[v * bs + r] * tmp[v * bs + x];
            }
            out[r * bs + x] = s;
        }
    }
}

fn extract_block_f64_raw(src: &[f64], width: usize, x: usize, y: usize, bs: usize) -> Vec<f64> {
    let mut block = Vec::with_capacity(bs * bs);
    for by in 0..bs {
        for bx in 0..bs {
            block.push(src[(y + by) * width + (x + bx)]);
        }
    }
    block
}

fn block_distance(a: &[f64], b: &[f64], _bs: usize) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
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

#[cfg(test)]
mod bm3d_dct_tests {
    use super::*;

    #[test]
    fn test_dct_round_trip() {
        // Random-ish block must survive dct→idct exactly (orthonormal).
        let bs = 8;
        let basis = dct_basis(bs);
        let mut block = vec![0.0; bs * bs];
        for i in 0..bs * bs {
            block[i] = ((i * 37) % 100) as f64 / 100.0 - 0.5;
        }
        let mut tmp = vec![0.0; bs * bs];
        let mut coefs = vec![0.0; bs * bs];
        let mut rec = vec![0.0; bs * bs];
        dct_2d(&block, bs, &basis, &mut tmp, &mut coefs);
        idct_2d(&coefs, bs, &basis, &mut tmp, &mut rec);
        for i in 0..bs * bs {
            assert!(
                (block[i] - rec[i]).abs() < 1e-10,
                "DCT round trip failed at {}: {} vs {}",
                i,
                block[i],
                rec[i]
            );
        }
    }

    #[test]
    fn test_dct_dc_and_flat_block() {
        // A flat block of value v transforms to a single DC coefficient
        // sqrt(N)·v with all AC coefficients zero.
        let bs = 8;
        let basis = dct_basis(bs);
        let v = 0.5;
        let block = vec![v; bs * bs];
        let mut tmp = vec![0.0; bs * bs];
        let mut coefs = vec![0.0; bs * bs];
        dct_2d(&block, bs, &basis, &mut tmp, &mut coefs);

        // Each 1D DCT DC basis row sums to sqrt(bs), so the 2D DC
        // coefficient of a flat block is v * bs.
        let expected_dc = v * bs as f64;
        assert!((coefs[0] - expected_dc).abs() < 1e-10, "DC {}", coefs[0]);
        for (i, &c) in coefs.iter().enumerate().skip(1) {
            assert!(c.abs() < 1e-10, "AC coefficient {} non-zero: {}", i, c);
        }
    }

    #[test]
    fn test_bm3d_reduces_noise() {
        // Noisy step edge: denoised output must be closer to the clean
        // signal than the noisy input.
        let (h, w) = (32usize, 32usize);
        let clean: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| if x < w / 2 { 0.3 } else { 0.7 }))
            .collect();

        // Deterministic pseudo-noise.
        let mut seed = 12345u64;
        let mut noisy = vec![0.0f32; h * w];
        for i in 0..h * w {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) % 200) as f32 / 100.0 - 1.0; // [-1, 1]
            noisy[i] = (clean[i] + 0.15 * noise).clamp(0.0, 1.0);
        }

        let mse = |img: &[f32]| -> f64 {
            img.iter()
                .zip(clean.iter())
                .map(|(&a, &b)| {
                    let d = (a - b) as f64;
                    d * d
                })
                .sum::<f64>()
                / img.len() as f64
        };

        let noisy_mse = mse(&noisy);

        let tensor =
            CpuTensor::<f32>::from_vec(noisy.clone(), TensorShape::new(1, h, w)).unwrap();
        let denoised = bm3d(&tensor, 0.15, 8, 16, 15).unwrap();
        let den_mse = mse(denoised.as_slice().unwrap());

        assert!(
            den_mse < noisy_mse * 0.9,
            "BM3D did not reduce noise: noisy {} vs denoised {}",
            noisy_mse,
            den_mse
        );
    }
}
