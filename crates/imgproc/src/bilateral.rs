//! Bilateral Filter
//!
//! Edge-preserving smoothing filter that uses both spatial and range (intensity) distance.

use cv_runtime::orchestrator::RuntimeRunner;
use rayon::prelude::*;

/// Bilateral filter parameters
#[derive(Debug, Clone)]
pub struct BilateralFilterParams {
    /// Filter size (must be odd)
    pub kernel_size: i32,
    /// Spatial sigma (larger = more smoothing)
    pub sigma_space: f32,
    /// Range sigma (larger = less edge preservation, more smoothing)
    pub sigma_range: f32,
}

impl Default for BilateralFilterParams {
    fn default() -> Self {
        Self {
            kernel_size: 5,
            sigma_space: 3.0,
            sigma_range: 0.1,
        }
    }
}

/// One spatial weight per kernel offset, in `(ky, kx)` loop order.
///
/// Each entry is `(-((kx*kx + ky*ky) as f32) / sigma_space_sq).exp()`, the
/// same value the tap used to evaluate. Callers multiply it on the left of
/// the range weight (`spatial * range`), which is the previous order.
fn spatial_weights(half_kernel: i32, sigma_space_sq: f32) -> Vec<f32> {
    let mut weights = Vec::new();
    for ky in -half_kernel..=half_kernel {
        for kx in -half_kernel..=half_kernel {
            let spatial_dist = (kx * kx + ky * ky) as f32;
            weights.push((-spatial_dist / sigma_space_sq).exp());
        }
    }
    weights
}

fn spatial_index(kx: i32, ky: i32, half_kernel: i32) -> usize {
    let extent = (half_kernel * 2 + 1) as usize;
    (ky + half_kernel) as usize * extent + (kx + half_kernel) as usize
}

/// `(-(d as f32) * (d as f32) / sigma_range_sq).exp()` for integer diffs 0..=255.
fn u8_range_lut(sigma_range_sq: f32) -> [f32; 256] {
    std::array::from_fn(|d| {
        let range_dist = d as f32;
        (-range_dist * range_dist / sigma_range_sq).exp()
    })
}

impl BilateralFilterParams {
    pub fn fast() -> Self {
        Self {
            kernel_size: 3,
            sigma_space: 2.0,
            sigma_range: 0.1,
        }
    }

    pub fn high_quality() -> Self {
        Self {
            kernel_size: 9,
            sigma_space: 6.0,
            sigma_range: 0.1,
        }
    }
}

/// Apply bilateral filter to depth image (1D array)
pub fn bilateral_filter_depth(
    depth: &[f32],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) -> Vec<f32> {
    let runner = cv_runtime::scheduler()
        .and_then(|s| {
            s.best_gpu_or_cpu_for(cv_runtime::orchestrator::WorkloadHint::Throughput)
                .map(cv_runtime::RuntimeRunner::Group)
        })
        .or_else(|_| cv_runtime::default_runner())
        .unwrap_or_else(|_| {
            // Absolute fallback: use the CPU registry if available
            cv_runtime::registry()
                .ok()
                .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
                .unwrap_or_else(|| {
                    // If even registry fails, create a minimal CPU runner (id = 0)
                    cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0))
                })
        });
    bilateral_filter_depth_ctx(depth, width, height, params, &runner)
}

#[allow(clippy::needless_range_loop)]
fn bilateral_filter_depth_internal(
    depth: &[f32],
    output: &mut [f32],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) {
    if width == 0 || height == 0 {
        return;
    }
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    let sigma_range_sq = 2.0 * params.sigma_range * params.sigma_range;
    // Spatial weight depends only on the offset. Range stays per tap: depth
    // differences are not integer 0..=255.
    let spatial = spatial_weights(half_kernel, sigma_space_sq);

    output
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width as usize {
                let idx = y as usize * width as usize + x;
                let center_val = depth[idx];

                if center_val <= 0.0 || center_val.is_nan() {
                    row[x] = center_val;
                    continue;
                }

                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for ky in -half_kernel..=half_kernel {
                    for kx in -half_kernel..=half_kernel {
                        let nx = x as i32 + kx;
                        let ny = y + ky;

                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }

                        let nidx = ny as usize * width as usize + nx as usize;
                        let neighbor_val = depth[nidx];

                        if neighbor_val <= 0.0 || neighbor_val.is_nan() {
                            continue;
                        }

                        let spatial_weight = spatial[spatial_index(kx, ky, half_kernel)];

                        let range_dist = (center_val - neighbor_val).abs();
                        let range_weight = (-range_dist * range_dist / sigma_range_sq).exp();

                        let weight = spatial_weight * range_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                row[x] = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    center_val
                };
            }
        });
}

pub fn bilateral_filter_depth_ctx(
    depth: &[f32],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
    group: &RuntimeRunner,
) -> Vec<f32> {
    group.run(|| {
        let mut output = vec![0.0f32; depth.len()];
        bilateral_filter_depth_internal(depth, &mut output, width, height, params);
        output
    })
}

/// Apply bilateral filter to RGB image
pub fn bilateral_filter_rgb(
    image: &[u8],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) -> Vec<u8> {
    let runner = cv_runtime::scheduler()
        .and_then(|s| {
            s.best_gpu_or_cpu_for(cv_runtime::orchestrator::WorkloadHint::Throughput)
                .map(cv_runtime::RuntimeRunner::Group)
        })
        .or_else(|_| cv_runtime::default_runner())
        .unwrap_or_else(|_| {
            // Absolute fallback: use the CPU registry if available
            cv_runtime::registry()
                .ok()
                .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
                .unwrap_or_else(|| {
                    // If even registry fails, create a minimal CPU runner (id = 0)
                    cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0))
                })
        });
    bilateral_filter_rgb_ctx(image, width, height, params, &runner)
}

fn bilateral_filter_rgb_internal(
    image: &[u8],
    output: &mut [u8],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) {
    if width == 0 || height == 0 {
        return;
    }
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    // Range distances here are 0-255 per channel; scale the configured
    // sigma into that domain (sigma_range=0.1 on raw u8 distances gave
    // weights of e^-(1/0.02), making the filter an exact no-op).
    let sigma_range_u8 = params.sigma_range * 255.0;
    let sigma_range_sq = 2.0 * sigma_range_u8 * sigma_range_u8;
    let spatial = spatial_weights(half_kernel, sigma_space_sq);
    // Per-channel integer diffs. The product of these entries is the
    // factored form of exp(-(dr²+dg²+db²) / sigma_range_sq), multiplied
    // as spatial * range in the same order as the direct weight.
    let range_lut = u8_range_lut(sigma_range_sq);

    let channels = 3;
    let stride = width as usize * channels;

    output
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width as usize {
                let pixel_offset = x * channels;
                let idx = (y as usize * width as usize + x) * channels;

                let center_r = image[idx] as f32;
                let center_g = image[idx + 1] as f32;
                let center_b = image[idx + 2] as f32;

                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut weight_sum = 0.0f32;

                for ky in -half_kernel..=half_kernel {
                    for kx in -half_kernel..=half_kernel {
                        let nx = x as i32 + kx;
                        let ny = y + ky;

                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }

                        let nidx = (ny as usize * width as usize + nx as usize) * channels;

                        let nr = image[nidx] as f32;
                        let ng = image[nidx + 1] as f32;
                        let nb = image[nidx + 2] as f32;

                        let spatial_weight = spatial[spatial_index(kx, ky, half_kernel)];

                        let dr = (center_r - nr).abs() as usize;
                        let dg = (center_g - ng).abs() as usize;
                        let db = (center_b - nb).abs() as usize;
                        let range_weight = range_lut[dr] * range_lut[dg] * range_lut[db];

                        let weight = spatial_weight * range_weight;
                        sum_r += nr * weight;
                        sum_g += ng * weight;
                        sum_b += nb * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    row[pixel_offset] = (sum_r / weight_sum).round().clamp(0.0, 255.0) as u8;
                    row[pixel_offset + 1] = (sum_g / weight_sum).round().clamp(0.0, 255.0) as u8;
                    row[pixel_offset + 2] = (sum_b / weight_sum).round().clamp(0.0, 255.0) as u8;
                } else {
                    row[pixel_offset] = image[idx];
                    row[pixel_offset + 1] = image[idx + 1];
                    row[pixel_offset + 2] = image[idx + 2];
                }
            }
        });
}

pub fn bilateral_filter_rgb_ctx(
    image: &[u8],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
    group: &RuntimeRunner,
) -> Vec<u8> {
    group.run(|| {
        let mut output = vec![0u8; image.len()];
        bilateral_filter_rgb_internal(image, &mut output, width, height, params);
        output
    })
}

/// Joint bilateral filter (uses guidance image for edge detection)
pub fn joint_bilateral_filter(
    depth: &[f32],
    guidance: &[u8],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) -> Vec<f32> {
    let runner = cv_runtime::scheduler()
        .and_then(|s| {
            s.best_gpu_or_cpu_for(cv_runtime::orchestrator::WorkloadHint::Throughput)
                .map(cv_runtime::RuntimeRunner::Group)
        })
        .or_else(|_| cv_runtime::default_runner())
        .unwrap_or_else(|_| {
            // Absolute fallback: use the CPU registry if available
            cv_runtime::registry()
                .ok()
                .map(|reg| cv_runtime::RuntimeRunner::Sync(reg.default_cpu().id()))
                .unwrap_or_else(|| {
                    // If even registry fails, create a minimal CPU runner (id = 0)
                    cv_runtime::RuntimeRunner::Sync(cv_hal::DeviceId(0))
                })
        });
    joint_bilateral_filter_ctx(depth, guidance, width, height, params, &runner)
}

#[allow(clippy::needless_range_loop)]
fn joint_bilateral_filter_internal(
    depth: &[f32],
    guidance: &[u8],
    output: &mut [f32],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
) {
    if width == 0 || height == 0 {
        return;
    }
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    // Guidance is u8 (0-255); scale the range sigma accordingly.
    let sigma_range_u8 = params.sigma_range * 255.0;
    let sigma_range_sq = 2.0 * sigma_range_u8 * sigma_range_u8;
    // Float path: the range weight is still the direct exp, so f32 results
    // stay bit-exact. The spatial factor depends only on the offset.
    let spatial = spatial_weights(half_kernel, sigma_space_sq);

    let channels = 3;

    output
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width as usize {
                let idx = y as usize * width as usize + x;
                let center_depth = depth[idx];

                let gidx = idx * channels;
                let center_gi = [
                    guidance[gidx] as f32,
                    guidance[gidx + 1] as f32,
                    guidance[gidx + 2] as f32,
                ];

                if center_depth <= 0.0 || center_depth.is_nan() {
                    row[x] = center_depth;
                    continue;
                }

                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for ky in -half_kernel..=half_kernel {
                    for kx in -half_kernel..=half_kernel {
                        let nx = x as i32 + kx;
                        let ny = y + ky;

                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }

                        let nidx = ny as usize * width as usize + nx as usize;
                        let neighbor_depth = depth[nidx];

                        if neighbor_depth <= 0.0 || neighbor_depth.is_nan() {
                            continue;
                        }

                        let ngidx = nidx * channels;
                        let neighbor_gi = [
                            guidance[ngidx] as f32,
                            guidance[ngidx + 1] as f32,
                            guidance[ngidx + 2] as f32,
                        ];

                        let spatial_weight = spatial[spatial_index(kx, ky, half_kernel)];

                        let range_dist_sq = (center_gi[0] - neighbor_gi[0]).powi(2)
                            + (center_gi[1] - neighbor_gi[1]).powi(2)
                            + (center_gi[2] - neighbor_gi[2]).powi(2);
                        let range_weight = (-range_dist_sq / sigma_range_sq).exp();

                        let weight = spatial_weight * range_weight;
                        sum += neighbor_depth * weight;
                        weight_sum += weight;
                    }
                }

                row[x] = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    center_depth
                };
            }
        });
}

pub fn joint_bilateral_filter_ctx(
    depth: &[f32],
    guidance: &[u8],
    width: u32,
    height: u32,
    params: BilateralFilterParams,
    group: &RuntimeRunner,
) -> Vec<f32> {
    group.run(|| {
        let mut output = vec![0.0f32; depth.len()];
        joint_bilateral_filter_internal(depth, guidance, &mut output, width, height, params);
        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilateral_preserves_uniform() {
        // A uniform depth image (all 1.0) should be unchanged after filtering.
        let width = 20u32;
        let height = 20u32;
        let depth: Vec<f32> = vec![1.0; (width * height) as usize];
        let mut output = vec![0.0f32; depth.len()];
        bilateral_filter_depth_internal(
            &depth,
            &mut output,
            width,
            height,
            BilateralFilterParams::default(),
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "pixel {} expected 1.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_bilateral_preserves_edge() {
        // Step edge image: left half = 0.5, right half = 1.5.
        // With a small sigma_range the filter should not blend across the edge.
        let width = 40u32;
        let height = 20u32;
        let mut depth = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                depth[idx] = if x < width / 2 { 0.5 } else { 1.5 };
            }
        }

        let params = BilateralFilterParams {
            kernel_size: 5,
            sigma_space: 3.0,
            sigma_range: 0.01, // very small range sigma => strong edge preservation
        };

        let mut output = vec![0.0f32; depth.len()];
        bilateral_filter_depth_internal(&depth, &mut output, width, height, params);

        // Check pixels well inside each half (away from the edge boundary).
        // Left interior (columns 0..15) should stay close to 0.5.
        // Right interior (columns 25..40) should stay close to 1.5.
        for y in 2..height - 2 {
            for x in 0..15 {
                let v = output[(y * width + x) as usize];
                assert!(
                    (v - 0.5).abs() < 0.05,
                    "left interior pixel ({},{}) expected ~0.5, got {}",
                    x,
                    y,
                    v
                );
            }
            for x in 25..width {
                let v = output[(y * width + x) as usize];
                assert!(
                    (v - 1.5).abs() < 0.05,
                    "right interior pixel ({},{}) expected ~1.5, got {}",
                    x,
                    y,
                    v
                );
            }
        }
    }

    /// A constant image is a fixed point: every range weight is 1 and the
    /// spatial weights cancel, so the sample comes back unchanged.
    #[test]
    fn uniform_images_are_fixed_points() {
        let params = BilateralFilterParams {
            kernel_size: 5,
            sigma_space: 3.0,
            sigma_range: 0.1,
        };
        let depth = vec![1.0f32; 16 * 16];
        let mut depth_out = vec![0.0f32; depth.len()];
        bilateral_filter_depth_internal(&depth, &mut depth_out, 16, 16, params.clone());
        assert!(
            depth_out.iter().all(|v| v.to_bits() == 1.0f32.to_bits()),
            "uniform depth drifted to {}",
            depth_out[0]
        );

        for v in [1u8, 37, 128, 200, 254, 255] {
            let rgb = vec![v; 12 * 12 * 3];
            let mut rgb_out = vec![0u8; rgb.len()];
            bilateral_filter_rgb_internal(&rgb, &mut rgb_out, 12, 12, params.clone());
            assert!(
                rgb_out.iter().all(|&p| p == v),
                "uniform rgb {v} drifted to {}",
                rgb_out[0]
            );
        }

        let guidance = vec![40u8; 16 * 16 * 3];
        let mut joint_out = vec![0.0f32; depth.len()];
        joint_bilateral_filter_internal(&depth, &guidance, &mut joint_out, 16, 16, params);
        assert!(
            joint_out.iter().all(|v| v.to_bits() == 1.0f32.to_bits()),
            "uniform joint drifted to {}",
            joint_out[0]
        );
    }

    /// The u8 range table is the factored Gaussian. Rounded bytes must match
    /// the direct `exp(-(dr²+dg²+db²)/σ²)` the loop used to evaluate.
    #[test]
    fn rgb_range_table_matches_direct_exp_bytes() {
        let (w, h) = (16u32, 12u32);
        let mut image = vec![0u8; (w * h * 3) as usize];
        let mut s = 1u32;
        for p in image.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *p = (s >> 24) as u8;
        }
        let params = BilateralFilterParams {
            kernel_size: 5,
            sigma_space: 3.0,
            sigma_range: 0.1,
        };
        let mut got = vec![0u8; image.len()];
        bilateral_filter_rgb_internal(&image, &mut got, w, h, params.clone());

        let half = params.kernel_size / 2;
        let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
        let sigma_range_u8 = params.sigma_range * 255.0;
        let sigma_range_sq = 2.0 * sigma_range_u8 * sigma_range_u8;
        let mut expect = vec![0u8; image.len()];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let idx = ((y as u32 * w + x as u32) * 3) as usize;
                let cr = image[idx] as f32;
                let cg = image[idx + 1] as f32;
                let cb = image[idx + 2] as f32;
                let mut sr = 0.0f32;
                let mut sg = 0.0f32;
                let mut sb = 0.0f32;
                let mut weight_sum = 0.0f32;
                for ky in -half..=half {
                    for kx in -half..=half {
                        let nx = x + kx;
                        let ny = y + ky;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nidx = ((ny as u32 * w + nx as u32) * 3) as usize;
                        let nr = image[nidx] as f32;
                        let ng = image[nidx + 1] as f32;
                        let nb = image[nidx + 2] as f32;
                        let spatial_dist = (kx * kx + ky * ky) as f32;
                        let spatial_weight = (-spatial_dist / sigma_space_sq).exp();
                        let range_dist_sq =
                            (cr - nr).powi(2) + (cg - ng).powi(2) + (cb - nb).powi(2);
                        let range_weight = (-range_dist_sq / sigma_range_sq).exp();
                        let weight = spatial_weight * range_weight;
                        sr += nr * weight;
                        sg += ng * weight;
                        sb += nb * weight;
                        weight_sum += weight;
                    }
                }
                expect[idx] = (sr / weight_sum).round().clamp(0.0, 255.0) as u8;
                expect[idx + 1] = (sg / weight_sum).round().clamp(0.0, 255.0) as u8;
                expect[idx + 2] = (sb / weight_sum).round().clamp(0.0, 255.0) as u8;
            }
        }
        assert_eq!(got, expect);
    }
}
