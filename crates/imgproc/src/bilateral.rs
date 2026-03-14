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
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    let sigma_range_sq = 2.0 * params.sigma_range * params.sigma_range;

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

                        let spatial_dist = (kx * kx + ky * ky) as f32;
                        let spatial_weight = (-spatial_dist / sigma_space_sq).exp();

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
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    let sigma_range_sq = 2.0 * params.sigma_range * params.sigma_range;

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

                        let spatial_dist = (kx * kx + ky * ky) as f32;
                        let spatial_weight = (-spatial_dist / sigma_space_sq).exp();

                        let range_dist_sq = (center_r - nr).powi(2)
                            + (center_g - ng).powi(2)
                            + (center_b - nb).powi(2);
                        let range_weight = (-range_dist_sq / sigma_range_sq).exp();

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
    let kernel_size = params.kernel_size;
    let half_kernel = kernel_size / 2;
    let sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    let sigma_range_sq = 2.0 * params.sigma_range * params.sigma_range;

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

                        let spatial_dist = (kx * kx + ky * ky) as f32;
                        let spatial_weight = (-spatial_dist / sigma_space_sq).exp();

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
