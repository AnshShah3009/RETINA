#![allow(deprecated)]

use crate::{DisparityMap, Result, StereoMatcher, StereoMatcherCtx};
use cv_core::{storage::Storage, Error, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;

/// SGM stereo matcher
pub struct SgmMatcher {
    pub min_disparity: i32,
    pub max_disparity: i32,
    pub p1: u32,                // Penalty for disparity change of 1
    pub p2: u32,                // Penalty for larger disparity changes
    pub paths: Vec<(i32, i32)>, // Aggregation directions
}

impl Default for SgmMatcher {
    fn default() -> Self {
        // 8 directions: horizontal, vertical, and 4 diagonals
        let paths = vec![
            (1, 0),   // Right
            (-1, 0),  // Left
            (0, 1),   // Down
            (0, -1),  // Up
            (1, 1),   // Down-right
            (-1, 1),  // Down-left
            (1, -1),  // Up-right
            (-1, -1), // Up-left
        ];

        Self {
            min_disparity: 0,
            max_disparity: 64,
            p1: 10,
            p2: 120,
            paths,
        }
    }
}

impl StereoMatcher for SgmMatcher {
    fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        self.compute_ctx(left, right, &runner)
    }
}

impl StereoMatcherCtx for SgmMatcher {
    fn compute_ctx(
        &self,
        left: &GrayImage,
        right: &GrayImage,
        group: &RuntimeRunner,
    ) -> Result<DisparityMap> {
        if left.width() != right.width() || left.height() != right.height() {
            return Err(Error::DimensionMismatch(
                "Left and right images must have the same dimensions".to_string(),
            ));
        }

        let width = left.width() as usize;
        let height = left.height() as usize;
        let num_disparities = (self.max_disparity - self.min_disparity + 1) as usize;

        // Check for GPU acceleration
        if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
            if let Ok(res) = self.compute_gpu(gpu, left, right) {
                return Ok(res);
            }
        }

        // Step 1: Compute pixel-wise matching costs (Census or SAD)
        let cost_volume = self.compute_matching_costs(left, right);

        // Step 2: Aggregate costs along multiple paths
        let aggregated_costs = self.aggregate_costs(&cost_volume, width, height, num_disparities);

        // Step 3: Winner-take-all to select disparities
        let mut disparity = DisparityMap::new(
            left.width(),
            left.height(),
            self.min_disparity,
            self.max_disparity,
        );

        group.run(|| {
            disparity
                .data
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, px) in row.iter_mut().enumerate() {
                        let best_d = self.find_best_disparity(
                            &aggregated_costs,
                            x,
                            y,
                            width,
                            num_disparities,
                        );
                        *px = best_d as f32;
                    }
                });
        });

        Ok(disparity)
    }
}

impl SgmMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_disparity_range(mut self, min: i32, max: i32) -> Self {
        self.min_disparity = min;
        self.max_disparity = max;
        self
    }

    pub fn with_penalties(mut self, p1: u32, p2: u32) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self
    }

    fn compute_gpu(
        &self,
        gpu: &cv_hal::gpu::GpuContext,
        left: &GrayImage,
        right: &GrayImage,
    ) -> cv_hal::Result<DisparityMap> {
        use cv_hal::context::{ComputeContext, StereoMatchMethod, StereoMatchParams};
        use cv_hal::tensor_ext::TensorCast;

        let l_tensor = cv_core::CpuTensor::from_vec(
            left.as_raw().to_vec(),
            cv_core::TensorShape::new(1, left.height() as usize, left.width() as usize),
        )
        .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;
        let r_tensor = cv_core::CpuTensor::from_vec(
            right.as_raw().to_vec(),
            cv_core::TensorShape::new(1, right.height() as usize, right.width() as usize),
        )
        .map_err(|e| cv_hal::Error::RuntimeError(e.to_string()))?;

        let l_gpu = l_tensor.to_gpu_ctx(gpu)?;
        let r_gpu = r_tensor.to_gpu_ctx(gpu)?;

        let l_f32 = l_gpu.to_f32_ctx(gpu)?;
        let r_f32 = r_gpu.to_f32_ctx(gpu)?;

        let params = StereoMatchParams {
            method: StereoMatchMethod::SemiGlobalMatching,
            min_disparity: self.min_disparity,
            num_disparities: self.max_disparity - self.min_disparity,
            block_size: 1, // SGM is pixel-wise usually
        };

        let res_gpu: Tensor<f32, cv_hal::storage::GpuStorage<f32>> =
            gpu.stereo_match(&l_f32, &r_f32, &params)?;
        let res_cpu = res_gpu.to_cpu_ctx(gpu)?;

        Ok(DisparityMap {
            data: res_cpu.storage.as_slice().unwrap().to_vec(),
            width: left.width(),
            height: left.height(),
            min_disparity: self.min_disparity,
            max_disparity: self.max_disparity,
        })
    }

    fn compute_matching_costs(&self, left: &GrayImage, right: &GrayImage) -> Vec<u32> {
        let width = left.width() as usize;
        let height = left.height() as usize;
        let num_disparities = (self.max_disparity - self.min_disparity + 1) as usize;
        let left_data = left.as_raw();
        let right_data = right.as_raw();

        let mut costs = vec![0u32; width * height * num_disparities];

        // Use simple SAD (Sum of Absolute Differences) over small window
        let window_size = 3;
        let half_window = window_size / 2;

        let row_stride = width * num_disparities;
        costs
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_costs)| {
                if y < half_window || y >= height - half_window {
                    return;
                }

                for x in half_window..width - half_window {
                    for (d_idx, d) in (self.min_disparity..=self.max_disparity).enumerate() {
                        let mut cost = 0u32;

                        for dy in -(half_window as i32)..=half_window as i32 {
                            let ly = (y as i32 + dy) as usize;
                            for dx in -(half_window as i32)..=half_window as i32 {
                                let lx_i32 = x as i32 + dx;
                                let rx_i32 = lx_i32 - d;
                                if rx_i32 < 0 || rx_i32 >= width as i32 {
                                    continue;
                                }

                                let lx = lx_i32 as usize;
                                let rx = rx_i32 as usize;
                                let left_val = left_data[ly * width + lx] as i32;
                                let right_val = right_data[ly * width + rx] as i32;
                                cost += (left_val - right_val).unsigned_abs();
                            }
                        }

                        row_costs[x * num_disparities + d_idx] = cost;
                    }
                }
            });

        costs
    }

    fn aggregate_costs(
        &self,
        cost_volume: &[u32],
        width: usize,
        height: usize,
        num_disparities: usize,
    ) -> Vec<u32> {
        let mut aggregated = vec![0u32; width * height * num_disparities];
        // Reused scratch buffer to avoid per-direction allocation churn.
        let mut path_costs = vec![0u32; width * height * num_disparities];

        // For each aggregation direction
        for &(dx, dy) in &self.paths {
            self.aggregate_along_path(
                cost_volume,
                &mut aggregated,
                &mut path_costs,
                width,
                height,
                num_disparities,
                dx,
                dy,
            );
        }

        aggregated
    }

    fn aggregate_along_path(
        &self,
        cost_volume: &[u32],
        aggregated: &mut [u32],
        path_costs: &mut [u32],
        width: usize,
        height: usize,
        num_disparities: usize,
        dx: i32,
        dy: i32,
    ) {
        let (x_start, x_end, x_step) = if dx >= 0 {
            (0i32, width as i32, 1i32)
        } else {
            (width as i32 - 1, -1i32, -1i32)
        };
        let (y_start, y_end, y_step) = if dy >= 0 {
            (0i32, height as i32, 1i32)
        } else {
            (height as i32 - 1, -1i32, -1i32)
        };

        let mut y = y_start;
        while y != y_end {
            let mut x = x_start;
            while x != x_end {
                let px = x - dx;
                let py = y - dy;
                let idx_base = (y as usize * width + x as usize) * num_disparities;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let prev_idx = (py as usize * width + px as usize) * num_disparities;
                    let mut prev_min = u32::MAX;
                    for pd in 0..num_disparities {
                        prev_min = prev_min.min(path_costs[prev_idx + pd]);
                    }
                    if prev_min == u32::MAX {
                        prev_min = 0;
                    }
                    let p2_base = prev_min.saturating_add(self.p2);

                    for d in 0..num_disparities {
                        let cd = cost_volume[idx_base + d];
                        let l0 = path_costs[prev_idx + d];
                        let l1 = if d > 0 {
                            path_costs[prev_idx + d - 1].saturating_add(self.p1)
                        } else {
                            u32::MAX
                        };
                        let l2 = if d + 1 < num_disparities {
                            path_costs[prev_idx + d + 1].saturating_add(self.p1)
                        } else {
                            u32::MAX
                        };
                        let best_prev = l0.min(l1).min(l2).min(p2_base);
                        let lr = cd.saturating_add(best_prev.saturating_sub(prev_min));
                        path_costs[idx_base + d] = lr;
                        aggregated[idx_base + d] = aggregated[idx_base + d].saturating_add(lr);
                    }
                } else {
                    for d in 0..num_disparities {
                        let cd = cost_volume[idx_base + d];
                        path_costs[idx_base + d] = cd;
                        aggregated[idx_base + d] = aggregated[idx_base + d].saturating_add(cd);
                    }
                }

                x += x_step;
            }
            y += y_step;
        }
    }

    fn find_best_disparity(
        &self,
        aggregated_costs: &[u32],
        x: usize,
        y: usize,
        width: usize,
        num_disparities: usize,
    ) -> i32 {
        let idx_base = (y * width + x) * num_disparities;

        let mut best_d = 0;
        let mut min_cost = u32::MAX;

        for d in 0..num_disparities {
            let cost = aggregated_costs[idx_base + d];
            if cost < min_cost {
                min_cost = cost;
                best_d = d;
            }
        }

        self.min_disparity + best_d as i32
    }
}

/// Compute stereo matching using Semi-Global Matching
pub fn stereo_sgm(left: &GrayImage, right: &GrayImage, max_disparity: i32) -> Result<DisparityMap> {
    let matcher = SgmMatcher::new().with_disparity_range(0, max_disparity);

    matcher.compute(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 64u32;
        let height = 64u32;

        let mut left = GrayImage::new(width, height);
        let mut right = GrayImage::new(width, height);

        // Create vertical stripes with disparity
        for y in 0..height {
            for x in 0..width {
                let pattern = ((x / 8) % 2) * 200;
                left.put_pixel(x, y, Luma([pattern as u8]));

                // Shift pattern by 5 pixels for right image
                let shifted_x = if x >= 5 { x - 5 } else { 0 };
                let right_pattern = ((shifted_x / 8) % 2) * 200;
                right.put_pixel(x, y, Luma([right_pattern as u8]));
            }
        }

        (left, right)
    }

    #[test]
    fn test_sgm_matcher() {
        let (left, right) = create_test_stereo_pair();

        let matcher = SgmMatcher::new()
            .with_disparity_range(0, 16)
            .with_penalties(5, 60);

        let disparity = matcher.compute(&left, &right).unwrap();

        assert_eq!(disparity.width, left.width());
        assert_eq!(disparity.height, left.height());

        println!(
            "SGM computed disparity map: {}x{}",
            disparity.width, disparity.height
        );

        // Check that disparities are within expected range
        let valid_count = disparity
            .data
            .iter()
            .filter(|&&d| d >= 0.0 && d <= 16.0)
            .count();

        assert!(valid_count > 0, "Should have valid disparities");
    }
}
