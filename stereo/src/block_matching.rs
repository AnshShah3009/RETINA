#![allow(deprecated)]

use crate::{DisparityMap, Result, StereoMatcher, StereoMatcherCtx};
use cv_core::{storage::Storage, Error, Tensor};
use cv_hal::compute::ComputeDevice;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;
use wide::*;

/// Block matching stereo matcher
pub struct BlockMatcher {
    pub block_size: usize,
    pub min_disparity: i32,
    pub max_disparity: i32,
    pub metric: MatchingMetric,
    pub uniqueness_ratio: f32,
}

#[derive(Clone, Copy)]
pub enum MatchingMetric {
    SAD, // Sum of Absolute Differences
    SSD, // Sum of Squared Differences
    NCC, // Normalized Cross-Correlation
}

impl Default for BlockMatcher {
    fn default() -> Self {
        Self {
            block_size: 11,
            min_disparity: 0,
            max_disparity: 64,
            metric: MatchingMetric::SAD,
            uniqueness_ratio: 0.95,
        }
    }
}

impl StereoMatcher for BlockMatcher {
    fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap> {
        let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
            // Fallback to CPU registry on error
            cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
        });
        self.compute_ctx(left, right, &runner)
    }
}

impl StereoMatcherCtx for BlockMatcher {
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

        let width = left.width() as i32;
        let height = left.height() as i32;
        let width_usize = left.width() as usize;
        let half_block = (self.block_size / 2) as i32;

        // Check for GPU acceleration
        if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
            if let Ok(res) = self.compute_gpu(gpu, left, right) {
                return Ok(res);
            }
        }

        let left_data = left.as_raw();
        let right_data = right.as_raw();

        let mut disparity = DisparityMap::new(
            left.width(),
            left.height(),
            self.min_disparity,
            self.max_disparity,
        );

        // Compute disparity map row-wise in parallel.
        group.run(|| {
            disparity
                .data
                .par_chunks_mut(width_usize)
                .enumerate()
                .for_each(|(y_usize, row)| {
                    let y = y_usize as i32;
                    if y < half_block || y >= height - half_block {
                        return;
                    }
                    for x in half_block..width - half_block {
                        let best_disparity = self.find_best_disparity(
                            left_data,
                            right_data,
                            width_usize,
                            x,
                            y,
                            half_block,
                        );
                        row[x as usize] = best_disparity as f32;
                    }
                });
        });

        Ok(disparity)
    }
}

impl BlockMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_disparity_range(mut self, min: i32, max: i32) -> Self {
        self.min_disparity = min;
        self.max_disparity = max;
        self
    }

    pub fn with_metric(mut self, metric: MatchingMetric) -> Self {
        self.metric = metric;
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
            method: StereoMatchMethod::BlockMatching,
            min_disparity: self.min_disparity,
            num_disparities: self.max_disparity - self.min_disparity,
            block_size: self.block_size,
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

    fn find_best_disparity(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        half_block: i32,
    ) -> i32 {
        // Clamp disparity search so all right-image accesses stay in-bounds.
        let min_valid = (x + half_block - (width as i32 - 1)).max(self.min_disparity);
        let max_valid = (x - half_block).min(self.max_disparity);
        if min_valid > max_valid {
            return -1;
        }

        let mut best_disparity = self.min_disparity;
        let mut best_cost = f32::INFINITY;
        let mut second_best_cost = f32::INFINITY;

        for d in min_valid..=max_valid {
            let cost =
                self.compute_matching_cost(left_data, right_data, width, x, y, d, half_block);

            if cost < best_cost {
                second_best_cost = best_cost;
                best_cost = cost;
                best_disparity = d;
            } else if cost < second_best_cost {
                second_best_cost = cost;
            }
        }

        // Uniqueness check
        if second_best_cost < best_cost * self.uniqueness_ratio {
            // Ambiguous match
            -1
        } else {
            best_disparity
        }
    }

    fn compute_matching_cost(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        disparity: i32,
        half_block: i32,
    ) -> f32 {
        match self.metric {
            MatchingMetric::SAD => {
                self.compute_sad_simd(left_data, right_data, width, x, y, disparity, half_block)
            }
            MatchingMetric::SSD => {
                self.compute_ssd_simd(left_data, right_data, width, x, y, disparity, half_block)
            }
            MatchingMetric::NCC => {
                let mut cost = 0.0f32;
                let mut count = 0usize;

                for dy in -half_block..=half_block {
                    let ly = (y + dy) as usize;
                    for dx in -half_block..=half_block {
                        let lx = (x + dx) as usize;
                        let rx = (x + dx - disparity) as usize;
                        let left_val = left_data[ly * width + lx] as f32;
                        let right_val = right_data[ly * width + rx] as f32;
                        cost += left_val * right_val;
                        count += 1;
                    }
                }
                cost / count as f32
            }
        }
    }

    fn compute_sad_simd(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        disparity: i32,
        half_block: i32,
    ) -> f32 {
        let mut total_sad = f32x8::ZERO;
        let mut count = 0usize;

        for dy in -half_block..=half_block {
            let ly = (y + dy) as usize;
            let mut dx = -half_block;

            while dx <= half_block - 7 {
                let lx = (x + dx) as usize;
                let rx = (x + dx - disparity) as usize;

                let l_vals = f32x8::from([
                    left_data[ly * width + lx] as f32,
                    left_data[ly * width + lx + 1] as f32,
                    left_data[ly * width + lx + 2] as f32,
                    left_data[ly * width + lx + 3] as f32,
                    left_data[ly * width + lx + 4] as f32,
                    left_data[ly * width + lx + 5] as f32,
                    left_data[ly * width + lx + 6] as f32,
                    left_data[ly * width + lx + 7] as f32,
                ]);

                let r_vals = f32x8::from([
                    right_data[ly * width + rx] as f32,
                    right_data[ly * width + rx + 1] as f32,
                    right_data[ly * width + rx + 2] as f32,
                    right_data[ly * width + rx + 3] as f32,
                    right_data[ly * width + rx + 4] as f32,
                    right_data[ly * width + rx + 5] as f32,
                    right_data[ly * width + rx + 6] as f32,
                    right_data[ly * width + rx + 7] as f32,
                ]);

                total_sad += (l_vals - r_vals).abs();
                dx += 8;
                count += 8;
            }

            // Remainder
            for rem_dx in dx..=half_block {
                let lx = (x + rem_dx) as usize;
                let rx = (x + rem_dx - disparity) as usize;
                let diff =
                    (left_data[ly * width + lx] as f32 - right_data[ly * width + rx] as f32).abs();
                total_sad += f32x8::from([diff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                count += 1;
            }
        }

        total_sad.reduce_add() / count as f32
    }

    fn compute_ssd_simd(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        disparity: i32,
        half_block: i32,
    ) -> f32 {
        let mut total_ssd = f32x8::ZERO;
        let mut count = 0usize;

        for dy in -half_block..=half_block {
            let ly = (y + dy) as usize;
            let mut dx = -half_block;

            while dx <= half_block - 7 {
                let lx = (x + dx) as usize;
                let rx = (x + dx - disparity) as usize;

                let l_vals = f32x8::from([
                    left_data[ly * width + lx] as f32,
                    left_data[ly * width + lx + 1] as f32,
                    left_data[ly * width + lx + 2] as f32,
                    left_data[ly * width + lx + 3] as f32,
                    left_data[ly * width + lx + 4] as f32,
                    left_data[ly * width + lx + 5] as f32,
                    left_data[ly * width + lx + 6] as f32,
                    left_data[ly * width + lx + 7] as f32,
                ]);

                let r_vals = f32x8::from([
                    right_data[ly * width + rx] as f32,
                    right_data[ly * width + rx + 1] as f32,
                    right_data[ly * width + rx + 2] as f32,
                    right_data[ly * width + rx + 3] as f32,
                    right_data[ly * width + rx + 4] as f32,
                    right_data[ly * width + rx + 5] as f32,
                    right_data[ly * width + rx + 6] as f32,
                    right_data[ly * width + rx + 7] as f32,
                ]);

                let diff = l_vals - r_vals;
                total_ssd += diff * diff;
                dx += 8;
                count += 8;
            }

            // Remainder
            for rem_dx in dx..=half_block {
                let lx = (x + rem_dx) as usize;
                let rx = (x + rem_dx - disparity) as usize;
                let diff = left_data[ly * width + lx] as f32 - right_data[ly * width + rx] as f32;
                total_ssd += f32x8::from([diff * diff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                count += 1;
            }
        }

        total_ssd.reduce_add() / count as f32
    }
}

/// Compute stereo matching using block matching
pub fn stereo_block_match(
    left: &GrayImage,
    right: &GrayImage,
    block_size: usize,
    max_disparity: i32,
) -> Result<DisparityMap> {
    let matcher = BlockMatcher::new()
        .with_block_size(block_size)
        .with_disparity_range(0, max_disparity);

    matcher.compute(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 100u32;
        let height = 100u32;

        // Create left image with a pattern
        let mut left = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = ((x / 10) * 25) as u8;
                left.put_pixel(x, y, Luma([val]));
            }
        }

        // Create right image with horizontal shift (disparity)
        let mut right = GrayImage::new(width, height);
        let disparity = 10i32;

        for y in 0..height {
            for x in 0..width {
                let src_x = if x >= disparity as u32 {
                    x - disparity as u32
                } else {
                    0
                };
                let val = ((src_x / 10) * 25) as u8;
                right.put_pixel(x, y, Luma([val]));
            }
        }

        (left, right)
    }

    #[test]
    fn test_block_matcher() {
        let (left, right) = create_test_stereo_pair();

        let matcher = BlockMatcher::new()
            .with_block_size(5)
            .with_disparity_range(0, 20);

        let disparity = matcher.compute(&left, &right).unwrap();

        assert_eq!(disparity.width, left.width());
        assert_eq!(disparity.height, left.height());

        // Check that we have some valid disparities
        let valid_count = disparity.data.iter().filter(|&&d| d >= 0.0).count();

        assert!(valid_count > 0, "Should have some valid disparities");
        println!("Found {} valid disparities", valid_count);
    }
}
