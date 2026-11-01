#![allow(deprecated)]

use super::{DisparityMap, StereoMatcher, StereoMatcherCtx};
use crate::Result;
use cv_core::{Error, Tensor, storage::Storage};
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
            uniqueness_ratio: 0.15,
        }
    }
}

impl StereoMatcher for BlockMatcher {
    fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap> {
        let runner = cv_runtime::best_runner()
            .map(|(r, _)| r)
            .unwrap_or_else(|_| {
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
        if let Ok(ComputeDevice::Gpu(gpu)) = group.device()
            && let Ok(res) = self.compute_gpu(gpu, left, right)
        {
            return Ok(res);
        }

        let left_data = left.as_raw();
        let right_data = right.as_raw();

        let mut disparity = DisparityMap::new(
            left.width(),
            left.height(),
            self.min_disparity,
            self.max_disparity,
        );

        // SAD / SSD over a full window is a box filter of the per-disparity
        // cost image. A column running sum slid along the row reproduces the
        // dense walk. u8 sums of an 11×11 block fit in f32 exactly, so the
        // costs — and the low-to-high, first-on-tie winner — stay identical.
        // Larger windows keep the dense walk.
        let window = 2 * (self.block_size / 2) + 1;
        let squared = matches!(self.metric, MatchingMetric::SSD);
        if matches!(self.metric, MatchingMetric::SAD | MatchingMetric::SSD)
            && box_sum_exact_in_f32(window, squared)
        {
            group.run(|| {
                self.fill_disparity_running_box(
                    left_data,
                    right_data,
                    width_usize,
                    left.height() as usize,
                    self.block_size / 2,
                    &mut disparity.data,
                );
            });
            return Ok(disparity);
        }

        // NCC means are box sums of the two images. The cross term stays a
        // direct walk. Empty integrals mean "use the dense mean".
        let integral_means =
            matches!(self.metric, MatchingMetric::NCC) && box_sum_exact_in_f32(window, false);
        let (sat_l, sat_r) = if integral_means {
            (
                integral_image(left_data, width_usize, left.height() as usize),
                integral_image(right_data, width_usize, left.height() as usize),
            )
        } else {
            (Vec::new(), Vec::new())
        };

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
                            &sat_l,
                            &sat_r,
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

    #[allow(clippy::too_many_arguments)]
    fn find_best_disparity(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        half_block: i32,
        sat_l: &[u64],
        sat_r: &[u64],
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
            let cost = self.compute_matching_cost(
                left_data, right_data, width, x, y, d, half_block, sat_l, sat_r,
            );

            if cost < best_cost {
                second_best_cost = best_cost;
                best_cost = cost;
                best_disparity = d;
            } else if cost < second_best_cost {
                second_best_cost = cost;
            }
        }

        // Uniqueness check
        if second_best_cost < best_cost * (1.0 + self.uniqueness_ratio) {
            // Ambiguous match
            -1
        } else {
            best_disparity
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_matching_cost(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        disparity: i32,
        half_block: i32,
        sat_l: &[u64],
        sat_r: &[u64],
    ) -> f32 {
        match self.metric {
            MatchingMetric::SAD => {
                self.compute_sad_simd(left_data, right_data, width, x, y, disparity, half_block)
            }
            MatchingMetric::SSD => {
                self.compute_ssd_simd(left_data, right_data, width, x, y, disparity, half_block)
            }
            MatchingMetric::NCC => {
                // Means are box sums. When the integral is exact in f32 it
                // matches the dense accumulation; the cross term stays a walk.
                let (mean_l, mean_r) = if !sat_l.is_empty() {
                    let span = (half_block * 2 + 1) as usize;
                    let y0 = (y - half_block) as usize;
                    let x0 = (x - half_block) as usize;
                    let rx0 = (x - half_block - disparity) as usize;
                    let stride = width + 1;
                    let sum_l = rect_sum(sat_l, stride, x0, y0, x0 + span, y0 + span) as f32;
                    let sum_r = rect_sum(sat_r, stride, rx0, y0, rx0 + span, y0 + span) as f32;
                    let count = (span * span) as f32;
                    (sum_l / count, sum_r / count)
                } else {
                    let mut sum_l = 0.0f32;
                    let mut sum_r = 0.0f32;
                    let mut count = 0usize;

                    for dy in -half_block..=half_block {
                        let ly = (y + dy) as usize;
                        for dx in -half_block..=half_block {
                            let lx = (x + dx) as usize;
                            let rx = (x + dx - disparity) as usize;
                            sum_l += left_data[ly * width + lx] as f32;
                            sum_r += right_data[ly * width + rx] as f32;
                            count += 1;
                        }
                    }

                    (sum_l / count as f32, sum_r / count as f32)
                };

                // Second pass: compute ZNCC
                let mut cross = 0.0f32;
                let mut var_l = 0.0f32;
                let mut var_r = 0.0f32;

                for dy in -half_block..=half_block {
                    let ly = (y + dy) as usize;
                    for dx in -half_block..=half_block {
                        let lx = (x + dx) as usize;
                        let rx = (x + dx - disparity) as usize;
                        let dl = left_data[ly * width + lx] as f32 - mean_l;
                        let dr = right_data[ly * width + rx] as f32 - mean_r;
                        cross += dl * dr;
                        var_l += dl * dl;
                        var_r += dr * dr;
                    }
                }

                let denom = (var_l * var_r).sqrt();
                let ncc = if denom > 1e-10 { cross / denom } else { 0.0 };
                // Negate because the framework minimizes cost, but NCC should be maximized
                -ncc
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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

    /// SAD or SSD via a per-disparity column running sum and a row running sum.
    ///
    /// Disparities run low to high and a tie keeps the first, matching the
    /// dense loop. The caller only uses this when every block sum is an exact
    /// f32 integer, so the averaged costs — and the uniqueness test — match.
    fn fill_disparity_running_box(
        &self,
        left: &[u8],
        right: &[u8],
        w: usize,
        h: usize,
        half: usize,
        out: &mut [f32],
    ) {
        let window = 2 * half + 1;
        let count = (window * window) as f32;
        let squared = matches!(self.metric, MatchingMetric::SSD);
        let min_d = self.min_disparity;
        let max_d = self.max_disparity;
        let uniq = self.uniqueness_ratio;
        if h < window || w < window {
            return;
        }
        if min_d > max_d {
            for y in half..h - half {
                for x in half..w - half {
                    out[y * w + x] = -1.0;
                }
            }
            return;
        }

        let nd = (max_d as i64 - min_d as i64 + 1) as usize;
        let (y_lo, y_hi) = (half, h - half);
        let (x_lo, x_hi) = (half as i64, (w - half) as i64);

        const BAND: usize = 16;
        out[y_lo * w..y_hi * w]
            .par_chunks_mut(w * BAND)
            .enumerate()
            .for_each(|(band, out_band)| {
                let y0 = y_lo + band * BAND;
                let rows = out_band.len() / w;
                let mut colsum = vec![0u64; nd * w];
                let mut best = vec![u64::MAX; w];
                let mut second = vec![u64::MAX; w];
                let mut best_d = vec![0i32; w];
                let mut tested = vec![false; w];
                for dy in 0..rows {
                    let y = y0 + dy;
                    let out_row = &mut out_band[dy * w..(dy + 1) * w];
                    best.fill(u64::MAX);
                    second.fill(u64::MAX);
                    tested.fill(false);
                    for di in 0..nd {
                        let disp = min_d as i64 + di as i64;
                        // Columns where both c and c - disp lie inside the image.
                        let c_lo = disp.clamp(0, w as i64) as usize;
                        let c_hi = (w as i64 + disp).clamp(c_lo as i64, w as i64) as usize;
                        if c_lo == c_hi {
                            continue;
                        }
                        let r_lo = (c_lo as i64 - disp) as usize;
                        let cs = &mut colsum[di * w..(di + 1) * w];
                        if dy == 0 {
                            cs[c_lo..c_hi].fill(0);
                            for r in y - half..y - half + window {
                                let lrow = &left[r * w..(r + 1) * w];
                                let rrow = &right[r * w..(r + 1) * w];
                                for i in 0..(c_hi - c_lo) {
                                    cs[c_lo + i] +=
                                        pair_cost(lrow[c_lo + i], rrow[r_lo + i], squared);
                                }
                            }
                        } else {
                            let add = y - half + window - 1;
                            let sub = y - half - 1;
                            let la = &left[add * w..(add + 1) * w];
                            let ra = &right[add * w..(add + 1) * w];
                            let ls = &left[sub * w..(sub + 1) * w];
                            let rs = &right[sub * w..(sub + 1) * w];
                            for i in 0..(c_hi - c_lo) {
                                cs[c_lo + i] = cs[c_lo + i]
                                    + pair_cost(la[c_lo + i], ra[r_lo + i], squared)
                                    - pair_cost(ls[c_lo + i], rs[r_lo + i], squared);
                            }
                        }
                        // Pixels whose full window stays inside both images.
                        let xa = x_lo.max(x_lo + disp);
                        let xb = x_hi.min(x_hi + disp);
                        if xa >= xb {
                            continue;
                        }
                        let (xa, xb) = (xa as usize, xb as usize);
                        let mut s: u64 = cs[xa - half..xa - half + window].iter().sum();
                        for x in xa..xb {
                            if x > xa {
                                s += cs[x - half + window - 1];
                                s -= cs[x - half - 1];
                            }
                            tested[x] = true;
                            if s < best[x] {
                                second[x] = best[x];
                                best[x] = s;
                                best_d[x] = disp as i32;
                            } else if s < second[x] {
                                second[x] = s;
                            }
                        }
                    }
                    for x in half..w - half {
                        out_row[x] = if !tested[x] {
                            -1.0
                        } else {
                            let best_cost = best[x] as f32 / count;
                            let second_cost = if second[x] == u64::MAX {
                                f32::INFINITY
                            } else {
                                second[x] as f32 / count
                            };
                            if second_cost < best_cost * (1.0 + uniq) {
                                -1.0
                            } else {
                                best_d[x] as f32
                            }
                        };
                    }
                }
            });
    }
}

/// Every integer in `[0, 2^24]` is exact in f32. A u8 block sum at or below
/// that limit matches the dense f32 accumulation, so the argmin does too.
fn box_sum_exact_in_f32(window: usize, squared: bool) -> bool {
    let per = if squared { 255u64 * 255 } else { 255 };
    (window as u64)
        .checked_mul(window as u64)
        .and_then(|n| n.checked_mul(per))
        .is_some_and(|worst| worst <= (1u64 << 24))
}

fn pair_cost(a: u8, b: u8, squared: bool) -> u64 {
    let d = u64::from(a.abs_diff(b));
    if squared { d * d } else { d }
}

/// Inclusive-exclusive box sums from a summed-area table built with a row
/// running sum and a column running sum. `stride` is `width + 1`.
fn integral_image(data: &[u8], w: usize, h: usize) -> Vec<u64> {
    let stride = w + 1;
    let mut sat = vec![0u64; stride * (h + 1)];
    for y in 0..h {
        let mut row = 0u64;
        for x in 0..w {
            row += u64::from(data[y * w + x]);
            sat[(y + 1) * stride + (x + 1)] = sat[y * stride + (x + 1)] + row;
        }
    }
    sat
}

fn rect_sum(sat: &[u64], stride: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
    sat[y1 * stride + x1] + sat[y0 * stride + x0] - sat[y0 * stride + x1] - sat[y1 * stride + x0]
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
                let src_x = x.saturating_sub(disparity as u32);
                let val = ((src_x / 10) * 25) as u8;
                right.put_pixel(x, y, Luma([val]));
            }
        }

        (left, right)
    }

    /// Scalar SAD with the same window, disparity order, and uniqueness rule
    /// as `BlockMatcher`. Even `block_size` still uses `block_size / 2` as the
    /// half-window, so the scored window is `2 * (block_size / 2) + 1`.
    fn naive_sad(
        left: &GrayImage,
        right: &GrayImage,
        block_size: usize,
        min_d: i32,
        max_d: i32,
        uniqueness_ratio: f32,
    ) -> Vec<f32> {
        let width = left.width() as i32;
        let height = left.height() as i32;
        let half = (block_size / 2) as i32;
        let left_data = left.as_raw();
        let right_data = right.as_raw();
        let stride = left.width() as usize;
        let mut out = vec![f32::NAN; (width * height) as usize];

        for y in 0..height {
            if y < half || y >= height - half {
                continue;
            }
            for x in half..width - half {
                let min_valid = (x + half - (width - 1)).max(min_d);
                let max_valid = (x - half).min(max_d);
                let idx = (y as usize) * stride + x as usize;
                if min_valid > max_valid {
                    out[idx] = -1.0;
                    continue;
                }
                let mut best_d = min_d;
                let mut best_cost = f32::INFINITY;
                let mut second = f32::INFINITY;
                for d in min_valid..=max_valid {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for dy in -half..=half {
                        let ly = (y + dy) as usize;
                        for dx in -half..=half {
                            let lx = (x + dx) as usize;
                            let rx = (x + dx - d) as usize;
                            sum += (left_data[ly * stride + lx] as f32
                                - right_data[ly * stride + rx] as f32)
                                .abs();
                            count += 1;
                        }
                    }
                    let cost = sum / count as f32;
                    if cost < best_cost {
                        second = best_cost;
                        best_cost = cost;
                        best_d = d;
                    } else if cost < second {
                        second = cost;
                    }
                }
                out[idx] = if second < best_cost * (1.0 + uniqueness_ratio) {
                    -1.0
                } else {
                    best_d as f32
                };
            }
        }
        out
    }

    #[test]
    fn block_matching_matches_naive_sad_even_and_odd_blocks() {
        let width = 16u32;
        let height = 12u32;
        let mut left = GrayImage::new(width, height);
        let mut right = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 17 + y * 13) % 64) as u8 * 3;
                left.put_pixel(x, y, Luma([v]));
                let sx = x.saturating_sub(2);
                let rv = ((sx * 17 + y * 13) % 64) as u8 * 3;
                right.put_pixel(x, y, Luma([rv]));
            }
        }

        for block_size in [4usize, 5] {
            let matcher = BlockMatcher::new()
                .with_block_size(block_size)
                .with_disparity_range(0, 6)
                .with_metric(MatchingMetric::SAD);
            let got = matcher.compute(&left, &right).unwrap();
            let expect = naive_sad(&left, &right, block_size, 0, 6, matcher.uniqueness_ratio);
            assert_eq!(got.data.len(), expect.len());
            for (i, (&g, &e)) in got.data.iter().zip(expect.iter()).enumerate() {
                assert!(
                    (g.is_nan() && e.is_nan()) || g == e,
                    "block {block_size} pixel {i}: matcher {g} != naive {e}"
                );
            }
        }
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
