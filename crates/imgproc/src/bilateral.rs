//! Bilateral Filter
//!
//! Edge-preserving smoothing filter that uses both spatial and range (intensity) distance.
//!
//! The kernel follows `cv::bilateralFilter`: the disc of offsets within the
//! radius, a reflect-101 (`BORDER_DEFAULT`) border, and OpenCV's radius rule.
//! [`bilateral_filter_gray`] and [`bilateral_filter_rgb`] reproduce
//! `cv2.bilateralFilter` on 1- and 3-channel `u8` images byte for byte; see
//! `tests/bilateral_opencv_parity.rs` and `tests/bilateral_gray_opencv_parity.rs`.

use cv_runtime::orchestrator::RuntimeRunner;
use image::GrayImage;
use rayon::prelude::*;

/// Bilateral filter parameters
#[derive(Debug, Clone)]
pub struct BilateralFilterParams {
    /// Kernel diameter, OpenCV's `d`: the radius is `kernel_size / 2`, or
    /// `cvRound(1.5 * sigma_space)` when `kernel_size <= 0`, and at least 1.
    pub kernel_size: i32,
    /// Spatial sigma (larger = more smoothing)
    pub sigma_space: f32,
    /// Range sigma (larger = less edge preservation, more smoothing).
    ///
    /// For the `u8` paths this is normalised intensity: OpenCV's
    /// `sigmaColor` is `sigma_range * 255`. For depth it is in depth units.
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

/// OpenCV's bilateral kernel: the disc of offsets with
/// `sqrt(dx² + dy²) <= radius` and the spatial weight of each, in OpenCV's
/// `(dy, dx)` raster order.
///
/// * `radius = kernel_size / 2` for `kernel_size > 0`, otherwise
///   `cvRound(1.5 * sigma_space)`; never below 1.
/// * A non-positive `sigma_space` is replaced by 1, as OpenCV does.
/// * Weights are `exp(-r² / (2 sigma_space²))` evaluated in f64 and stored as
///   f32, like OpenCV's `space_weight` table.
fn disc_taps(kernel_size: i32, sigma_space: f32) -> Vec<(i32, i32, f32)> {
    let sigma_space = if sigma_space > 0.0 { sigma_space } else { 1.0 } as f64;
    let radius = if kernel_size <= 0 {
        // cvRound: round half to even.
        (sigma_space * 1.5).round_ties_even() as i32
    } else {
        kernel_size / 2
    }
    .max(1);
    let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    let mut taps = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r = ((dx * dx + dy * dy) as f64).sqrt();
            if r > radius as f64 {
                continue;
            }
            taps.push((dy, dx, (r * r * gauss_space_coeff).exp() as f32));
        }
    }
    taps
}

/// `-0.5 / sigma²`, with OpenCV's replacement of a non-positive sigma by 1.
fn gauss_color_coeff(sigma_color: f32) -> f64 {
    let sigma_color = if sigma_color > 0.0 { sigma_color } else { 1.0 } as f64;
    -0.5 / (sigma_color * sigma_color)
}

/// OpenCV's `borderInterpolate(p, len, BORDER_REFLECT_101)`: `gfedcb|abcdefgh|gfedcba`.
fn reflect_101(mut p: i32, len: i32) -> usize {
    if len == 1 {
        return 0;
    }
    while p < 0 || p >= len {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * len - 2 - p;
        }
    }
    p as usize
}

/// `cv::bilateralFilter` on an 8-bit single-channel image (OpenCV's
/// `bilateralFilter_8u`, `cn = 1`), byte for byte.
///
/// Unlike the [`BilateralFilterParams`] entry points, `sigma_color` is
/// OpenCV's `sigmaColor` in absolute grey levels, not normalised intensity:
/// scaling by 1/255 and back is not exact in `f32` and would move some
/// weights.
///
/// * The kernel is the disc of offsets with `sqrt(dx² + dy²) <= radius`
///   as OpenCV builds it: `radius = d / 2` for `d > 0`, otherwise
///   `cvRound(1.5 * sigma_space)`, and never below 1.
/// * The image is extended with `BORDER_DEFAULT` (reflect-101), so every
///   output pixel averages the full disc.
/// * A non-positive sigma is replaced by 1, as `bilateralFilter_8u` does.
///   (OpenCV 4.13's `cv::bilateralFilter` returns a copy of the input before
///   reaching that rule; this function keeps the documented replacement.)
/// * The range weight is a 256-entry table `exp(-diff² / (2 sigma_color²))`;
///   weights and sums are `f32`, accumulated in OpenCV's tap order, and the
///   mean is rounded with `cvRound` (half to even).
///
/// `cv_photo::bilateral_filter` delegates here.
pub fn bilateral_filter_gray(
    src: &GrayImage,
    d: i32,
    sigma_color: f32,
    sigma_space: f32,
) -> GrayImage {
    let (width, height) = src.dimensions();
    let mut dst = GrayImage::new(width, height);
    if width == 0 || height == 0 {
        return dst;
    }
    let taps = disc_taps(d, sigma_space);
    let color_coeff = gauss_color_coeff(sigma_color);
    // u8 intensities differ by 0..=255, so the range weight is a table.
    let mut color_w = [0.0f32; 256];
    for (diff, w) in color_w.iter_mut().enumerate() {
        let diff = diff as f64;
        *w = (diff * diff * color_coeff).exp() as f32;
    }

    let raw = src.as_raw();
    let stride = width as usize;
    let (w, h) = (width as i32, height as i32);
    dst.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for (x, out) in row.iter_mut().enumerate() {
            let center = raw[y as usize * stride + x];
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for &(dy, dx, space_w) in &taps {
                let ny = reflect_101(y + dy, h);
                let nx = reflect_101(x as i32 + dx, w);
                let val = raw[ny * stride + nx];
                let weight = space_w * color_w[val.abs_diff(center) as usize];
                sum += val as f32 * weight;
                weight_sum += weight;
            }
            // The centre tap has weight `space_w(0) * 1 = 1`, so
            // `weight_sum >= 1`.
            *out = (sum / weight_sum).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    dst
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

/// Depth bilateral filter with `cv::bilateralFilter`'s kernel geometry.
///
/// On a depth map with no invalid samples this is `cv2.bilateralFilter` on a
/// `CV_32FC1` image with `sigmaColor = sigma_range` (in depth units): a disc
/// kernel (see [`disc_taps`]), a reflect-101 border, and the Gaussian of the
/// absolute depth difference as the range weight. OpenCV interpolates that
/// weight from a 4096-bin table spanning the image's value range; this
/// evaluates it directly, which agrees to a few ulp.
///
/// Depth-specific on top of OpenCV: a centre that is `<= 0` or NaN (no
/// measurement) is passed through, and such neighbours are left out of the
/// average.
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
    let taps = disc_taps(params.kernel_size, params.sigma_space);
    let color_coeff = gauss_color_coeff(params.sigma_range);
    let (w, h) = (width as i32, height as i32);

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

                for &(dy, dx, space_w) in &taps {
                    let ny = reflect_101(y + dy, h);
                    let nx = reflect_101(x as i32 + dx, w);
                    let neighbor_val = depth[ny * width as usize + nx];

                    if neighbor_val <= 0.0 || neighbor_val.is_nan() {
                        continue;
                    }

                    let diff = (neighbor_val - center_val).abs() as f64;
                    let range_w = (diff * diff * color_coeff).exp() as f32;
                    let weight = space_w * range_w;
                    sum += neighbor_val * weight;
                    weight_sum += weight;
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

/// `cv::bilateralFilter` on an interleaved 3-channel `u8` image
/// (OpenCV's `bilateralFilter_8u`, `cn = 3`), byte for byte.
///
/// `sigmaColor` is `sigma_range * 255` (the crate's range sigma is normalised
/// intensity). The range weight is OpenCV's: a table over the **L1** colour
/// distance `|d0| + |d1| + |d2|`, `exp(-L1² / (2 sigmaColor²))`. Taps are
/// the disc of [`disc_taps`] over a reflect-101 border, accumulated in f32 in
/// OpenCV's order and rounded with `cvRound`.
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
    let taps = disc_taps(params.kernel_size, params.sigma_space);
    let color_coeff = gauss_color_coeff(params.sigma_range * 255.0);
    // L1 distance over three u8 channels spans 0..=765.
    let color_w: Vec<f32> = (0..3 * 256)
        .map(|i| {
            let d = i as f64;
            (d * d * color_coeff).exp() as f32
        })
        .collect();

    let channels = 3;
    let stride = width as usize * channels;
    let (w, h) = (width as i32, height as i32);

    output
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width as usize {
                let idx = (y as usize * width as usize + x) * channels;
                let c0 = image[idx];
                let c1 = image[idx + 1];
                let c2 = image[idx + 2];

                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut weight_sum = 0.0f32;

                for &(dy, dx, space_w) in &taps {
                    let ny = reflect_101(y + dy, h);
                    let nx = reflect_101(x as i32 + dx, w);
                    let nidx = (ny * width as usize + nx) * channels;
                    let n0 = image[nidx];
                    let n1 = image[nidx + 1];
                    let n2 = image[nidx + 2];
                    let l1 = n0.abs_diff(c0) as usize
                        + n1.abs_diff(c1) as usize
                        + n2.abs_diff(c2) as usize;
                    let weight = space_w * color_w[l1];
                    weight_sum += weight;
                    sum0 += n0 as f32 * weight;
                    sum1 += n1 as f32 * weight;
                    sum2 += n2 as f32 * weight;
                }

                // The centre tap always has weight `space_w(0) * 1 = 1`, so
                // `weight_sum >= 1`.
                let out = &mut row[x * channels..x * channels + channels];
                out[0] = (sum0 / weight_sum).round_ties_even().clamp(0.0, 255.0) as u8;
                out[1] = (sum1 / weight_sum).round_ties_even().clamp(0.0, 255.0) as u8;
                out[2] = (sum2 / weight_sum).round_ties_even().clamp(0.0, 255.0) as u8;
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

/// Joint bilateral filter: depth smoothed with range weights from an RGB
/// guidance image.
///
/// Uses the same kernel geometry as [`bilateral_filter_depth`] and
/// `cv::bilateralFilter` (disc taps from [`disc_taps`], reflect-101 border,
/// radius never below 1); the range weight is the Gaussian of the guidance's
/// L2 colour distance with `sigma = sigma_range * 255`. Invalid depth (`<= 0`
/// or NaN) is passed through at the centre and skipped as a neighbour.
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
    let taps = disc_taps(params.kernel_size, params.sigma_space);
    let color_coeff = gauss_color_coeff(params.sigma_range * 255.0) as f32;
    let (w, h) = (width as i32, height as i32);
    let channels = 3;

    output
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for x in 0..width as usize {
                let idx = y as usize * width as usize + x;
                let center_depth = depth[idx];

                if center_depth <= 0.0 || center_depth.is_nan() {
                    row[x] = center_depth;
                    continue;
                }

                let gidx = idx * channels;
                let center_gi = [
                    guidance[gidx] as f32,
                    guidance[gidx + 1] as f32,
                    guidance[gidx + 2] as f32,
                ];

                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for &(dy, dx, space_w) in &taps {
                    let ny = reflect_101(y + dy, h);
                    let nx = reflect_101(x as i32 + dx, w);
                    let nidx = ny * width as usize + nx;
                    let neighbor_depth = depth[nidx];

                    if neighbor_depth <= 0.0 || neighbor_depth.is_nan() {
                        continue;
                    }

                    let ngidx = nidx * channels;
                    let range_dist_sq = (center_gi[0] - guidance[ngidx] as f32).powi(2)
                        + (center_gi[1] - guidance[ngidx + 1] as f32).powi(2)
                        + (center_gi[2] - guidance[ngidx + 2] as f32).powi(2);
                    let range_weight = (range_dist_sq * color_coeff).exp();

                    let weight = space_w * range_weight;
                    sum += neighbor_depth * weight;
                    weight_sum += weight;
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
    use image::Luma;

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

    /// Spatial exp once per offset, times a 256-entry range exp, accumulated in
    /// the same neighbour order as `exp(space + color)`. u8 output stays identical.
    ///
    /// The reference below follows the same OpenCV semantics as
    /// `bilateral_filter_gray` (disc support, reflect-101 border, `cvRound` radius
    /// clamped to 1); only the weight evaluation differs.
    #[test]
    fn precomputed_exps_match_fused_exp_bytes() {
        let images = [
            GrayImage::from_fn(32, 8, |x, _| Luma([if x < 16 { 100 } else { 101 }])),
            GrayImage::from_fn(17, 13, |x, y| Luma([((x * 13 + y * 7) % 256) as u8])),
            GrayImage::from_fn(9, 9, |x, y| Luma([if (x + y) % 5 == 0 { 0 } else { 255 }])),
        ];
        let params = [
            (5, 50.0, 2.0),
            (5, 30.0, 3.0),
            (-1, 10.0, 5.0),
            (4, 20.0, 15.0),
        ];
        for img in &images {
            for &(d, sc, ss) in &params {
                let fast = bilateral_filter_gray(img, d, sc, ss);
                let fused = fused_exp_bilateral(img, d, sc, ss);
                assert_eq!(
                    fast.as_raw(),
                    fused.as_raw(),
                    "d={d} sigma_color={sc} sigma_space={ss}"
                );
            }
        }
    }

    fn fused_exp_bilateral(
        src: &GrayImage,
        d: i32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> GrayImage {
        let width = src.width() as usize;
        let height = src.height() as usize;
        let mut dst = GrayImage::new(src.width(), src.height());
        let sigma_color = if sigma_color > 0.0 { sigma_color } else { 1.0 };
        let sigma_space = if sigma_space > 0.0 { sigma_space } else { 1.0 };
        let radius = if d <= 0 {
            (sigma_space * 1.5).round_ties_even() as i32
        } else {
            d / 2
        }
        .max(1);
        let gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
        let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
        for y in 0..height {
            for x in 0..width {
                let mut sum_weight = 0.0f32;
                let mut sum_val = 0.0f32;
                let center_val = src.get_pixel(x as u32, y as u32)[0] as f32;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy > radius * radius {
                            continue;
                        }
                        let sy = reflect_101(y as i32 + dy, height as i32);
                        let sx = reflect_101(x as i32 + dx, width as i32);
                        let val = src.get_pixel(sx as u32, sy as u32)[0] as f32;
                        let dist_sq = (dx * dx + dy * dy) as f32;
                        let color_diff = val - center_val;
                        let weight = (dist_sq * gauss_space_coeff
                            + color_diff * color_diff * gauss_color_coeff)
                            .exp();
                        sum_val += val * weight;
                        sum_weight += weight;
                    }
                }
                dst.put_pixel(
                    x as u32,
                    y as u32,
                    Luma([(sum_val / sum_weight).round_ties_even().clamp(0.0, 255.0) as u8]),
                );
            }
        }
        dst
    }

    /// The precomputed disc and colour tables reproduce a direct evaluation of
    /// OpenCV's formula: for every offset with `dx² + dy² <= r²`, sample
    /// through a mirror border (written here independently as a fold), weight
    /// by `exp(-r²/2σs²) * exp(-L1²/2σc²)` and round the weighted mean.
    #[test]
    fn rgb_tables_match_direct_opencv_formula() {
        let (w, h) = (16i32, 12i32);
        let mut image = vec![0u8; (w * h * 3) as usize];
        let mut s = 1u32;
        for p in image.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *p = (s >> 24) as u8;
        }
        for (kernel_size, sigma_space, sigma_range) in [(5, 3.0f32, 0.1f32), (0, 1.5, 0.3)] {
            let params = BilateralFilterParams {
                kernel_size,
                sigma_space,
                sigma_range,
            };
            let mut got = vec![0u8; image.len()];
            bilateral_filter_rgb_internal(&image, &mut got, w as u32, h as u32, params);

            let radius = if kernel_size > 0 {
                kernel_size / 2
            } else {
                (sigma_space as f64 * 1.5).round_ties_even() as i32
            };
            let cs = -0.5 / (sigma_space as f64 * sigma_space as f64);
            let sc = (sigma_range * 255.0) as f64;
            let cc = -0.5 / (sc * sc);
            // Mirror about the first/last sample, without repeating it.
            let fold = |p: i32, n: i32| -> i32 {
                let period = 2 * n - 2;
                let m = p.rem_euclid(period);
                if m < n { m } else { period - m }
            };
            let mut expect = vec![0u8; image.len()];
            for y in 0..h {
                for x in 0..w {
                    let idx = ((y * w + x) * 3) as usize;
                    let mut sums = [0.0f32; 3];
                    let mut weight_sum = 0.0f32;
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let r2 = dx * dx + dy * dy;
                            if r2 > radius * radius {
                                continue;
                            }
                            let nidx = ((fold(y + dy, h) * w + fold(x + dx, w)) * 3) as usize;
                            let l1: i32 = (0..3)
                                .map(|c| (image[nidx + c] as i32 - image[idx + c] as i32).abs())
                                .sum();
                            let ws = (r2 as f64 * cs).exp() as f32;
                            let wc = ((l1 * l1) as f64 * cc).exp() as f32;
                            let weight = ws * wc;
                            weight_sum += weight;
                            for c in 0..3 {
                                sums[c] += image[nidx + c] as f32 * weight;
                            }
                        }
                    }
                    for c in 0..3 {
                        expect[idx + c] =
                            (sums[c] / weight_sum).round_ties_even().clamp(0.0, 255.0) as u8;
                    }
                }
            }
            assert_eq!(got, expect, "kernel_size={kernel_size}");
        }
    }
}
