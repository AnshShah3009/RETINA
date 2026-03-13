//! Image inpainting algorithms
//!
//! Provides two methods for filling in masked regions of an image:
//! - **Telea (Fast Marching Method):** Propagates known pixel values into the unknown region
//!   in order of increasing distance from the boundary, using gradient-weighted averages.
//! - **Navier-Stokes:** Iterative PDE-based diffusion that propagates isophote (edge) directions
//!   into the inpainting region.

use cv_core::float::Float;
use cv_core::tensor::{CpuTensor, TensorShape};
use cv_core::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Status flags for Fast Marching Method pixels.
const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;

/// A pixel entry in the Fast Marching narrow band, ordered by distance.
#[derive(Clone, Copy)]
struct FmmEntry {
    dist: f64,
    y: usize,
    x: usize,
}

impl PartialEq for FmmEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for FmmEntry {}

impl PartialOrd for FmmEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FmmEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering so smallest distance is popped first.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Telea inpainting using the Fast Marching Method.
///
/// Pixels where `mask` is non-zero are treated as the inpainting region.
/// Known pixels are propagated inward, weighted by distance, direction
/// (gradient alignment), and level-set proximity.
///
/// # Arguments
/// * `image` - Input image tensor (CHW layout, any number of channels).
/// * `mask`  - Single-channel mask (1, H, W). Non-zero values mark pixels to inpaint.
/// * `radius` - Neighbourhood radius for the weighted average.
///
/// # Returns
/// A new tensor with the masked region filled in.
pub fn inpaint_telea<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    mask: &CpuTensor<u8>,
    radius: f32,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();
    let (mc, mh, mw) = mask.shape.chw();

    if mh != height || mw != width {
        return Err(cv_core::Error::DimensionMismatch(
            "Mask dimensions must match image height and width".into(),
        ));
    }
    if mc != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Mask must be single-channel (1, H, W)".into(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(cv_core::Error::InvalidInput(
            "Image dimensions must be non-zero".into(),
        ));
    }

    let mask_data = mask.as_slice()?;
    let src_data = image.as_slice()?;

    // Work in f64 for precision.
    let mut result: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            let offset = c * height * width;
            src_data[offset..offset + height * width]
                .iter()
                .map(|v| Float::to_f64(*v))
                .collect()
        })
        .collect();

    // Distance map and flags.
    let n = height * width;
    let mut dist = vec![f64::MAX; n];
    let mut flags = vec![INSIDE; n];
    let mut heap = BinaryHeap::new();

    let r = (radius.ceil() as usize).max(1);

    // Initialize: mark KNOWN pixels and find BAND (boundary of inpaint region).
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask_data[idx] == 0 {
                flags[idx] = KNOWN;
                dist[idx] = 0.0;
            }
        }
    }

    // Band: INSIDE pixels adjacent to KNOWN pixels.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if flags[idx] == INSIDE {
                let mut near_known = false;
                for &(dy, dx) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;
                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let nidx = ny as usize * width + nx as usize;
                        if flags[nidx] == KNOWN {
                            near_known = true;
                            break;
                        }
                    }
                }
                if near_known {
                    flags[idx] = BAND;
                    dist[idx] = 1.0;
                    heap.push(FmmEntry { dist: 1.0, y, x });
                }
            }
        }
    }

    // Fast Marching: process band pixels in order of increasing distance.
    while let Some(entry) = heap.pop() {
        let idx = entry.y * width + entry.x;

        // Skip if already processed with a shorter distance.
        if flags[idx] == KNOWN {
            continue;
        }

        flags[idx] = KNOWN;
        dist[idx] = entry.dist;

        // Inpaint this pixel: weighted average of known neighbours within radius.
        for c in 0..channels {
            let mut sum_w = 0.0f64;
            let mut sum_v = 0.0f64;

            let y0 = entry.y.saturating_sub(r);
            let y1 = (entry.y + r + 1).min(height);
            let x0 = entry.x.saturating_sub(r);
            let x1 = (entry.x + r + 1).min(width);

            // Gradient estimation at this pixel (from distance map).
            let grad_y = if entry.y > 0 && entry.y + 1 < height {
                (dist[idx + width] - dist[idx - width]) * 0.5
            } else {
                0.0
            };
            let grad_x = if entry.x > 0 && entry.x + 1 < width {
                (dist[idx + 1] - dist[idx - 1]) * 0.5
            } else {
                0.0
            };

            for ny in y0..y1 {
                for nx in x0..x1 {
                    let nidx = ny * width + nx;
                    if flags[nidx] != KNOWN || (ny == entry.y && nx == entry.x) {
                        continue;
                    }
                    let d = dist[nidx];
                    if d >= f64::MAX * 0.5 {
                        continue;
                    }

                    let dy = ny as f64 - entry.y as f64;
                    let dx = nx as f64 - entry.x as f64;
                    let geom_dist = (dy * dy + dx * dx).sqrt();
                    if geom_dist > radius as f64 {
                        continue;
                    }

                    // Weight: 1/distance, direction alignment, level-set closeness.
                    let w_dist = 1.0 / (geom_dist + 1e-6);

                    // Direction factor: alignment of vector to neighbor with gradient.
                    let dir_len = (grad_y * grad_y + grad_x * grad_x).sqrt() + 1e-6;
                    let w_dir = ((dy * grad_y + dx * grad_x) / (geom_dist * dir_len))
                        .abs()
                        .max(0.01);

                    // Level-set factor: prefer neighbors with similar distance.
                    let w_level = 1.0 / (1.0 + (d - entry.dist).abs());

                    let w = w_dist * w_dir * w_level;
                    sum_w += w;
                    sum_v += w * result[c][nidx];
                }
            }

            if sum_w > 0.0 {
                result[c][idx] = sum_v / sum_w;
            }
        }

        // Add INSIDE neighbours to the band.
        for &(dy, dx) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
            let ny = entry.y as i32 + dy;
            let nx = entry.x as i32 + dx;
            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                let nidx = ny as usize * width + nx as usize;
                if flags[nidx] == INSIDE {
                    let new_dist = entry.dist + 1.0;
                    if new_dist < dist[nidx] {
                        dist[nidx] = new_dist;
                        flags[nidx] = BAND;
                        heap.push(FmmEntry {
                            dist: new_dist,
                            y: ny as usize,
                            x: nx as usize,
                        });
                    }
                }
            }
        }
    }

    // Reconstruct output tensor.
    let mut out_data = Vec::with_capacity(channels * height * width);
    for c in 0..channels {
        for val in &result[c] {
            out_data.push(T::from_f64(*val));
        }
    }

    CpuTensor::<T>::from_vec(out_data, TensorShape::new(channels, height, width))
}

/// Navier-Stokes inpainting using iterative PDE diffusion.
///
/// Propagates image structure (isophote directions) into the masked region
/// by solving a simplified Navier-Stokes-like equation iteratively.
///
/// # Arguments
/// * `image`      - Input image tensor (CHW layout).
/// * `mask`       - Single-channel mask (1, H, W). Non-zero = inpaint region.
/// * `radius`     - Not used for iteration but controls initial smoothing extent.
/// * `iterations` - Number of diffusion iterations.
///
/// # Returns
/// A new tensor with the masked region filled by PDE diffusion.
pub fn inpaint_ns<T: Float + Default + 'static>(
    image: &CpuTensor<T>,
    mask: &CpuTensor<u8>,
    _radius: f32,
    iterations: u32,
) -> Result<CpuTensor<T>> {
    let (channels, height, width) = image.shape.chw();
    let (mc, mh, mw) = mask.shape.chw();

    if mh != height || mw != width {
        return Err(cv_core::Error::DimensionMismatch(
            "Mask dimensions must match image height and width".into(),
        ));
    }
    if mc != 1 {
        return Err(cv_core::Error::InvalidInput(
            "Mask must be single-channel (1, H, W)".into(),
        ));
    }
    if height < 3 || width < 3 {
        return Err(cv_core::Error::InvalidInput(
            "Image must be at least 3x3 for Navier-Stokes inpainting".into(),
        ));
    }

    let mask_data = mask.as_slice()?;
    let src_data = image.as_slice()?;

    // Work per-channel in f64.
    let mut result: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            let offset = c * height * width;
            src_data[offset..offset + height * width]
                .iter()
                .map(|v| Float::to_f64(*v))
                .collect()
        })
        .collect();

    // Build a boolean mask: true = needs inpainting.
    let inpaint_mask: Vec<bool> = mask_data.iter().map(|&v| v != 0).collect();

    // Initialize masked pixels with average of known neighbors (simple seed).
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if inpaint_mask[idx] {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &(dy, dx) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                            let nidx = ny as usize * width + nx as usize;
                            if !inpaint_mask[nidx] {
                                sum += result[c][nidx];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        result[c][idx] = sum / count as f64;
                    }
                }
            }
        }
    }

    let dt = 0.1; // Time step for diffusion.

    for _iter in 0..iterations {
        for c in 0..channels {
            let prev = result[c].clone();

            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = y * width + x;
                    if !inpaint_mask[idx] {
                        continue;
                    }

                    // Laplacian of the image (isotropic diffusion).
                    let laplacian =
                        prev[idx - width] + prev[idx + width] + prev[idx - 1] + prev[idx + 1]
                            - 4.0 * prev[idx];

                    // Gradient (isophote direction).
                    let gy = (prev[idx + width] - prev[idx - width]) * 0.5;
                    let gx = (prev[idx + 1] - prev[idx - 1]) * 0.5;
                    let g_mag = (gx * gx + gy * gy).sqrt() + 1e-10;

                    // Smoothness Laplacian (second derivative of the smoothness field).
                    // Approximate Laplacian of the Laplacian for NS-like behavior.
                    let lap_up = prev[(y - 1) * width + x - 1]
                        + prev[(y - 1) * width + x + 1]
                        + prev[idx - width]
                        - 3.0 * prev[(y - 1) * width + x];
                    let lap_down = prev[(y + 1) * width + x - 1]
                        + prev[(y + 1) * width + x + 1]
                        + prev[idx + width]
                        - 3.0 * prev[(y + 1) * width + x];
                    let lap_left = prev[(y - 1) * width + x - 1]
                        + prev[(y + 1) * width + x - 1]
                        + prev[idx - 1]
                        - 3.0 * prev[y * width + x - 1];
                    let lap_right = prev[(y - 1) * width + x + 1]
                        + prev[(y + 1) * width + x + 1]
                        + prev[idx + 1]
                        - 3.0 * prev[y * width + x + 1];

                    let smooth_grad_y = (lap_down - lap_up) * 0.5;
                    let smooth_grad_x = (lap_right - lap_left) * 0.5;

                    // Project smoothness gradient onto isophote direction (perpendicular to gradient).
                    // Isophote direction: (-gy, gx) / |g|.
                    let projection = (-gy * smooth_grad_x + gx * smooth_grad_y) / g_mag;

                    // Update: combine Laplacian diffusion + NS correction.
                    result[c][idx] = prev[idx] + dt * (laplacian + projection);
                }
            }
        }
    }

    // Reconstruct output tensor.
    let mut out_data = Vec::with_capacity(channels * height * width);
    for c in 0..channels {
        for val in &result[c] {
            out_data.push(T::from_f64(*val));
        }
    }

    CpuTensor::<T>::from_vec(out_data, TensorShape::new(channels, height, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::tensor::TensorShape;

    fn make_test_image(height: usize, width: usize, value: f32) -> CpuTensor<f32> {
        let data = vec![value; height * width];
        CpuTensor::<f32>::from_vec(data, TensorShape::new(1, height, width)).unwrap()
    }

    fn make_mask(height: usize, width: usize, masked: &[(usize, usize)]) -> CpuTensor<u8> {
        let mut data = vec![0u8; height * width];
        for &(y, x) in masked {
            data[y * width + x] = 255;
        }
        CpuTensor::<u8>::from_vec(data, TensorShape::new(1, height, width)).unwrap()
    }

    #[test]
    fn test_telea_no_mask_preserves_image() {
        let img = make_test_image(5, 5, 0.5);
        let mask = make_mask(5, 5, &[]);
        let result = inpaint_telea(&img, &mask, 3.0).unwrap();
        let data = result.as_slice().unwrap();
        for &v in data {
            assert!((v - 0.5).abs() < 1e-6, "Expected 0.5, got {}", v);
        }
    }

    #[test]
    fn test_telea_fills_center_pixel() {
        // 5x5 image with uniform value, mask center pixel.
        let data: Vec<f32> = (0..25).map(|_| 0.8).collect();
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 5, 5)).unwrap();
        let mask = make_mask(5, 5, &[(2, 2)]);
        let result = inpaint_telea(&img, &mask, 3.0).unwrap();
        let out = result.as_slice().unwrap();
        let center = out[2 * 5 + 2];
        // Center should be close to 0.8 since all neighbors are 0.8.
        assert!(
            (center - 0.8).abs() < 0.1,
            "Center pixel should be ~0.8, got {}",
            center
        );
    }

    #[test]
    fn test_telea_fills_smoothly() {
        // Gradient image: left=0.0, right=1.0. Mask a column in the middle.
        let h = 5;
        let w = 10;
        let mut data = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = x as f32 / (w - 1) as f32;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, h, w)).unwrap();
        // Mask column 5.
        let masked: Vec<(usize, usize)> = (0..h).map(|y| (y, 5)).collect();
        let mask = make_mask(h, w, &masked);
        let result = inpaint_telea(&img, &mask, 3.0).unwrap();
        let out = result.as_slice().unwrap();
        // The inpainted values at x=5 should be between values at x=4 and x=6.
        for y in 0..h {
            let v4 = out[y * w + 4];
            let v5 = out[y * w + 5];
            let v6 = out[y * w + 6];
            assert!(
                v5 >= v4 - 0.15 && v5 <= v6 + 0.15,
                "Row {}: v4={}, v5={}, v6={}",
                y,
                v4,
                v5,
                v6
            );
        }
    }

    #[test]
    fn test_telea_dimension_mismatch() {
        let img = make_test_image(5, 5, 0.5);
        let mask = make_mask(3, 3, &[]);
        let result = inpaint_telea(&img, &mask, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ns_no_mask_preserves_image() {
        let img = make_test_image(5, 5, 0.5);
        let mask = make_mask(5, 5, &[]);
        let result = inpaint_ns(&img, &mask, 3.0, 100).unwrap();
        let data = result.as_slice().unwrap();
        for &v in data {
            assert!((v - 0.5).abs() < 1e-5, "Expected 0.5, got {}", v);
        }
    }

    #[test]
    fn test_ns_fills_center_pixel() {
        let data: Vec<f32> = (0..25).map(|_| 0.8).collect();
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(1, 5, 5)).unwrap();
        let mask = make_mask(5, 5, &[(2, 2)]);
        let result = inpaint_ns(&img, &mask, 3.0, 200).unwrap();
        let out = result.as_slice().unwrap();
        let center = out[2 * 5 + 2];
        assert!(
            (center - 0.8).abs() < 0.15,
            "Center pixel should be ~0.8, got {}",
            center
        );
    }

    #[test]
    fn test_ns_dimension_mismatch() {
        let img = make_test_image(5, 5, 0.5);
        let mask = make_mask(3, 3, &[]);
        let result = inpaint_ns(&img, &mask, 3.0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_ns_too_small_image() {
        let img = make_test_image(2, 2, 0.5);
        let mask = make_mask(2, 2, &[]);
        let result = inpaint_ns(&img, &mask, 3.0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_telea_multichannel() {
        // 3-channel image, mask center.
        let h = 5;
        let w = 5;
        let mut data = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for i in 0..h * w {
                data[c * h * w + i] = (c + 1) as f32 * 0.25;
            }
        }
        let img = CpuTensor::<f32>::from_vec(data, TensorShape::new(3, h, w)).unwrap();
        let mask = make_mask(h, w, &[(2, 2)]);
        let result = inpaint_telea(&img, &mask, 3.0).unwrap();
        let out = result.as_slice().unwrap();
        for c in 0..3 {
            let expected = (c + 1) as f32 * 0.25;
            let center = out[c * h * w + 2 * w + 2];
            assert!(
                (center - expected).abs() < 0.1,
                "Channel {} center should be ~{}, got {}",
                c,
                expected,
                center
            );
        }
    }
}
