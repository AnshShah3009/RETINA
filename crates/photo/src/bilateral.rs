use image::{GrayImage, Luma};

/// Edge-preserving bilateral filter on an 8-bit luma image, matching
/// `cv::bilateralFilter` (OpenCV's `bilateralFilter_8u`).
///
/// * The kernel is the **disc** of offsets with `sqrt(dx² + dy²) <= radius`,
///   not the square box around it.
/// * The image is extended with OpenCV's `BORDER_DEFAULT` (reflect-101), so
///   every output pixel averages the full disc.
/// * `radius = d / 2` for `d > 0`, otherwise `cvRound(sigma_space * 1.5)`.
/// * A non-positive sigma is replaced by 1, as OpenCV does.
///
/// The previous version used the square window, skipped out-of-image taps
/// (renormalising over fewer neighbours) and took `ceil` of
/// `1.5 * sigma_space`. On an 11x9 textured image it was off OpenCV by up to
/// 16 grey levels in the interior and 26 at the border; see
/// `tests/bilateral_opencv_parity.rs`.
pub fn bilateral_filter(src: &GrayImage, d: i32, sigma_color: f32, sigma_space: f32) -> GrayImage {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut dst = GrayImage::new(src.width(), src.height());
    if width == 0 || height == 0 {
        return dst;
    }

    // Guard degenerate sigmas: sigma == 0 makes the coefficient -inf, and
    // `0 * -inf` is NaN for any neighbour equal to the centre. OpenCV
    // replaces a non-positive sigma with 1.
    let sigma_color = if sigma_color > 0.0 { sigma_color } else { 1.0 } as f64;
    let sigma_space = if sigma_space > 0.0 { sigma_space } else { 1.0 } as f64;

    let radius = if d <= 0 {
        // cvRound: round half to even.
        (sigma_space * 1.5).round_ties_even() as i32
    } else {
        d / 2
    }
    .max(1);

    let gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    // Disc of offsets and their spatial weights, computed in f64 and stored
    // as f32 like OpenCV's tables.
    let mut taps: Vec<(i32, i32, f32)> = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r = ((dx * dx + dy * dy) as f64).sqrt();
            if r > radius as f64 {
                continue;
            }
            taps.push((dy, dx, (r * r * gauss_space_coeff).exp() as f32));
        }
    }
    // u8 intensities differ by 0..=255, so the range weight is a table.
    let mut color_w = [0.0f32; 256];
    for (diff, w) in color_w.iter_mut().enumerate() {
        let d = diff as f64;
        *w = (d * d * gauss_color_coeff).exp() as f32;
    }

    let raw = src.as_raw();
    for y in 0..height {
        for x in 0..width {
            let mut sum_weight = 0.0f32;
            let mut sum_val = 0.0f32;
            let center = raw[y * width + x];

            for &(dy, dx, space_w) in &taps {
                let sy = reflect_101(y as i32 + dy, height as i32);
                let sx = reflect_101(x as i32 + dx, width as i32);
                let val = raw[sy * width + sx];
                let weight = space_w * color_w[val.abs_diff(center) as usize];
                sum_val += val as f32 * weight;
                sum_weight += weight;
            }

            dst.put_pixel(
                x as u32,
                y as u32,
                // Round, as OpenCV's saturate_cast does; truncation biased
                // every output ~0.5 grey level dark (127.9 -> 127).
                Luma([(sum_val / sum_weight).round().clamp(0.0, 255.0) as u8]),
            );
        }
    }

    dst
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilateral_filter_output_dimensions() {
        let img = GrayImage::new(5, 7);
        let filtered = bilateral_filter(&img, 5, 10.0, 10.0);

        assert_eq!(filtered.width(), img.width());
        assert_eq!(filtered.height(), img.height());
    }

    #[test]
    fn test_bilateral_filter_preserves_uniform_image() {
        let mut img = GrayImage::new(5, 5);
        for pixel in img.iter_mut() {
            *pixel = 128;
        }

        let filtered = bilateral_filter(&img, 5, 10.0, 10.0);

        // Uniform image should remain unchanged
        for pixel in filtered.iter() {
            assert_eq!(*pixel, 128);
        }
    }

    #[test]
    fn test_bilateral_filter_range_check() {
        let mut img = GrayImage::new(5, 5);
        // Create an image with varying values
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([50 + (x + y) as u8 * 10]));
            }
        }

        let filtered = bilateral_filter(&img, 5, 10.0, 10.0);

        // Output image has same dimensions (values are u8, guaranteed 0..=255)
        assert_eq!(filtered.len(), (5 * 5) as usize);
    }

    #[test]
    fn test_bilateral_filter_with_auto_radius() {
        let img = GrayImage::new(5, 5);
        let filtered = bilateral_filter(&img, -1, 10.0, 5.0);

        // Auto radius should work (d <= 0)
        assert_eq!(filtered.width(), img.width());
        assert_eq!(filtered.height(), img.height());
    }

    #[test]
    fn test_bilateral_filter_with_explicit_radius() {
        let img = GrayImage::new(7, 7);
        let filtered = bilateral_filter(&img, 4, 20.0, 15.0);

        // Explicit radius should work (d > 0)
        assert_eq!(filtered.width(), img.width());
        assert_eq!(filtered.height(), img.height());
    }

    #[test]
    fn test_bilateral_filter_zero_sigma_is_sane() {
        // Regression: sigma_color == 0 (or sigma_space == 0) divided by zero,
        // producing -inf coefficients and NaN weights; every output pixel then
        // came back as 255 because `NaN.min(255.0) == 255.0`.
        let mut img = GrayImage::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([(20 + x * 10 + y * 5) as u8]));
            }
        }

        for (sigma_color, sigma_space) in [(0.0f32, 5.0f32), (5.0, 0.0), (0.0, 0.0)] {
            let filtered = bilateral_filter(&img, 5, sigma_color, sigma_space);
            assert!(
                filtered.as_raw().iter().all(|&p| p != 255),
                "zero sigma produced garbage output for ({sigma_color}, {sigma_space})"
            );
            // A constant image must stay untouched with any sigma.
            let flat = GrayImage::from_pixel(5, 5, Luma([128]));
            let filtered_flat = bilateral_filter(&flat, 5, sigma_color, sigma_space);
            assert!(filtered_flat.as_raw().iter().all(|&p| p == 128));
        }
    }

    #[test]
    fn test_bilateral_filter_single_pixel() {
        let mut img = GrayImage::new(1, 1);
        img.put_pixel(0, 0, Luma([200]));

        let filtered = bilateral_filter(&img, 3, 50.0, 50.0);

        // Single pixel should not change
        assert_eq!(filtered.get_pixel(0, 0)[0], 200);
    }

    /// Spatial exp once per offset, times a 256-entry range exp, accumulated in
    /// the same neighbour order as `exp(space + color)`. u8 output stays identical.
    ///
    /// The reference below follows the same OpenCV semantics as
    /// `bilateral_filter` (disc support, reflect-101 border, `cvRound` radius
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
                let fast = bilateral_filter(img, d, sc, ss);
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
                    Luma([(sum_val / sum_weight).round().clamp(0.0, 255.0) as u8]),
                );
            }
        }
        dst
    }

    #[test]
    fn test_bilateral_filter_noise_reduction() {
        let mut img = GrayImage::new(5, 5);
        // Create base image with value 128
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([128]));
            }
        }
        // Add salt-and-pepper noise at center
        img.put_pixel(2, 2, Luma([255]));
        img.put_pixel(2, 3, Luma([0]));

        let filtered = bilateral_filter(&img, 5, 30.0, 30.0);

        // Center pixels should be smoothed toward uniform value
        let center = filtered.get_pixel(2, 2)[0] as i32;
        let noise_val = 255i32;
        let diff = (center - noise_val).abs();
        // After filtering, should be significantly reduced
        assert!(diff < 100);
    }
}
