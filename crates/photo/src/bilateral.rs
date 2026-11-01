use image::{GrayImage, Luma};

pub fn bilateral_filter(src: &GrayImage, d: i32, sigma_color: f32, sigma_space: f32) -> GrayImage {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut dst = GrayImage::new(src.width(), src.height());

    // Guard degenerate sigmas: sigma == 0 makes the coefficient -inf, and
    // `0 * -inf` is NaN for any neighbour equal to the centre, so the whole
    // image came back as garbage (`NaN.min(255.0) as u8 == 255`). OpenCV
    // derives sigma from the kernel size when it is not positive; fall back to
    // unit sigma here (same guard as the HAL bilateral filter).
    let sigma_color = if sigma_color > 0.0 { sigma_color } else { 1.0 };
    let sigma_space = if sigma_space > 0.0 { sigma_space } else { 1.0 };

    let radius = if d <= 0 {
        (sigma_space * 1.5).ceil() as i32
    } else {
        d / 2
    };

    let gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    // Spatial weight depends only on the kernel offset. u8 intensities differ
    // by 0..=255, so the range weight is a 256-entry table of the same exp.
    let k = (2 * radius + 1) as usize;
    let mut space_w = vec![0.0f32; k * k];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let dist_sq = (dx * dx + dy * dy) as f32;
            let iy = (dy + radius) as usize;
            let ix = (dx + radius) as usize;
            space_w[iy * k + ix] = (dist_sq * gauss_space_coeff).exp();
        }
    }
    let mut color_w = [0.0f32; 256];
    for diff in 0..256 {
        let d = diff as f32;
        color_w[diff] = (d * d * gauss_color_coeff).exp();
    }

    for y in 0..height {
        for x in 0..width {
            let mut sum_weight = 0.0f32;
            let mut sum_val = 0.0f32;
            let center_u8 = src.get_pixel(x as u32, y as u32)[0];

            for dy in -radius..=radius {
                let sy = y as i32 + dy;
                if sy < 0 || sy >= height as i32 {
                    continue;
                }

                for dx in -radius..=radius {
                    let sx = x as i32 + dx;
                    if sx < 0 || sx >= width as i32 {
                        continue;
                    }

                    let val_u8 = src.get_pixel(sx as u32, sy as u32)[0];
                    let val = val_u8 as f32;
                    let diff = (val_u8 as i32 - center_u8 as i32).unsigned_abs() as usize;
                    let weight = space_w[(dy + radius) as usize * k + (dx + radius) as usize]
                        * color_w[diff];

                    sum_val += val * weight;
                    sum_weight += weight;
                }
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
            (sigma_space * 1.5).ceil() as i32
        } else {
            d / 2
        };
        let gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
        let gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
        for y in 0..height {
            for x in 0..width {
                let mut sum_weight = 0.0f32;
                let mut sum_val = 0.0f32;
                let center_val = src.get_pixel(x as u32, y as u32)[0] as f32;
                for dy in -radius..=radius {
                    let sy = y as i32 + dy;
                    if sy < 0 || sy >= height as i32 {
                        continue;
                    }
                    for dx in -radius..=radius {
                        let sx = x as i32 + dx;
                        if sx < 0 || sx >= width as i32 {
                            continue;
                        }
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
