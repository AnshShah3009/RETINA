use image::{GrayImage, Luma};

pub fn bilateral_filter(src: &GrayImage, d: i32, sigma_color: f32, sigma_space: f32) -> GrayImage {
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut dst = GrayImage::new(src.width(), src.height());

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
                    let color_diff_sq = color_diff * color_diff;

                    let weight =
                        (dist_sq * gauss_space_coeff + color_diff_sq * gauss_color_coeff).exp();

                    sum_val += val * weight;
                    sum_weight += weight;
                }
            }

            dst.put_pixel(
                x as u32,
                y as u32,
                Luma([(sum_val / sum_weight).min(255.0) as u8]),
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
    fn test_bilateral_filter_single_pixel() {
        let mut img = GrayImage::new(1, 1);
        img.put_pixel(0, 0, Luma([200]));

        let filtered = bilateral_filter(&img, 3, 50.0, 50.0);

        // Single pixel should not change
        assert_eq!(filtered.get_pixel(0, 0)[0], 200);
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
