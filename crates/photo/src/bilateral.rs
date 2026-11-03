use image::GrayImage;

/// Edge-preserving bilateral filter on an 8-bit luma image, matching
/// `cv::bilateralFilter` (OpenCV's `bilateralFilter_8u`).
///
/// `sigma_color` is OpenCV's `sigmaColor`, in absolute grey levels. This is
/// [`cv_imgproc::bilateral_filter_gray`]; the algorithm lives there, next to
/// the colour and depth bilateral filters it shares its kernel with:
///
/// * The kernel is the **disc** of offsets with `sqrt(dx² + dy²) <= radius`,
///   not the square box around it.
/// * The image is extended with OpenCV's `BORDER_DEFAULT` (reflect-101), so
///   every output pixel averages the full disc.
/// * `radius = d / 2` for `d > 0`, otherwise `cvRound(sigma_space * 1.5)`,
///   and never below 1.
/// * A non-positive sigma is replaced by 1, as OpenCV's `bilateralFilter_8u`
///   does.
/// * The output is rounded with `cvRound`, as OpenCV's `saturate_cast` does.
///
/// See `tests/bilateral_opencv_parity.rs` for the OpenCV reference values.
pub fn bilateral_filter(src: &GrayImage, d: i32, sigma_color: f32, sigma_space: f32) -> GrayImage {
    cv_imgproc::bilateral_filter_gray(src, d, sigma_color, sigma_space)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

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
