use crate::KeyPoints;
use cv_core::KeyPoint;
use image::GrayImage;
use rayon::prelude::*;

/// Detect corners using the Harris corner detector.
///
/// Computes the Harris response `det(M) - k * trace(M)^2` at each pixel using
/// a 3×3 Sobel gradient window and returns keypoints whose response exceeds `threshold`.
///
/// * `k` - Harris sensitivity parameter (typically 0.04–0.06)
/// * `threshold` - Minimum response value for a pixel to be returned as a keypoint
pub fn harris_detect(
    image: &GrayImage,
    _block_size: i32,
    _ksize: i32,
    k: f64,
    threshold: f64,
) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;

    let half_window = 1i32;

    let kps: Vec<KeyPoint> = (half_window + 1..height - half_window - 1)
        .into_par_iter()
        .flat_map(|y| {
            let mut row_kps = Vec::new();
            for x in (half_window + 1)..(width - half_window - 1) {
                let mut i_xx = 0.0f64;
                let mut i_yy = 0.0f64;
                let mut i_xy = 0.0f64;

                for by in -half_window..=half_window {
                    for bx in -half_window..=half_window {
                        let gx = image.get_pixel((x + bx + 1) as u32, (y + by) as u32)[0] as f64
                            - image.get_pixel((x + bx - 1) as u32, (y + by) as u32)[0] as f64;
                        let gy = image.get_pixel((x + bx) as u32, (y + by + 1) as u32)[0] as f64
                            - image.get_pixel((x + bx) as u32, (y + by - 1) as u32)[0] as f64;

                        i_xx += gx * gx;
                        i_yy += gy * gy;
                        i_xy += gx * gy;
                    }
                }

                let det = i_xx * i_yy - i_xy * i_xy;
                let trace = i_xx + i_yy;
                let response = det - k * trace * trace;

                if response > threshold {
                    let kp = KeyPoint::new(x as f64, y as f64).with_response(response);
                    row_kps.push(kp);
                }
            }
            row_kps
        })
        .collect();

    KeyPoints { keypoints: kps }
}

/// Detect corners using the Shi-Tomasi (Good Features to Track) criterion.
///
/// Delegates to [`harris_detect`] with `k = 0.04` and a threshold derived from
/// `quality_level * 255^2`.
pub fn shi_tomasi_detect(
    image: &GrayImage,
    _max_corners: usize,
    quality_level: f64,
    _min_distance: f64,
) -> KeyPoints {
    harris_detect(image, 3, 3, 0.04, quality_level * 255.0 * 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_image_with_corners() -> GrayImage {
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                let val = if (x < 5 && y < 5)
                    || (x > 14 && y < 5)
                    || (x < 5 && y > 14)
                    || (x > 14 && y > 14)
                {
                    255
                } else {
                    0
                };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }

    fn create_uniform_image() -> GrayImage {
        GrayImage::from_pixel(20, 20, Luma([128]))
    }

    #[test]
    fn test_harris_detect_finds_corners() {
        let img = create_test_image_with_corners();
        let kps = harris_detect(&img, 3, 3, 0.04, 1000.0);
        assert!(
            !kps.keypoints.is_empty(),
            "Should detect corners in image with corners"
        );
    }

    #[test]
    fn test_harris_detect_uniform_image() {
        let img = create_uniform_image();
        let kps = harris_detect(&img, 3, 3, 0.04, 1000.0);
        assert!(
            kps.keypoints.is_empty(),
            "Uniform image should have no corners above threshold"
        );
    }

    #[test]
    fn test_harris_detect_keypoint_properties() {
        let img = create_test_image_with_corners();
        let kps = harris_detect(&img, 3, 3, 0.04, 1000.0);

        for kp in &kps.keypoints {
            assert!(kp.x >= 0.0 && kp.x < 20.0);
            assert!(kp.y >= 0.0 && kp.y < 20.0);
            assert!(kp.response > 1000.0);
        }
    }

    #[test]
    fn test_shi_tomasi_detect() {
        let img = create_test_image_with_corners();
        let kps = shi_tomasi_detect(&img, 100, 0.01, 1.0);
        assert!(
            !kps.keypoints.is_empty(),
            "Shi-Tomasi should detect corners"
        );
    }

    #[test]
    fn test_harris_detect_low_threshold() {
        let img = create_test_image_with_corners();
        let kps_low = harris_detect(&img, 3, 3, 0.04, 10.0);
        let kps_high = harris_detect(&img, 3, 3, 0.04, 10000.0);
        assert!(kps_low.keypoints.len() >= kps_high.keypoints.len());
    }

    #[test]
    fn test_harris_detect_k_parameter() {
        let img = create_test_image_with_corners();
        let kps1 = harris_detect(&img, 3, 3, 0.04, 1000.0);
        let kps2 = harris_detect(&img, 3, 3, 0.06, 1000.0);
        assert!(!kps1.keypoints.is_empty() || !kps2.keypoints.is_empty());
    }
}
