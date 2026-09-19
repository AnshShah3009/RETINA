use crate::KeyPoints;
use cv_core::KeyPoint;
use image::GrayImage;
use rayon::prelude::*;

use crate::gradients::{compute_sobel_gradients, gaussian_blur_f64, gaussian_kernel_1d};

/// Detect corners using the Harris corner detector.
///
/// Computes the Harris response `det(M) - k * trace(M)^2` at each pixel where
/// the structure tensor `M` is Gaussian-weighted over a window of `block_size`.
/// Sobel gradients use aperture size `ksize`.
///
/// Non-maximum suppression (3x3) is applied so only local maxima are returned.
///
/// * `block_size` - Size of the Gaussian window for the structure tensor (must be odd, >= 3)
/// * `ksize` - Sobel operator aperture size (3, 5, or 7)
/// * `k` - Harris sensitivity parameter (typically 0.04-0.06)
/// * `threshold` - Minimum response value for a pixel to be returned as a keypoint
pub fn harris_detect(
    image: &GrayImage,
    block_size: i32,
    ksize: i32,
    k: f64,
    threshold: f64,
) -> KeyPoints {
    let width = image.width() as usize;
    let height = image.height() as usize;

    // Step 1: Compute Sobel gradients Ix, Iy
    let (ix, iy) = compute_sobel_gradients(image, ksize);

    // Step 2: Compute gradient products
    let n = width * height;
    let mut ixx = Vec::with_capacity(n);
    let mut iyy = Vec::with_capacity(n);
    let mut ixy = Vec::with_capacity(n);
    for i in 0..n {
        ixx.push(ix[i] * ix[i]);
        iyy.push(iy[i] * iy[i]);
        ixy.push(ix[i] * iy[i]);
    }

    // Step 3: Gaussian blur each product image with sigma derived from block_size
    let bs = (block_size.max(3) | 1) as usize; // ensure odd and >= 3
    let sigma = bs as f64 * 0.5;
    let gauss = gaussian_kernel_1d(bs, sigma);

    let sxx = gaussian_blur_f64(&ixx, width, height, &gauss);
    let syy = gaussian_blur_f64(&iyy, width, height, &gauss);
    let sxy = gaussian_blur_f64(&ixy, width, height, &gauss);

    // Step 4: Compute Harris response at each pixel
    let mut response = vec![0.0f64; n];
    for i in 0..n {
        let det = sxx[i] * syy[i] - sxy[i] * sxy[i];
        let trace = sxx[i] + syy[i];
        response[i] = det - k * trace * trace;
    }

    // Step 5: Non-maximum suppression (3x3) + threshold
    let kps: Vec<KeyPoint> = (1..height.saturating_sub(1))
        .into_par_iter()
        .flat_map(|y| {
            let mut row_kps = Vec::new();
            for x in 1..width.saturating_sub(1) {
                let r = response[y * width + x];
                if r <= threshold {
                    continue;
                }
                // Check 3x3 neighbourhood: must be strictly greater than all neighbours
                let mut is_max = true;
                'nms: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        if response[ny * width + nx] >= r {
                            is_max = false;
                            break 'nms;
                        }
                    }
                }
                if is_max {
                    row_kps.push(KeyPoint::new(x as f64, y as f64).with_response(r));
                }
            }
            row_kps
        })
        .collect();

    KeyPoints { keypoints: kps }
}

/// Detect corners using the Shi-Tomasi (Good Features to Track) criterion.
///
/// Uses the minimum eigenvalue `R = min(lambda_1, lambda_2)` of the structure tensor
/// as corner response. Returns at most `max_corners` keypoints with a minimum quality of
/// `quality_level` (fraction of the strongest response) and spaced at least
/// `min_distance` pixels apart.
pub fn shi_tomasi_detect(
    image: &GrayImage,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
) -> KeyPoints {
    crate::gftt::gftt_detect(image, max_corners, quality_level, min_distance)
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

        // The four white blocks are at corners (0-4,0-4), (15-19,0-4),
        // (0-4,15-19), (15-19,15-19). The block junctions where all four
        // quadrants meet are approximately at (5, 5), (14, 5), (5, 14), (14, 14).
        let expected_corners: [(f64, f64); 4] =
            [(5.0, 5.0), (14.0, 5.0), (5.0, 14.0), (14.0, 14.0)];
        let tolerance = 3.0;

        for &(ex, ey) in &expected_corners {
            let has_nearby = kps.keypoints.iter().any(|kp| {
                let dx = kp.x - ex;
                let dy = kp.y - ey;
                (dx * dx + dy * dy).sqrt() <= tolerance
            });
            assert!(
                has_nearby,
                "Expected a Harris keypoint within {}px of ({}, {}), but none found. \
                 Detected keypoints: {:?}",
                tolerance,
                ex,
                ey,
                kps.keypoints
                    .iter()
                    .map(|kp| (kp.x, kp.y))
                    .collect::<Vec<_>>()
            );
        }
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

    #[test]
    fn test_harris_nms_reduces_detections() {
        // With NMS enabled we should get fewer (sparser) detections
        // than the total number of above-threshold pixels
        let img = create_test_image_with_corners();
        let kps = harris_detect(&img, 3, 3, 0.04, 100.0);
        // The NMS should significantly thin out dense clusters
        // Just verify we still get corners and that the count is reasonable
        assert!(!kps.keypoints.is_empty());
        // On a 20x20 image the NMS corners should be relatively few
        assert!(kps.keypoints.len() < 20 * 20);
    }

    #[test]
    fn test_harris_block_size_effect() {
        let img = create_test_image_with_corners();
        // Larger block_size uses a wider Gaussian window, should still detect corners
        let kps3 = harris_detect(&img, 3, 3, 0.04, 1000.0);
        let kps5 = harris_detect(&img, 5, 3, 0.04, 1000.0);
        assert!(!kps3.keypoints.is_empty());
        assert!(!kps5.keypoints.is_empty());
    }
}
