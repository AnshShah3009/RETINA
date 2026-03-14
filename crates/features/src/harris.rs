use crate::KeyPoints;
use cv_core::KeyPoint;
use image::GrayImage;
use rayon::prelude::*;

/// Return 1-D Sobel kernel pair (derivative, smoothing) for the given aperture size.
fn sobel_kernels_1d(ksize: i32) -> (Vec<f64>, Vec<f64>) {
    match ksize {
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        5 => (
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        ),
        7 => (
            vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0],
            vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        ),
        _ => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
    }
}

/// Build a normalised 1-D Gaussian kernel of the given (odd) size and sigma.
fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64> {
    let center = (size / 2) as isize;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = (i as isize - center) as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        sum += v;
    }
    if sum != 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    kernel
}

/// Clamp-border pixel fetch helper.
fn pixel_at(image: &GrayImage, x: i32, y: i32) -> f64 {
    let cx = x.clamp(0, image.width() as i32 - 1) as u32;
    let cy = y.clamp(0, image.height() as i32 - 1) as u32;
    image.get_pixel(cx, cy)[0] as f64
}

/// Compute Sobel gradients Ix, Iy for every pixel using separable 1-D kernels.
///
/// Returns two `Vec<f64>` of length width*height, stored in row-major order.
fn compute_sobel_gradients(
    image: &GrayImage,
    ksize: i32,
) -> (Vec<f64>, Vec<f64>) {
    let w = image.width() as i32;
    let h = image.height() as i32;
    let n = (w * h) as usize;

    let (deriv, smooth) = sobel_kernels_1d(ksize);
    let half = (deriv.len() / 2) as i32;

    // Ix: derivative in x, smoothing in y  => horiz pass uses deriv, vert pass uses smooth
    // Iy: smoothing in x, derivative in y  => horiz pass uses smooth, vert pass uses deriv

    // Horizontal pass for Ix (deriv in x)
    let mut tmp_ix = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                sum += pixel_at(image, x + k, y) * deriv[(k + half) as usize];
            }
            tmp_ix[(y * w + x) as usize] = sum;
        }
    }

    // Vertical pass for Ix (smooth in y)
    let mut ix = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                let sy = (y + k).clamp(0, h - 1);
                sum += tmp_ix[(sy * w + x) as usize] * smooth[(k + half) as usize];
            }
            ix[(y * w + x) as usize] = sum;
        }
    }

    // Horizontal pass for Iy (smooth in x)
    let mut tmp_iy = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                sum += pixel_at(image, x + k, y) * smooth[(k + half) as usize];
            }
            tmp_iy[(y * w + x) as usize] = sum;
        }
    }

    // Vertical pass for Iy (deriv in y)
    let mut iy = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in -(half)..=(half) {
                let sy = (y + k).clamp(0, h - 1);
                sum += tmp_iy[(sy * w + x) as usize] * deriv[(k + half) as usize];
            }
            iy[(y * w + x) as usize] = sum;
        }
    }

    (ix, iy)
}

/// Apply separable 1-D Gaussian blur to an f64 buffer (row-major, width x height).
fn gaussian_blur_f64(
    buf: &[f64],
    width: usize,
    height: usize,
    kernel: &[f64],
) -> Vec<f64> {
    let half = (kernel.len() / 2) as isize;
    let n = width * height;

    // Horizontal pass
    let mut tmp = vec![0.0f64; n];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for k in -half..=half {
                let sx = (x as isize + k).clamp(0, width as isize - 1) as usize;
                sum += buf[y * width + sx] * kernel[(k + half) as usize];
            }
            tmp[y * width + x] = sum;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f64; n];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for k in -half..=half {
                let sy = (y as isize + k).clamp(0, height as isize - 1) as usize;
                sum += tmp[sy * width + x] * kernel[(k + half) as usize];
            }
            out[y * width + x] = sum;
        }
    }

    out
}

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
