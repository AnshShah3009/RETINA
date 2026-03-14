use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;

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
fn compute_sobel_gradients(
    image: &GrayImage,
    ksize: i32,
) -> (Vec<f64>, Vec<f64>) {
    let w = image.width() as i32;
    let h = image.height() as i32;
    let n = (w * h) as usize;

    let (deriv, smooth) = sobel_kernels_1d(ksize);
    let half = (deriv.len() / 2) as i32;

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

/// Detect corners using the Good Features to Track (Shi-Tomasi) criterion.
///
/// Computes the minimum eigenvalue of the Gaussian-weighted structure tensor
/// at each pixel (`R = min(lambda_1, lambda_2)`). Sobel gradients use aperture
/// size 3. The structure tensor window is `block_size` with Gaussian weighting.
///
/// Non-maximum suppression (3x3) is applied before quality and distance filtering.
///
/// Returns at most `max_corners` keypoints with a minimum quality of
/// `quality_level` (fraction of the strongest response) and spaced at least
/// `min_distance` pixels apart.
pub fn gftt_detect(
    image: &GrayImage,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
) -> KeyPoints {
    gftt_detect_with_params(image, max_corners, quality_level, min_distance, 3, 3)
}

/// Full-parameter variant of GFTT detection.
pub fn gftt_detect_with_params(
    image: &GrayImage,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    ksize: i32,
) -> KeyPoints {
    let width = image.width() as usize;
    let height = image.height() as usize;

    // Step 1: Compute Sobel gradients
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

    // Step 3: Gaussian blur each product image
    let bs = (block_size.max(3) | 1) as usize;
    let sigma = bs as f64 * 0.5;
    let gauss = gaussian_kernel_1d(bs, sigma);

    let sxx = gaussian_blur_f64(&ixx, width, height, &gauss);
    let syy = gaussian_blur_f64(&iyy, width, height, &gauss);
    let sxy = gaussian_blur_f64(&ixy, width, height, &gauss);

    // Step 4: Compute minimum eigenvalue at each pixel
    let mut response = vec![0.0f64; n];
    let mut max_score = 0.0f64;

    for i in 0..n {
        let trace = sxx[i] + syy[i];
        let term = ((sxx[i] - syy[i]).powi(2) + 4.0 * sxy[i] * sxy[i]).sqrt();
        let lambda_min = (trace - term) * 0.5;

        if lambda_min > 0.0 {
            response[i] = lambda_min;
            if lambda_min > max_score {
                max_score = lambda_min;
            }
        }
    }

    // Step 5: Non-maximum suppression (3x3) + quality threshold
    let threshold = max_score * quality_level;
    let mut candidates = Vec::new();

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let r = response[y * width + x];
            if r < threshold {
                continue;
            }
            // 3x3 NMS: must be strictly greater than all neighbours
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
                candidates.push((x, y, r));
            }
        }
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Step 6: Min-distance filtering and max-corners limit
    let mut corners = KeyPoints::new();
    let min_dist_sq = min_distance * min_distance;

    for (x, y, score) in candidates {
        if corners.len() >= max_corners {
            break;
        }

        let mut too_close = false;
        for kp in corners.keypoints.iter() {
            let dx = (x as f64) - kp.x;
            let dy = (y as f64) - kp.y;
            if dx * dx + dy * dy < min_dist_sq {
                too_close = true;
                break;
            }
        }

        if !too_close {
            corners.push(KeyPoint::new(x as f64, y as f64).with_response(score));
        }
    }

    corners
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_image_with_corners() -> GrayImage {
        let mut img = GrayImage::new(30, 30);
        for y in 0..30 {
            for x in 0..30 {
                let val = if (x < 10 && y < 10)
                    || (x > 19 && y < 10)
                    || (x < 10 && y > 19)
                    || (x > 19 && y > 19)
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
        GrayImage::from_pixel(30, 30, Luma([128]))
    }

    #[test]
    fn test_gftt_detect_finds_corners() {
        let img = create_test_image_with_corners();
        let kps = gftt_detect(&img, 100, 0.01, 5.0);
        assert!(!kps.keypoints.is_empty(), "Should detect corners");
    }

    #[test]
    fn test_gftt_detect_uniform_image() {
        let img = create_uniform_image();
        let kps = gftt_detect(&img, 100, 0.01, 5.0);
        assert!(
            kps.keypoints.is_empty(),
            "Uniform image should have no corners"
        );
    }

    #[test]
    fn test_gftt_detect_max_corners_limit() {
        let img = create_test_image_with_corners();
        let kps = gftt_detect(&img, 2, 0.01, 5.0);
        assert!(kps.keypoints.len() <= 2);
    }

    #[test]
    fn test_gftt_detect_min_distance() {
        let img = create_test_image_with_corners();
        let kps = gftt_detect(&img, 100, 0.01, 10.0);

        for i in 0..kps.keypoints.len() {
            for j in (i + 1)..kps.keypoints.len() {
                let dx = kps.keypoints[i].x - kps.keypoints[j].x;
                let dy = kps.keypoints[i].y - kps.keypoints[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
                assert!(dist >= 10.0 || dist < 0.1);
            }
        }
    }

    #[test]
    fn test_gftt_detect_quality_level() {
        let img = create_test_image_with_corners();
        let kps_low = gftt_detect(&img, 100, 0.001, 5.0);
        let kps_high = gftt_detect(&img, 100, 0.5, 5.0);
        assert!(kps_low.keypoints.len() >= kps_high.keypoints.len());
    }

    #[test]
    fn test_gftt_keypoint_response() {
        let img = create_test_image_with_corners();
        let kps = gftt_detect(&img, 100, 0.01, 5.0);

        for kp in &kps.keypoints {
            assert!(kp.response > 0.0);
        }
    }
}
