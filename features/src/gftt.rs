use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;

/// Detect corners using the Good Features to Track (Shi-Tomasi) criterion.
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
    let width = image.width() as i32;
    let height = image.height() as i32;
    let half_window = 1i32;

    let mut scores = Vec::new();

    let mut max_score = 0.0f64;

    for y in (half_window + 1)..(height - half_window - 1) {
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

            // Min eigenvalue calculation
            let trace = i_xx + i_yy;
            let _det = i_xx * i_yy - i_xy * i_xy;
            let term = ((i_xx - i_yy).powi(2) + 4.0 * i_xy * i_xy).sqrt();
            let lambda_min = (trace - term) * 0.5;

            if lambda_min > 0.0 {
                if lambda_min > max_score {
                    max_score = lambda_min;
                }
                scores.push((x, y, lambda_min));
            }
        }
    }

    // Filter by quality level
    let threshold = max_score * quality_level;
    let mut candidates: Vec<_> = scores
        .into_iter()
        .filter(|&(_, _, s)| s >= threshold)
        .collect();

    // Sort by score descending
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

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
