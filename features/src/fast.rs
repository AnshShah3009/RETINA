use cv_core::{KeyPoint, KeyPoints};
use image::GrayImage;

/// FAST-9 corner detector
/// Uses a 16-pixel Bresenham circle of radius 3
/// A corner is detected if 9 contiguous pixels are all brighter or all darker
pub fn fast_detect(image: &GrayImage, threshold: u8, max_keypoints: usize) -> KeyPoints {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let mut keypoints = Vec::new();

    // Bresenham circle of radius 3 - 16 points
    // These are the (x, y) offsets from the center pixel
    #[rustfmt::skip]
    let circle_offsets: [(i32, i32); 16] = [
        (0, -3),  (1, -3),  (2, -2),  (3, -1),
        (3, 0),   (3, 1),   (2, 2),   (1, 3),
        (0, 3),   (-1, 3),  (-2, 2),  (-3, 1),
        (-3, 0),  (-3, -1), (-2, -2), (-1, -3),
    ];

    for y in 3..height - 3 {
        for x in 3..width - 3 {
            let p = image.get_pixel(x as u32, y as u32)[0];

            let high_threshold = p.saturating_add(threshold);
            let low_threshold = p.saturating_sub(threshold);

            // Full test - check all 16 pixels directly
            let mut pixel_values = [0u8; 16];
            let mut brighter_count = 0u32;
            let mut darker_count = 0u32;

            for (i, (dx, dy)) in circle_offsets.iter().enumerate() {
                let px = (x + dx) as u32;
                let py = (y + dy) as u32;
                let val = image.get_pixel(px, py)[0];
                pixel_values[i] = val;

                if val > high_threshold {
                    brighter_count += 1;
                } else if val < low_threshold {
                    darker_count += 1;
                }
            }

            // Quick rejection: need at least 9 bright or 9 dark to have a chance
            if brighter_count < 9 && darker_count < 9 {
                continue;
            }

            // Check for 9 contiguous brighter or darker pixels
            if has_n_contiguous(&pixel_values, p, threshold, 9, true)
                || has_n_contiguous(&pixel_values, p, threshold, 9, false)
            {
                let kp = KeyPoint::new(x as f64, y as f64);
                keypoints.push(kp);
            }
        }
    }

    if keypoints.len() > max_keypoints {
        keypoints.truncate(max_keypoints);
    }

    KeyPoints { keypoints }
}

/// Check if there are n contiguous pixels that are all brighter or all darker
fn has_n_contiguous(
    pixels: &[u8; 16],
    center: u8,
    threshold: u8,
    n: usize,
    check_brighter: bool,
) -> bool {
    let high_threshold = center.saturating_add(threshold);
    let low_threshold = center.saturating_sub(threshold);

    // Create a binary array: 1 if pixel meets condition, 0 otherwise
    let mut binary = [0u8; 16];
    for (i, &p) in pixels.iter().enumerate() {
        if check_brighter {
            if p > high_threshold {
                binary[i] = 1;
            }
        } else if p < low_threshold {
            binary[i] = 1;
        }
    }

    // Check for n contiguous 1s in the circular array
    // We need to check the array twice to handle wrap-around
    let mut max_consecutive = 0;
    let mut current_consecutive = 0;

    for i in 0..(16 + n) {
        let idx = i % 16;
        if binary[idx] == 1 {
            current_consecutive += 1;
            max_consecutive = max_consecutive.max(current_consecutive);
            if max_consecutive >= n {
                return true;
            }
        } else {
            current_consecutive = 0;
        }
    }

    false
}

/// Compute corner score using the sum of absolute differences
pub fn corner_score(image: &GrayImage, x: i32, y: i32, _threshold: u8) -> u8 {
    let circle_offsets: [(i32, i32); 16] = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ];

    let p = image.get_pixel(x as u32, y as u32)[0];
    let mut min_diff = 255u8;

    for (dx, dy) in &circle_offsets {
        let px = (x + dx) as u32;
        let py = (y + dy) as u32;
        let val = image.get_pixel(px, py)[0];
        let diff = val.abs_diff(p);
        min_diff = min_diff.min(diff);
    }

    min_diff
}

/// Non-maximum suppression for FAST keypoints
pub fn non_max_suppression(keypoints: KeyPoints, image: &GrayImage, threshold: u8) -> KeyPoints {
    let mut scored_kps: Vec<(KeyPoint, u8)> = keypoints
        .keypoints
        .into_iter()
        .map(|kp| {
            let score = corner_score(image, kp.x as i32, kp.y as i32, threshold);
            (kp, score)
        })
        .collect();

    // Sort by score descending
    scored_kps.sort_by(|a, b| b.1.cmp(&a.1));

    let mut suppressed: Vec<KeyPoint> = Vec::new();
    let min_distance = 5.0; // Minimum distance between keypoints

    for (kp, _) in scored_kps {
        // Check if this keypoint is far enough from all kept keypoints
        let mut keep = true;
        for kept_kp in &suppressed {
            let dx = kp.x - kept_kp.x;
            let dy = kp.y - kept_kp.y;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < min_distance * min_distance {
                keep = false;
                break;
            }
        }
        if keep {
            suppressed.push(kp);
        }
    }

    KeyPoints {
        keypoints: suppressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_fast_detector_simple() {
        // Create a simple image with a clear corner
        // Pattern: black background with white square in center
        let size = 50u32;
        let mut img = GrayImage::new(size, size);

        // Fill with black
        for y in 0..size {
            for x in 0..size {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        // Draw white square from (15, 15) to (35, 35)
        for y in 15..35 {
            for x in 15..35 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        // Debug: Check pixel values at a corner (15, 15)
        println!("Pixel at (15, 15): {}", img.get_pixel(15, 15)[0]);
        println!("Pixel at (14, 14): {}", img.get_pixel(14, 14)[0]);
        println!("Pixel at (16, 16): {}", img.get_pixel(16, 16)[0]);

        // Check the circle pixels around (15, 15)
        let circle_offsets: [(i32, i32); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        println!("\nCircle pixels around (15, 15):");
        for (i, (dx, dy)) in circle_offsets.iter().enumerate() {
            let x = (15i32 + dx) as u32;
            let y = (15i32 + dy) as u32;
            if x < size && y < size {
                println!("  [{}] ({}, {}): {}", i, x, y, img.get_pixel(x, y)[0]);
            }
        }

        // Detect with threshold 20
        let kps = fast_detect(&img, 20, 100);

        println!("\nDetected {} keypoints", kps.len());
        for kp in &kps.keypoints {
            println!("  Keypoint at ({}, {})", kp.x, kp.y);
        }

        // We expect at least 4 corners of the white square
        assert!(
            kps.len() >= 4,
            "Expected at least 4 corners, found {}",
            kps.len()
        );
    }

    #[test]
    fn test_fast_detector_circle_pattern() {
        // Create a pattern with a bright spot - should detect corners around it
        let size = 64u32;
        let mut img = GrayImage::new(size, size);

        // Fill with black
        for y in 0..size {
            for x in 0..size {
                img.put_pixel(x, y, Luma([0]));
            }
        }

        // Draw a white circle in the center - corners should be at the circle's edge
        let center_x = size as f32 / 2.0;
        let center_y = size as f32 / 2.0;
        let radius = 15.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Create a sharp edge at the circle boundary
                if dist <= radius {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }

        // Detect with threshold
        let kps = fast_detect(&img, 50, 500);

        println!("\nDetected {} keypoints in circle pattern", kps.len());
        for kp in &kps.keypoints {
            println!("  Keypoint at ({:.1}, {:.1})", kp.x, kp.y);
        }

        // Should detect corners around the circle
        assert!(
            kps.len() >= 4,
            "Expected at least 4 corners in circle pattern, found {}",
            kps.len()
        );
    }

    #[test]
    fn test_has_n_contiguous() {
        // Test the contiguous detection function directly
        let pixels_all_dark = [0u8; 16];
        let center = 255u8;
        let threshold = 50u8;

        // All 16 pixels are 0, center is 255, threshold 50
        // Dark condition: pixel < 255 - 50 = 205
        // All pixels are 0 < 205, so all 16 should be "dark"
        let result = has_n_contiguous(&pixels_all_dark, center, threshold, 9, false);
        println!("All dark pixels (0), center 255, threshold 50:");
        println!("  Has 9 contiguous dark: {}", result);
        assert!(result, "Should detect 9 contiguous dark pixels");

        // Test alternating pattern - 8 bright, 8 dark
        let mut pixels_alt = [0u8; 16];
        for i in 0..16 {
            pixels_alt[i] = if i % 2 == 0 { 255 } else { 0 };
        }
        let result_alt = has_n_contiguous(&pixels_alt, 128, 50, 9, false);
        println!("\nAlternating pattern with center 128:");
        println!("  Has 9 contiguous dark: {}", result_alt);

        // Test with 9 consecutive darks
        let mut pixels_mixed = [255u8; 16];
        for i in 0..9 {
            pixels_mixed[i] = 0;
        }
        let result_mixed = has_n_contiguous(&pixels_mixed, 255, 50, 9, false);
        println!("\n9 consecutive darks, 7 brights, center 255:");
        println!("  Has 9 contiguous dark: {}", result_mixed);
        assert!(result_mixed, "Should detect 9 contiguous dark pixels");
    }
}
