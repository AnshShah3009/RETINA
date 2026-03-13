use cv_stereo::*;
use image::{GrayImage, Luma};

#[test]
fn test_block_matching_sad_simd() {
    let width = 64;
    let height = 64;
    let mut left = GrayImage::new(width, height);
    let mut right = GrayImage::new(width, height);

    // Create a pattern
    for y in 0..height {
        for x in 0..width {
            // Shifted left image
            let lx = if x >= 4 { x - 4 } else { 0 };
            let lval = if (lx / 8 + y / 8) % 2 == 0 { 200 } else { 50 };
            left.put_pixel(x, y, Luma([lval]));

            let rval = if (x / 8 + y / 8) % 2 == 0 { 200 } else { 50 };
            right.put_pixel(x, y, Luma([rval]));
        }
    }

    let matcher = BlockMatcher::new()
        .with_block_size(7)
        .with_disparity_range(0, 10)
        .with_metric(MatchingMetric::SAD);

    let disparity = matcher.compute(&left, &right).unwrap();

    // Check center pixel (should have disparity approx 4)
    let d = disparity.get(32, 32);
    assert!((d - 4.0).abs() < 1.0);
}

#[test]
fn test_block_matching_ssd_simd() {
    let width = 64;
    let height = 64;
    let mut left = GrayImage::new(width, height);
    let mut right = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let lx = if x >= 2 { x - 2 } else { 0 };
            let lval = (lx % 255) as u8;
            left.put_pixel(x, y, Luma([lval]));

            let rval = (x % 255) as u8;
            right.put_pixel(x, y, Luma([rval]));
        }
    }

    let matcher = BlockMatcher::new()
        .with_block_size(5)
        .with_disparity_range(0, 5)
        .with_metric(MatchingMetric::SSD);

    let disparity = matcher.compute(&left, &right).unwrap();

    let d = disparity.get(32, 32);
    assert!((d - 2.0).abs() < 1.0);
}
