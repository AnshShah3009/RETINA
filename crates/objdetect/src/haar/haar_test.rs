#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use image::Luma;

#[test]
fn test_haar_integral_image() {
    let mut img = GrayImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, Luma([1]));
        }
    }

    let integral = compute_integral_image(&img).expect("Failed to compute integral image");
    // 4x4 image -> 5x5 integral image
    assert_eq!(integral.shape.width, 5);
    assert_eq!(integral.shape.height, 5);

    // Sum of 2x2 area at (1, 1)
    let sum = get_rect_sum(&integral, 1, 1, 2, 2).expect("Failed to get rect sum");
    assert_eq!(sum, 4);
}

#[test]
fn test_haar_detection_simple() {
    // Create a 100x100 black image with a 20x20 white square at (40, 40)
    let mut img = GrayImage::new(100, 100);
    for y in 40..60 {
        for x in 40..60 {
            img.put_pixel(x, y, Luma([255]));
        }
    }

    // Create a simple cascade that detects a 20x20 white square
    // A single stage with one feature: a 20x20 rectangle with weight 1.0
    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0, // Should be much higher for a white square
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");
    println!("Detections: {:?}", detections);

    assert!(!detections.is_empty());
    // Check if at least one detection is near (40, 40)
    let found = detections
        .iter()
        .any(|r| (r.x - 40.0).abs() < 5.0 && (r.y - 40.0).abs() < 5.0);
    assert!(found);
}

#[test]
fn test_haar_empty_image() {
    let mut img = GrayImage::new(50, 50);
    // Black (empty) image
    for pixel in img.iter_mut() {
        *pixel = 0;
    }

    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0,
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");
    // Empty image should have no detections
    assert!(detections.is_empty());
}

#[test]
fn test_haar_small_image() {
    // Image smaller than cascade size
    let img = GrayImage::new(10, 10);

    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0,
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");
    // Image is smaller than cascade window
    assert!(detections.is_empty());
}

#[test]
fn test_haar_multiple_scales() {
    let mut img = GrayImage::new(200, 200);

    // Create white squares at different scales
    // Small square at (50, 50)
    for y in 50..70 {
        for x in 50..70 {
            img.put_pixel(x, y, Luma([255]));
        }
    }
    // Larger square at (120, 120)
    for y in 120..160 {
        for x in 120..160 {
            img.put_pixel(x, y, Luma([255]));
        }
    }

    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0,
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");

    // Should find detections at multiple scales
    assert!(!detections.is_empty());
}

#[test]
fn test_haar_scale_factor_impact() {
    let mut img = GrayImage::new(100, 100);
    // Create white region
    for y in 30..70 {
        for x in 30..70 {
            img.put_pixel(x, y, Luma([255]));
        }
    }

    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 100.0,
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    // Test with different scale factors
    let detections_fast = cascade.detect(&img, 1.5, 0).expect("Detection failed"); // Larger scale factor
    let detections_slow = cascade.detect(&img, 1.05, 0).expect("Detection failed"); // Smaller scale factor

    // Smaller scale factor should check more scales, potentially finding more
    assert!(!detections_fast.is_empty());
    assert!(!detections_slow.is_empty());
}

#[test]
fn test_haar_integral_image_bounds() {
    // Test integral image computation doesn't overflow for large images
    let mut img = GrayImage::new(256, 256);

    // Create a high-value image
    for pixel in img.iter_mut() {
        *pixel = 200;
    }

    let integral = compute_integral_image(&img).expect("Failed to compute integral image");

    // Integral image should be 257x257 (width + 1, height + 1)
    assert_eq!(integral.shape.width, 257);
    assert_eq!(integral.shape.height, 257);

    // Final value should be sum of all pixels
    let data = integral.as_slice().expect("Failed to get integral slice");
    let total = data[data.len() - 1];
    assert_eq!(total, 256 * 256 * 200);
}

#[test]
fn test_haar_rect_sum_correctness() {
    let mut img = GrayImage::new(8, 8);

    // Create a pattern: half white, half black
    for y in 0..8 {
        for x in 0..8 {
            let val = if x < 4 { 100u8 } else { 200u8 };
            img.put_pixel(x, y, Luma([val]));
        }
    }

    let integral = compute_integral_image(&img).expect("Failed to compute integral image");

    // Sum of left half (4x8, value 100) should be 4 * 8 * 100 = 3200
    let left_sum = get_rect_sum(&integral, 0, 0, 4, 8).expect("Failed to get rect sum");
    assert_eq!(left_sum, 3200);

    // Sum of right half (4x8, value 200) should be 4 * 8 * 200 = 6400
    let right_sum = get_rect_sum(&integral, 4, 0, 4, 8).expect("Failed to get rect sum");
    assert_eq!(right_sum, 6400);
}

#[test]
fn test_haar_detection_no_false_positives() {
    // Create image with no features
    let img = GrayImage::new(100, 100);

    // Create a cascade that should never trigger
    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 20.0, 20.0), 1.0)],
        threshold: 5000.0, // Very high threshold
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 1000.0, // Very high stage threshold
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (20, 20),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");
    // With high thresholds on empty image, should have no detections
    assert!(detections.is_empty());
}

#[test]
fn test_haar_cascade_size_validation() {
    let mut img = GrayImage::new(100, 100);
    // Create white region matching cascade size
    for y in 0..40 {
        for x in 0..40 {
            img.put_pixel(x, y, Luma([255]));
        }
    }

    let feature = HaarFeature {
        rects: vec![(Rect::new(0.0, 0.0, 40.0, 40.0), 1.0)],
        threshold: 100.0,
        left_val: -1.0,
        right_val: 1.0,
    };

    let stage = CascadeStage {
        threshold: 0.5,
        features: vec![feature],
    };

    let cascade = HaarCascade {
        stages: vec![stage],
        size: (40, 40),
    };

    let detections = cascade.detect(&img, 1.1, 0).expect("Detection failed");
    // Should detect the matching region
    assert!(!detections.is_empty());
}
