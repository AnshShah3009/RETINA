use cv_hal::context::TemplateMatchMethod;
use cv_imgproc::*;
use image::{GrayImage, Luma, Rgb, RgbImage};

#[test]
fn test_resize_functional() {
    let mut img = GrayImage::new(100, 100);
    img.put_pixel(50, 50, Luma([255]));

    // Scale up
    let up = resize(&img, 200, 200, Interpolation::Linear);
    assert_eq!(up.width(), 200);
    assert_eq!(up.height(), 200);

    // Scale down
    let down = resize(&img, 50, 50, Interpolation::Linear);
    assert_eq!(down.width(), 50);
    assert_eq!(down.height(), 50);
}

#[test]
fn test_resize_rgb() {
    let mut img = RgbImage::new(10, 10);
    img.put_pixel(5, 5, Rgb([255, 128, 64]));
    let resized = resize_rgb(&img, 20, 20, Interpolation::Linear);
    assert_eq!(resized.width(), 20);
    assert_eq!(resized.height(), 20);
    // Rough check that color is preserved somewhere
    let has_color = resized.pixels().any(|p| p[0] > 0);
    assert!(has_color);
}

#[test]
fn test_morphology_consistency() {
    let mut img = GrayImage::new(10, 10);
    img.put_pixel(5, 5, Luma([255]));

    let kernel = create_morph_kernel(MorphShape::Rectangle, 3, 3);

    let dilated = dilate(&img, &kernel, 1);
    // Center 3x3 should be white
    assert_eq!(dilated.get_pixel(4, 4)[0], 255);
    assert_eq!(dilated.get_pixel(6, 6)[0], 255);

    let eroded = erode(&dilated, &kernel, 1);
    // Should be back to mostly original (single pixel)
    assert_eq!(eroded.get_pixel(5, 5)[0], 255);
    assert_eq!(eroded.get_pixel(4, 4)[0], 0);
}

#[test]
fn test_geometry_warp_identity() {
    let mut img = GrayImage::new(10, 10);
    img.put_pixel(5, 5, Luma([128]));

    let identity = nalgebra::Matrix3::identity();
    let warped = warp_perspective(&img, &identity, 10, 10);

    assert_eq!(warped.get_pixel(5, 5)[0], 128);
}

#[test]
fn test_local_threshold_niblack() {
    let mut img = GrayImage::new(20, 20);
    for y in 0..20 {
        for x in 0..20 {
            let dx = x as i32 - 10;
            let dy = y as i32 - 10;
            let val = if dx * dx + dy * dy < 25 { 200 } else { 50 };
            img.put_pixel(x, y, Luma([val]));
        }
    }

    let thresholded = local_threshold(
        &img,
        255,
        LocalThresholdMethod::Niblack,
        ThresholdType::Binary,
        11,
        -0.2,
        0.0,
    );

    assert_eq!(thresholded.width(), 20);
    assert_eq!(thresholded.height(), 20);
    // Center should be white, edges black
    assert_eq!(thresholded.get_pixel(10, 10)[0], 255);
    assert_eq!(thresholded.get_pixel(0, 0)[0], 0);
}

#[test]
fn test_histogram_equalization_improves_contrast() {
    let mut img = GrayImage::new(10, 10);
    // Very low contrast image
    for p in img.pixels_mut() {
        *p = Luma([100]);
    }
    img.put_pixel(0, 0, Luma([101]));

    let equalized = histogram_equalization(&img);
    let max_val = equalized.pixels().map(|p| p[0]).max().unwrap();
    let min_val = equalized.pixels().map(|p| p[0]).min().unwrap();

    // Contrast should be expanded
    assert!(max_val > 101);
    assert!(min_val < 100);
}

#[test]
fn test_template_matching() {
    let mut img = GrayImage::new(20, 20);
    let mut templ = GrayImage::new(5, 5);

    for y in 0..5 {
        for x in 0..5 {
            img.put_pixel(x + 7, y + 3, Luma([255]));
            templ.put_pixel(x, y, Luma([255]));
        }
    }

    let res = match_template(&img, &templ, TemplateMatchMethod::CcorrNormed);
    let ((_min_x, _min_y, _min_v), (max_x, max_y, max_v)) = min_max_loc(&res);

    assert_eq!(max_x, 7);
    assert_eq!(max_y, 3);
    assert!((max_v - 1.0).abs() < 1e-5);
}

#[test]
fn test_geometry_remap() {
    let mut img = GrayImage::new(10, 10);
    img.put_pixel(2, 3, Luma([200]));

    // Create maps for 90 degree rotation
    let mut map_x = vec![0.0f32; 100];
    let mut map_y = vec![0.0f32; 100];

    for y in 0..10 {
        for x in 0..10 {
            // dst(x, y) = src(y, 9-x)
            map_x[y * 10 + x] = y as f32;
            map_y[y * 10 + x] = (9 - x) as f32;
        }
    }

    let remapped = remap(
        &img,
        &map_x,
        &map_y,
        10,
        10,
        Interpolation::Nearest,
        BorderMode::Constant(0),
    );

    // Original (2, 3) should move to (9-3, 2) = (6, 2)
    // Wait, map is dst -> src.
    // So src(map_x(x,y), map_y(x,y))
    // We want dst(6, 2) to sample from src(2, 3).
    // x=6, y=2 -> map_x(6,2) = 2, map_y(6,2) = 9-6 = 3. Yes.
    assert_eq!(remapped.get_pixel(6, 2)[0], 200);
}
