use cv_imgproc::color::{adjust_brightness_in_pool, convert_rgb_to_gray_in_pool};
use image::{Rgb, RgbImage};
use rayon::ThreadPoolBuilder;

#[test]
fn test_custom_pool_execution() {
    // Create a strict pool with 1 thread to verify it works even under constraints
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    let mut rgb = RgbImage::new(100, 100);
    for p in rgb.pixels_mut() {
        *p = Rgb([100, 150, 200]);
    }

    // Run conversion in custom pool
    let gray = convert_rgb_to_gray_in_pool(&rgb, Some(&pool));

    assert_eq!(gray.width(), 100);
    // 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 = 140.75 -> 140
    assert_eq!(gray.get_pixel(0, 0)[0], 140);

    // Run brightness in custom pool
    let bright = adjust_brightness_in_pool(&gray, 1.2, Some(&pool));
    // 140 * 1.2 = 168
    assert_eq!(bright.get_pixel(0, 0)[0], 168);
}
