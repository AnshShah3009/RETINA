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
    // 0.299*100 + 0.587*150 + 0.114*200 = 140.75, rounded to 141 as
    // cv2.cvtColor(..., COLOR_RGB2GRAY) gives. (This expected the truncated
    // 140 until a7c10ff made the CPU path round like OpenCV.)
    assert_eq!(gray.get_pixel(0, 0)[0], 141);

    // Run brightness in custom pool
    let bright = adjust_brightness_in_pool(&gray, 1.2, Some(&pool));
    // 141 * 1.2 = 169.2 -> 169 (cv2.convertScaleAbs gives 169).
    assert_eq!(bright.get_pixel(0, 0)[0], 169);
}
