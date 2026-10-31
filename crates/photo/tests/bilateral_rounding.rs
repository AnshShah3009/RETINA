//! The bilateral filter rounds its output like OpenCV's saturate_cast; it
//! used to truncate, biasing results ~0.5 grey level dark.

use cv_photo::bilateral_filter;
use image::{GrayImage, Luma};

#[test]
fn uniform_images_are_fixed_points() {
    for v in [1u8, 37, 128, 200, 254, 255] {
        let img = GrayImage::from_pixel(16, 16, Luma([v]));
        let out = bilateral_filter(&img, 5, 30.0, 3.0);
        assert!(
            out.pixels().all(|p| p[0] == v),
            "value {v} drifted to {:?}",
            out.get_pixel(8, 8)
        );
    }
}

#[test]
fn output_is_rounded_not_truncated() {
    // Two-level step: near the edge the weighted mean is fractional; over
    // the image the mean output must match the rounded float filter within
    // a tenth of a level, where truncation is ~0.5 low.
    let img = GrayImage::from_fn(32, 8, |x, _| Luma([if x < 16 { 100 } else { 101 }]));
    let out = bilateral_filter(&img, 5, 50.0, 2.0);
    let mean_in: f64 = img.pixels().map(|p| p[0] as f64).sum::<f64>() / 256.0;
    let mean_out: f64 = out.pixels().map(|p| p[0] as f64).sum::<f64>() / 256.0;
    assert!(
        (mean_out - mean_in).abs() < 0.1,
        "mean {mean_in} -> {mean_out}"
    );
}
