//! `threshold_otsu` against `cv2.threshold(img, 0, max, type | THRESH_OTSU)`
//! (OpenCV 4.13).
//!
//! OpenCV's `getThreshVal_Otsu_8u` starts at `max_val = 0` and only moves it
//! when a split has strictly positive between-class variance. A single-valued
//! image has no split, so cv2 returns threshold 0 - and THRESH_BINARY maps a
//! constant 100 image to `max`, not 0. The Rust port fell back to the
//! histogram mode instead (100), which inverted every binary output on flat
//! input. Values below are cv2's, measured:
//!
//! ```text
//! full((7,5), v), max 200     thresh  BINARY  BINARY_INV  TOZERO  TRUNC
//! v = 100                     0       200     0           100     0
//! v = 0                       0       0       200         0       0
//! v = 255                     0       200     0           255     0
//! v = 1                       0       200     0           1       0
//! ```

use cv_imgproc::{ThresholdType, threshold_otsu};
use image::GrayImage;

fn rand_img(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut s = seed;
    GrayImage::from_fn(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        image::Luma([(s >> 24) as u8])
    })
}

#[test]
fn uniform_image_thresholds_at_zero_like_opencv() {
    use ThresholdType::*;
    // (value, [BINARY, BINARY_INV, TOZERO, TRUNC]) from cv2.
    let cases: [(u8, [u8; 4]); 4] = [
        (100, [200, 0, 100, 0]),
        (0, [0, 200, 0, 0]),
        (255, [200, 0, 255, 0]),
        (1, [200, 0, 1, 0]),
    ];
    for (v, want) in cases {
        let img = GrayImage::from_pixel(5, 7, image::Luma([v]));
        for (typ, w) in [Binary, BinaryInv, ToZero, Trunc].into_iter().zip(want) {
            let (t, out) = threshold_otsu(&img, 200, typ);
            assert_eq!(t, 0, "constant {v}: cv2's Otsu threshold is 0");
            assert!(
                out.as_raw().iter().all(|&p| p == w),
                "constant {v}, {typ:?}: expected {w} everywhere, got {}",
                out.as_raw()[0]
            );
        }
    }
}

#[test]
fn thresholds_match_opencv_on_textured_images() {
    // cv2.threshold(img, 0, 255, THRESH_BINARY | THRESH_OTSU)[0] for the LCG
    // images below, and a two-level image whose between-class variance is
    // flat over [50, 199] (cv2 keeps the first maximum, 50).
    assert_eq!(
        threshold_otsu(&rand_img(37, 29, 7), 255, ThresholdType::Binary).0,
        127
    );
    assert_eq!(
        threshold_otsu(&rand_img(64, 48, 3), 255, ThresholdType::Binary).0,
        127
    );
    let two = GrayImage::from_fn(10, 10, |_, y| image::Luma([if y < 5 { 50 } else { 200 }]));
    assert_eq!(threshold_otsu(&two, 255, ThresholdType::Binary).0, 50);
}
