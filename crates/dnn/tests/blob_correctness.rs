//! Regression for `blob_to_image`'s silent all-black fallback.
//!
//! `image::GrayImage::from_raw` returns `None` when the buffer length does not
//! match `width * height`; the fallback was `GrayImage::new(width, height)`, so
//! a caller that handed over the wrong number of values got a well-formed black
//! frame back with every value it supplied discarded — indistinguishable
//! downstream from a genuinely black observation.

use cv_dnn::blob::{blob_to_image, image_to_blob};
use image::GrayImage;

#[test]
#[should_panic(expected = "blob_to_image")]
fn a_size_mismatch_is_reported_not_painted_black() {
    // 3 values for a 2x2 image: used to return an all-black 2x2 image.
    let _ = blob_to_image(&[0.25, 0.5, 0.75], 2, 2);
}

#[test]
#[should_panic(expected = "blob_to_image")]
fn an_oversized_blob_is_reported_too() {
    let _ = blob_to_image(&[1.0, 1.0, 1.0, 1.0, 1.0], 2, 2);
}

#[test]
fn exact_buffers_still_round_trip() {
    // Control: the fix may not break the well-formed case.
    let mut original = GrayImage::new(3, 2);
    for y in 0..2u32 {
        for x in 0..3u32 {
            original.put_pixel(x, y, image::Luma([(20 + 30 * (x + y)) as u8]));
        }
    }
    let blob = image_to_blob(&original);
    assert_eq!(blob.len(), 6);

    let recovered = blob_to_image(&blob, 3, 2);
    assert_eq!(recovered.width(), 3);
    assert_eq!(recovered.height(), 2);
    for y in 0..2u32 {
        for x in 0..3u32 {
            let expected = original.get_pixel(x, y)[0];
            let got = recovered.get_pixel(x, y)[0];
            assert!(
                (i32::from(expected) - i32::from(got)).abs() <= 1,
                "({x},{y}): {expected} -> {got}"
            );
        }
    }

    // Row-major order, with mid-range values kept (not collapsed to 0/255).
    let mid = blob_to_image(&[0.5, 0.25, 0.0, 0.75], 2, 2);
    assert_eq!(mid.get_pixel(0, 0)[0], 127); // blob[0], truncated 127.5
    assert_eq!(mid.get_pixel(1, 0)[0], 63); // blob[1], truncated 63.75
    assert_eq!(mid.get_pixel(0, 1)[0], 0); // blob[2]
    assert_eq!(mid.get_pixel(1, 1)[0], 191); // blob[3], truncated 191.25
}
