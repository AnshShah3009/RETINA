//! LBD compares gradient magnitudes `|gx| + |gy|` on either side of a line.
//! A magnitude does not change when the image is inverted (`255 - I` negates
//! the gradient), so neither may the descriptor.
//!
//! The extractor took its gradients from `cv_imgproc::sobel`, which returns
//! `u8` images: every negative derivative saturates to 0 before the `abs()`.
//! A bright-to-dark edge therefore had no gradient at all and described as
//! all-zero bytes, while the same edge dark-to-bright set the bits.

use cv_features::{Lbd, LbdParams};
use cv_imgproc::hough::LineSegment;
use image::GrayImage;

fn step(left: u8, right: u8) -> GrayImage {
    GrayImage::from_fn(40, 40, |x, _| {
        image::Luma([if x < 20 { left } else { right }])
    })
}

#[test]
fn descriptor_is_invariant_to_intensity_inversion() {
    // A vertical segment two pixels left of the step, so the band straddles
    // the edge asymmetrically and the comparisons are not all ties.
    let seg = LineSegment {
        x1: 18.0,
        y1: 5.0,
        x2: 18.0,
        y2: 35.0,
    };
    let lbd = Lbd::new(LbdParams::default());
    let rising = &lbd.compute(&step(50, 200), &[seg])[0].data;
    let falling = &lbd.compute(&step(200, 50), &[seg])[0].data;
    assert!(
        rising.iter().any(|&b| b != 0),
        "dark-to-bright edge produced an empty descriptor"
    );
    assert_eq!(
        rising, falling,
        "inverting the image changed the descriptor (bright-to-dark edge lost its gradient)"
    );
}
