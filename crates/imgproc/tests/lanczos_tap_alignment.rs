//! Lanczos sampling must be centred on the sample point.
//!
//! The 8-tap Lanczos-4 kernel weights tap `k` for the source pixel at offset
//! `k - 3` from `floor(x)`, but the sampler read offset `k - 2`. Every Lanczos
//! warp and remap was therefore shifted by one whole pixel in x and in y: an
//! identity warp of `[0, 10, 20, ..]` returned `[10, 20, 30, ..]`. Cubic, on
//! the same path, returned the input, which is what isolated the offset.

use cv_imgproc::{remap, warp_affine_ex, BorderMode, Interpolation};
use image::GrayImage;

fn ramp() -> GrayImage {
    GrayImage::from_fn(16, 16, |x, y| image::Luma([(x * 10 + y) as u8]))
}

#[test]
fn identity_remap_with_lanczos_returns_the_input() {
    let src = ramp();
    let (w, h) = src.dimensions();
    let mut mx = Vec::new();
    let mut my = Vec::new();
    for y in 0..h {
        for x in 0..w {
            mx.push(x as f32);
            my.push(y as f32);
        }
    }
    for interp in [Interpolation::Cubic, Interpolation::Lanczos] {
        let out = remap(&src, &mx, &my, w, h, interp, BorderMode::Replicate);
        assert_eq!(
            out.as_raw(),
            src.as_raw(),
            "{interp:?} identity remap moved pixels"
        );
    }
}

#[test]
fn identity_warp_with_lanczos_returns_the_input() {
    let src = ramp();
    let (w, h) = src.dimensions();
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let out = warp_affine_ex(
        &src,
        id,
        w,
        h,
        Interpolation::Lanczos,
        BorderMode::Replicate,
    );
    assert_eq!(
        out.as_raw(),
        src.as_raw(),
        "Lanczos identity warp moved pixels"
    );
}
