//! `INTER_NEAREST` tie rounding in `remap`, `warp_perspective_ex` and
//! `warp_affine_ex`, against cv2 4.13.
//!
//! A nearest-neighbour sample exactly half-way between two pixels is common:
//! every odd output column of an integer 2x upscale lands on one. OpenCV
//! resolves it two ways: `warpAffine` rounds halves up (`floor(x + 0.5)`,
//! from its fixed-point `>> AB_BITS`), `warpPerspective` and `remap` round
//! halves to even (`saturate_cast<int>`). The Rust samplers used
//! `f32::round` for all three, so `remap` and `warpPerspective` picked the
//! odd neighbour on every `2.5`, `4.5`, ... and `warpAffine` sent `-0.5` to
//! the border instead of pixel 0.
//!
//! Input: 8x3, `a[y][x] = 10 * (x + 1) + y`. Border constant 7. Tables are
//! cv2 output.

use cv_imgproc::{BorderMode, Interpolation, remap, warp_affine_ex, warp_perspective_ex};
use image::GrayImage;
use nalgebra::Matrix3;

fn input() -> GrayImage {
    GrayImage::from_fn(8, 3, |x, y| image::Luma([(10 * (x + 1) + y) as u8]))
}

fn rows(img: &GrayImage) -> Vec<Vec<u8>> {
    img.as_raw()
        .chunks(img.width() as usize)
        .map(|r| r.to_vec())
        .collect()
}

#[test]
fn remap_rounds_half_to_even() {
    // cv2.remap(a, mx, my, INTER_NEAREST, BORDER_CONSTANT, 7)
    let xs = [0.5f32, 1.5, 2.5, 3.5, -0.5, -0.4, 4.5, 6.5];
    let ys = [0.0f32, 0.5, 1.5];
    let map_x: Vec<f32> = ys.iter().flat_map(|_| xs).collect();
    let map_y: Vec<f32> = ys.iter().flat_map(|&y| [y; 8]).collect();
    let out = remap(
        &input(),
        &map_x,
        &map_y,
        8,
        3,
        Interpolation::Nearest,
        BorderMode::Constant(7),
    );
    assert_eq!(
        rows(&out),
        vec![
            vec![10, 30, 30, 50, 10, 10, 50, 70],
            vec![10, 30, 30, 50, 10, 10, 50, 70],
            vec![12, 32, 32, 52, 12, 12, 52, 72],
        ]
    );
}

#[test]
fn warp_perspective_upscale_rounds_half_to_even() {
    // cv2.warpPerspective(a, diag(2, 2, 1), (16, 6), INTER_NEAREST, ..., 7)
    let m = Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
    let out = warp_perspective_ex(
        &input(),
        &m,
        16,
        6,
        Interpolation::Nearest,
        BorderMode::Constant(7),
    );
    let r0: Vec<u8> = vec![
        10, 10, 20, 30, 30, 30, 40, 50, 50, 50, 60, 70, 70, 70, 80, 7,
    ];
    let shift = |d: u8| r0.iter().map(|&v| if v == 7 { 7 } else { v + d }).collect();
    assert_eq!(
        rows(&out),
        vec![
            r0.clone(),
            r0.clone(),
            shift(1),
            shift(2),
            shift(2),
            shift(2)
        ]
    );
}

#[test]
fn warp_perspective_half_pixel_shift_rounds_half_to_even() {
    // cv2.warpPerspective(a, [[1,0,.5],[0,1,.5],[0,0,1]], (8, 3), INTER_NEAREST)
    let m = Matrix3::new(1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0);
    let out = warp_perspective_ex(
        &input(),
        &m,
        8,
        3,
        Interpolation::Nearest,
        BorderMode::Constant(7),
    );
    assert_eq!(
        rows(&out),
        vec![
            vec![10, 10, 30, 30, 50, 50, 70, 70],
            vec![10, 10, 30, 30, 50, 50, 70, 70],
            vec![12, 12, 32, 32, 52, 52, 72, 72],
        ]
    );
}

#[test]
fn warp_affine_rounds_half_up() {
    // cv2.warpAffine(a, [[1,0,.5],[0,1,.5]], (8, 3), INTER_NEAREST, ..., 7):
    // the source of output (0, 0) is (-0.5, -0.5), which cv2 reads as pixel
    // (0, 0); half-away-from-zero rounding sent it to the border (7).
    let out = warp_affine_ex(
        &input(),
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
        8,
        3,
        Interpolation::Nearest,
        BorderMode::Constant(7),
    );
    assert_eq!(
        rows(&out),
        vec![
            vec![10, 20, 30, 40, 50, 60, 70, 80],
            vec![11, 21, 31, 41, 51, 61, 71, 81],
            vec![12, 22, 32, 42, 52, 62, 72, 82],
        ]
    );
    // cv2.warpAffine(a, [[2,0,0],[0,2,0]], (16, 6), INTER_NEAREST, ..., 7)
    let up = warp_affine_ex(
        &input(),
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        16,
        6,
        Interpolation::Nearest,
        BorderMode::Constant(7),
    );
    let r0: Vec<u8> = vec![
        10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 7,
    ];
    let shift = |d: u8| r0.iter().map(|&v| if v == 7 { 7 } else { v + d }).collect();
    assert_eq!(
        rows(&up),
        vec![
            r0.clone(),
            shift(1),
            shift(1),
            shift(2),
            shift(2),
            vec![7; 16]
        ]
    );
}
