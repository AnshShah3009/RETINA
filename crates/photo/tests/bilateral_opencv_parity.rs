//! `cv_photo::bilateral_filter` must produce what `cv2.bilateralFilter` does
//! on an 8-bit single-channel image.
//!
//! Three things differed from OpenCV's `bilateralFilter_8u`:
//!
//! * **Kernel support.** OpenCV keeps only the offsets with
//!   `sqrt(dx² + dy²) <= radius` (a disc); the square window also summed the
//!   corners of the box. For `d = 5` that is 25 taps instead of 21, and it moved
//!   *interior* pixels - far from any border - by up to 16 grey levels.
//! * **Border.** OpenCV pads with `BORDER_DEFAULT` (reflect-101) and always
//!   averages the full disc; skipping the out-of-image taps renormalised over
//!   fewer neighbours, so the outer rows and columns were off by up to 26.
//! * **Radius from `sigma_space`.** For `d <= 0` OpenCV uses
//!   `cvRound(sigma_space * 1.5)`; `ceil` turned `sigma_space = 1.5` (2.25)
//!   into a radius of 3 instead of 2.
//!
//! Reference values: OpenCV 4.13,
//! `cv2.bilateralFilter(img, d, sigma_color, sigma_space)` on the image built by
//! [`input`].

use cv_photo::bilateral_filter;
use image::{GrayImage, Luma};

const W: u32 = 11;
const H: u32 = 9;

/// `(x*37 + y*101 + (x*y % 7)*23) % 256` - deterministic, high-contrast texture.
fn input() -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        Luma([((x * 37 + y * 101 + (x * y) % 7 * 23) % 256) as u8])
    })
}

type Case = (i32, f32, f32, [[u8; W as usize]; H as usize]);

const CASES: &[Case] = &[
    (
        5,
        30.0,
        3.0,
        [
            [20, 38, 76, 101, 138, 188, 206, 16, 41, 78, 95],
            [96, 160, 212, 35, 92, 145, 200, 104, 166, 217, 40],
            [175, 30, 118, 201, 115, 200, 37, 193, 32, 120, 208],
            [30, 154, 21, 204, 64, 234, 107, 40, 154, 18, 206],
            [149, 22, 233, 118, 87, 208, 185, 154, 25, 241, 131],
            [241, 141, 132, 125, 34, 23, 16, 241, 143, 136, 136],
            [99, 20, 32, 40, 53, 70, 82, 92, 26, 32, 32],
            [209, 225, 23, 45, 76, 117, 154, 196, 225, 24, 39],
            [61, 98, 165, 213, 48, 85, 141, 65, 109, 166, 211],
        ],
    ),
    (
        0,
        40.0,
        2.0,
        [
            [30, 42, 77, 115, 148, 183, 198, 29, 45, 77, 111],
            [102, 160, 201, 43, 94, 149, 194, 108, 168, 209, 39],
            [190, 34, 112, 189, 116, 193, 40, 198, 37, 116, 195],
            [41, 152, 27, 196, 60, 223, 106, 46, 156, 24, 197],
            [142, 28, 223, 117, 83, 208, 181, 148, 31, 229, 131],
            [232, 141, 131, 118, 37, 38, 28, 232, 143, 138, 136],
            [94, 28, 34, 46, 57, 69, 81, 96, 31, 33, 40],
            [194, 220, 31, 50, 79, 112, 150, 185, 217, 30, 42],
            [33, 88, 157, 212, 44, 82, 134, 52, 98, 164, 219],
        ],
    ),
    (
        0,
        50.0,
        1.5,
        [
            [33, 49, 77, 104, 133, 178, 196, 31, 53, 77, 97],
            [93, 155, 198, 58, 102, 152, 193, 101, 160, 205, 59],
            [166, 39, 122, 184, 117, 193, 51, 181, 41, 125, 193],
            [39, 154, 27, 191, 73, 222, 112, 52, 154, 24, 196],
            [149, 29, 211, 114, 85, 204, 183, 155, 33, 221, 129],
            [217, 142, 127, 113, 44, 35, 29, 220, 144, 131, 129],
            [93, 27, 37, 47, 57, 67, 84, 95, 34, 38, 41],
            [203, 213, 31, 49, 72, 108, 143, 186, 210, 30, 41],
            [75, 96, 164, 197, 58, 94, 138, 79, 103, 167, 194],
        ],
    ),
    (
        9,
        60.0,
        1.5,
        [
            [38, 56, 82, 113, 144, 172, 190, 42, 58, 81, 112],
            [100, 149, 185, 60, 101, 149, 185, 111, 159, 194, 53],
            [174, 49, 114, 176, 121, 183, 56, 184, 51, 118, 182],
            [57, 146, 38, 182, 73, 211, 109, 63, 152, 36, 186],
            [143, 42, 204, 111, 85, 193, 172, 147, 45, 208, 125],
            [210, 137, 124, 110, 49, 46, 41, 211, 141, 132, 124],
            [93, 37, 44, 53, 62, 72, 85, 97, 42, 44, 50],
            [187, 206, 39, 56, 78, 106, 140, 175, 201, 38, 50],
            [49, 91, 146, 183, 54, 86, 130, 66, 99, 155, 197],
        ],
    ),
];

#[test]
fn bilateral_filter_matches_opencv_exactly() {
    let img = input();
    let mut failures = Vec::new();
    for (d, sigma_color, sigma_space, expected) in CASES {
        let out = bilateral_filter(&img, *d, *sigma_color, *sigma_space);
        let mut worst = (0i32, 0u32, 0u32);
        for y in 0..H {
            for x in 0..W {
                let got = out.get_pixel(x, y)[0] as i32;
                let want = expected[y as usize][x as usize] as i32;
                if (got - want).abs() > worst.0 {
                    worst = ((got - want).abs(), x, y);
                }
            }
        }
        if worst.0 != 0 {
            failures.push(format!(
                "d={d} sigma_color={sigma_color} sigma_space={sigma_space}: off OpenCV by {} at \
                 (x={}, y={})",
                worst.0, worst.1, worst.2
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A one-pixel-wide image: reflect-101 of a length-1 axis is that single
/// column (OpenCV's `borderInterpolate` special case), not an out-of-range
/// index.
///
/// `cv2.bilateralFilter(np.array([[10], [200], [90], [40]], np.uint8), 5, 30, 3)`
#[test]
fn single_column_image_matches_opencv() {
    let img = GrayImage::from_fn(1, 4, |_, y| Luma([[10u8, 200, 90, 40][y as usize]]));
    let out = bilateral_filter(&img, 5, 30.0, 3.0);
    assert_eq!(out.as_raw(), &vec![11u8, 200, 84, 52]);
}

/// OpenCV clamps the radius to at least 1 (`radius = MAX(radius, 1)`), so
/// `d = 1` still filters; the old `radius = d / 2 = 0` returned the input.
///
/// `cv2.bilateralFilter(np.array([[0,100,0],[100,0,100],[0,100,0]], np.uint8), 1, 100, 1)`
#[test]
fn diameter_one_still_filters_like_opencv() {
    let img = GrayImage::from_raw(3, 3, vec![0u8, 100, 0, 100, 0, 100, 0, 100, 0]).unwrap();
    let out = bilateral_filter(&img, 1, 100.0, 1.0);
    assert_eq!(out.as_raw(), &vec![60u8, 40, 60, 40, 60, 40, 60, 40, 60]);
}
