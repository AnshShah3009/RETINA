//! `hit_or_miss` must give `cv2.morphologyEx(src, cv2.MORPH_HITMISS, kernel)`.
//!
//! OpenCV computes it as `erode(src, k == 1) & erode(~src, k == -1)` with
//! erosion's default border, which never wins a minimum: an out-of-image tap
//! is **don't care** for both the hit and the miss set. The previous version
//! treated it as background, so a `1` reaching off the frame always failed
//! (e.g. `HIT_ONLY` below found nothing on the outer ring, where OpenCV
//! matches) while a `-1` there always passed. Also from OpenCV:
//!
//! * for a non-binary image the result is that bitwise AND of the two
//!   erosions, not a 0/255 mask;
//! * an all-zero kernel returns the image unchanged.
//!
//! Reference values: OpenCV 4.13 on the images below (`BINARY` is
//! `lcg > 110 ? 255 : 0`, `GRAY` raw LCG bytes).

use cv_imgproc::hit_or_miss;
use image::GrayImage;

const W: usize = 10;
const H: usize = 8;

fn image(rows: &[[u8; W]; H]) -> GrayImage {
    GrayImage::from_raw(W as u32, H as u32, rows.as_flattened().to_vec()).unwrap()
}

fn check(name: &str, got: &GrayImage, want: &[[u8; W]; H]) {
    let diffs: Vec<(usize, usize, u8, u8)> = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter_map(|(x, y)| {
            let g = got.get_pixel(x as u32, y as u32)[0];
            (g != want[y][x]).then_some((x, y, g, want[y][x]))
        })
        .collect();
    assert!(
        diffs.is_empty(),
        "{name}: {} pixels differ from OpenCV, first (x, y, got, opencv): {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(8)]
    );
}

#[test]
fn binary_images_match_opencv() {
    let src = image(&BINARY);
    let (k, kw, kh) = K_ISOLATED;
    check("isolated", &hit_or_miss(&src, &k, kw, kh), &BIN_ISOLATED);
    let (k, kw, kh) = K_CORNER;
    check("corner", &hit_or_miss(&src, &k, kw, kh), &BIN_CORNER);
    let (k, kw, kh) = K_HIT_ONLY;
    check("hit only", &hit_or_miss(&src, &k, kw, kh), &BIN_HIT_ONLY);
    let (k, kw, kh) = K_WIDE;
    check("wide", &hit_or_miss(&src, &k, kw, kh), &BIN_WIDE);
}

#[test]
fn grey_images_match_opencv() {
    let src = image(&GRAY);
    let (k, kw, kh) = K_ISOLATED;
    check("isolated", &hit_or_miss(&src, &k, kw, kh), &GRAY_ISOLATED);
    let (k, kw, kh) = K_CORNER;
    check("corner", &hit_or_miss(&src, &k, kw, kh), &GRAY_CORNER);
    let (k, kw, kh) = K_HIT_ONLY;
    check("hit only", &hit_or_miss(&src, &k, kw, kh), &GRAY_HIT_ONLY);
    let (k, kw, kh) = K_WIDE;
    check("wide", &hit_or_miss(&src, &k, kw, kh), &GRAY_WIDE);
}

/// A lone foreground pixel in the frame corner is isolated for OpenCV: the
/// five ring taps off the frame are don't-care.
#[test]
fn a_corner_pixel_is_isolated() {
    let mut img = GrayImage::new(5, 5);
    img.put_pixel(0, 0, image::Luma([255]));
    let (k, kw, kh) = K_ISOLATED;
    let out = hit_or_miss(&img, &k, kw, kh);
    assert_eq!(out.get_pixel(0, 0)[0], 255);
    assert!(out.as_raw()[1..].iter().all(|&v| v == 0));
}

#[test]
fn an_all_zero_kernel_returns_the_image() {
    let src = image(&GRAY);
    assert_eq!(hit_or_miss(&src, &[0; 9], 3, 3).as_raw(), src.as_raw());
}

const BINARY: [[u8; W]; H] = [
    [0, 0, 0, 255, 0, 0, 0, 255, 0, 255],
    [0, 0, 0, 255, 255, 0, 255, 255, 255, 0],
    [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
    [255, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    [255, 255, 0, 255, 255, 0, 0, 0, 255, 255],
    [0, 0, 255, 255, 255, 0, 255, 0, 255, 0],
    [255, 255, 0, 255, 0, 0, 255, 255, 255, 0],
    [255, 255, 255, 0, 0, 255, 255, 0, 0, 0],
];
const GRAY: [[u8; W]; H] = [
    [63, 69, 106, 227, 96, 75, 145, 223, 75, 122],
    [197, 86, 217, 75, 138, 229, 195, 208, 60, 232],
    [84, 29, 132, 157, 226, 216, 194, 236, 45, 129],
    [187, 36, 40, 99, 26, 19, 224, 76, 160, 102],
    [101, 113, 15, 46, 3, 34, 50, 234, 31, 51],
    [124, 136, 144, 62, 51, 46, 68, 28, 31, 198],
    [131, 73, 34, 148, 46, 255, 67, 94, 191, 68],
    [11, 223, 242, 65, 55, 154, 77, 131, 39, 62],
];
const K_ISOLATED: ([i8; 9], usize, usize) = ([-1, -1, -1, -1, 1, -1, -1, -1, -1], 3, 3);
const BIN_ISOLATED: [[u8; W]; H] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const GRAY_ISOLATED: [[u8; W]; H] = [
    [58, 4, 8, 34, 0, 10, 16, 15, 3, 18],
    [129, 6, 24, 8, 10, 5, 3, 16, 16, 104],
    [16, 4, 4, 29, 2, 24, 2, 12, 1, 1],
    [138, 4, 32, 1, 24, 17, 0, 0, 0, 70],
    [68, 64, 15, 46, 0, 2, 16, 10, 21, 49],
    [116, 8, 0, 42, 0, 0, 0, 20, 21, 64],
    [0, 9, 0, 4, 0, 101, 0, 64, 57, 0],
    [0, 13, 32, 1, 0, 0, 0, 0, 0, 0],
];
const K_CORNER: ([i8; 9], usize, usize) = ([0, -1, -1, 1, 1, -1, 0, 1, 0], 3, 3);
const BIN_CORNER: [[u8; W]; H] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0, 0, 255],
    [0, 0, 0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const GRAY_CORNER: [[u8; W]; H] = [
    [58, 21, 4, 11, 32, 74, 0, 144, 4, 75],
    [0, 4, 20, 8, 10, 8, 0, 0, 5, 4],
    [16, 4, 4, 1, 26, 18, 2, 12, 5, 5],
    [33, 32, 2, 8, 1, 19, 19, 0, 30, 50],
    [68, 69, 12, 12, 1, 3, 0, 28, 31, 25],
    [116, 73, 0, 12, 0, 42, 4, 20, 24, 12],
    [3, 73, 34, 0, 0, 42, 1, 64, 33, 56],
    [0, 9, 75, 65, 0, 0, 76, 64, 0, 35],
];
const K_HIT_ONLY: ([i8; 9], usize, usize) = ([0, 1, 0, 1, 1, 1, 0, 1, 0], 3, 3);
const BIN_HIT_ONLY: [[u8; W]; H] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 255, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
];
const GRAY_HIT_ONLY: [[u8; W]; H] = [
    [63, 63, 69, 75, 75, 75, 75, 75, 60, 75],
    [63, 29, 75, 75, 75, 75, 145, 60, 45, 60],
    [29, 29, 29, 75, 26, 19, 194, 45, 45, 45],
    [36, 29, 15, 26, 3, 19, 19, 76, 31, 51],
    [101, 15, 15, 3, 3, 3, 34, 28, 31, 31],
    [101, 73, 15, 46, 3, 34, 28, 28, 28, 31],
    [11, 34, 34, 34, 46, 46, 67, 28, 31, 62],
    [11, 11, 34, 55, 46, 55, 67, 39, 39, 39],
];
const K_WIDE: ([i8; 15], usize, usize) = ([1, 0, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, 1, 0, 0], 5, 3);
const BIN_WIDE: [[u8; W]; H] = [
    [0, 0, 0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0, 255, 255],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 255],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 0, 0, 0, 0, 0, 0],
];
const GRAY_WIDE: [[u8; W]; H] = [
    [128, 21, 4, 11, 128, 96, 0, 144, 4, 75],
    [20, 28, 31, 69, 10, 0, 32, 1, 5, 60],
    [34, 32, 0, 18, 2, 3, 2, 4, 32, 45],
    [97, 96, 13, 5, 1, 18, 19, 88, 25, 51],
    [12, 4, 81, 12, 8, 3, 16, 17, 12, 31],
    [3, 73, 0, 64, 13, 4, 0, 0, 24, 31],
    [11, 129, 73, 0, 0, 44, 33, 40, 35, 28],
    [32, 9, 130, 0, 32, 33, 0, 9, 65, 39],
];
