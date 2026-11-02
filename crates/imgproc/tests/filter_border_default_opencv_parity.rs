//! The border-less filter entry points use OpenCV's default border,
//! `BORDER_REFLECT_101` (`gfedcb|abcdefgh|gfedcba`).
//!
//! `sobel`, `scharr`, `laplacian`, `gaussian_blur`, `box_blur`, `convolve` (and
//! `sharpen`, through `convolve`) used to pad with `Replicate`
//! (`aaaaaa|abcdefgh|hhhhhhh`), so every pixel within the kernel radius of an
//! edge differed from the same call in OpenCV. On the 9x7 image below,
//! `cv2.Sobel(img, CV_8U, 1, 0)` is `0` down the whole first and last column
//! (a mirrored edge has no horizontal gradient) where replicate gave up to 255,
//! and `cv2.blur(img, (3, 3))` differs from the replicate blur in 27 of the 28
//! border pixels.
//!
//! The `*_with_border` variants still take any [`BorderMode`].
//!
//! Reference values: OpenCV 4.13 with default borders —
//! `cv2.Sobel(img, cv2.CV_8U, dx, dy, ksize=3)`, `cv2.Scharr(img, cv2.CV_8U, dx,
//! dy)`, `cv2.Laplacian(img, cv2.CV_8U, ksize=1)`, `cv2.blur(img, (k, k))`,
//! `cv2.GaussianBlur(img, (0, 0), 1.0)` and `cv2.filter2D(img, -1, k)`.

use cv_imgproc::{
    BorderMode, Kernel, box_blur, box_blur_with_border, convolve, gaussian_blur, laplacian, scharr,
    sobel, sobel_with_border,
};
use image::GrayImage;

const W: usize = 9;
const H: usize = 7;

fn lcg(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

fn input() -> GrayImage {
    let img = GrayImage::from_raw(W as u32, H as u32, lcg(W * H, 3)).unwrap();
    assert_eq!(img.as_raw().as_slice(), INPUT_CHECK.as_flattened());
    img
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
fn sobel_default_border_is_reflect_101() {
    let (gx, gy) = sobel(&input());
    check("sobel dx", &gx, &SOBEL_X);
    check("sobel dy", &gy, &SOBEL_Y);
}

#[test]
fn sobel_explicit_replicate_still_replicates() {
    let (gx, _) = sobel_with_border(&input(), BorderMode::Replicate);
    check("sobel dx replicate", &gx, &SOBEL_X_REPLICATE);
}

#[test]
fn scharr_default_border_is_reflect_101() {
    let (gx, gy) = scharr(&input());
    check("scharr dx", &gx, &SCHARR_X);
    check("scharr dy", &gy, &SCHARR_Y);
}

#[test]
fn laplacian_default_border_is_reflect_101() {
    check("laplacian", &laplacian(&input()), &LAPLACIAN);
}

/// Odd sizes only: for an even box OpenCV's 8-bit `ColumnSum<ushort, uchar>`
/// divides with a fixed-point reciprocal whose rounding offset is one too high
/// (`(s + 9) * 4096 >> 16` for 4x4), so a mean of `x.4375` rounds up. That
/// moves 3 of the 63 pixels of a 4x4 blur here by 1, independent of the border.
#[test]
fn box_blur_default_border_is_reflect_101() {
    check("box 3", &box_blur(&input(), 3), &BOX_3);
    check("box 5", &box_blur(&input(), 5), &BOX_5);
}

#[test]
fn box_blur_explicit_replicate_still_replicates() {
    check(
        "box 3 replicate",
        &box_blur_with_border(&input(), 3, BorderMode::Replicate),
        &BOX_3_REPLICATE,
    );
}

/// `GAUSS_1` is `np.round(cv2.GaussianBlur(img.astype(np.float32), (7, 7),
/// 1.0))`: the exact filter, rounded. OpenCV's 8-bit path instead runs a
/// 16-bit fixed-point kernel and lands within 1 of that (`GAUSS_1_U8`); that
/// rounding difference is separate from the border, which used to move edge
/// pixels by up to 28.
#[test]
fn gaussian_blur_default_border_is_reflect_101() {
    let got = gaussian_blur(&input(), 1.0);
    check("gaussian sigma 1", &got, &GAUSS_1);
    for y in 0..H {
        for x in 0..W {
            let g = got.get_pixel(x as u32, y as u32)[0];
            assert!(
                g.abs_diff(GAUSS_1_U8[y][x]) <= 1,
                "gaussian sigma 1 at ({x}, {y}): {g} vs cv2 u8 {}",
                GAUSS_1_U8[y][x]
            );
        }
    }
}

#[test]
fn convolve_default_border_is_reflect_101() {
    let k = Kernel::from_slice(
        &[1.0, 2.0, 0.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0].map(|v: f32| v / 7.0),
        3,
        3,
    );
    check("convolve", &convolve(&input(), &k), &FILTER2D);
}

const INPUT_CHECK: [[u8; W]; H] = [
    [60, 140, 223, 199, 98, 221, 163, 99, 8],
    [193, 40, 202, 139, 189, 177, 233, 42, 91],
    [71, 78, 153, 58, 105, 118, 93, 151, 200],
    [233, 225, 252, 27, 124, 66, 50, 46, 100],
    [45, 163, 172, 1, 108, 193, 70, 124, 228],
    [41, 73, 168, 48, 148, 183, 189, 13, 203],
    [106, 217, 80, 45, 244, 107, 26, 134, 37],
];
const SOBEL_X: [[u8; W]; H] = [
    [0, 255, 255, 0, 120, 218, 0, 0, 0],
    [0, 255, 237, 0, 158, 141, 0, 0, 0],
    [0, 192, 0, 0, 197, 0, 0, 122, 0],
    [0, 247, 0, 0, 255, 0, 0, 255, 0],
    [0, 255, 0, 0, 255, 0, 0, 255, 0],
    [0, 255, 0, 60, 255, 0, 0, 197, 0],
    [0, 202, 0, 255, 255, 0, 0, 50, 0],
];
const SOBEL_Y: [[u8; W]; H] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 226, 255],
    [255, 255, 173, 0, 0, 0, 0, 0, 26],
    [118, 163, 66, 0, 24, 130, 2, 0, 2],
    [0, 0, 0, 0, 186, 255, 255, 176, 140],
    [230, 77, 0, 132, 230, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const SCHARR_X: [[u8; W]; H] = [
    [0, 255, 255, 0, 255, 255, 0, 0, 0],
    [0, 255, 255, 0, 255, 255, 0, 0, 0],
    [0, 255, 0, 0, 255, 0, 0, 255, 0],
    [0, 255, 0, 0, 255, 0, 0, 255, 0],
    [0, 255, 0, 0, 255, 0, 0, 255, 0],
    [0, 255, 0, 100, 255, 0, 0, 255, 0],
    [0, 255, 0, 255, 255, 0, 0, 194, 0],
];
const SCHARR_Y: [[u8; W]; H] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 255, 255],
    [255, 255, 255, 0, 0, 0, 0, 0, 114],
    [250, 255, 255, 0, 84, 255, 0, 0, 118],
    [0, 0, 0, 30, 255, 255, 255, 255, 255],
    [255, 255, 0, 255, 255, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const LAPLACIAN: [[u8; W]; H] = [
    [255, 0, 0, 0, 255, 0, 134, 0, 255],
    [0, 255, 0, 92, 0, 53, 0, 255, 0],
    [255, 177, 0, 192, 69, 0, 180, 0, 0],
    [0, 0, 0, 255, 0, 221, 75, 241, 120],
    [255, 0, 0, 255, 34, 0, 255, 0, 0],
    [133, 255, 0, 170, 0, 0, 0, 255, 0],
    [92, 0, 255, 240, 0, 208, 255, 0, 255],
];
const BOX_3: [[u8; W]; H] = [
    [98, 144, 147, 176, 170, 187, 154, 111, 62],
    [93, 129, 137, 152, 145, 155, 144, 120, 98],
    [131, 161, 130, 139, 111, 128, 108, 112, 97],
    [142, 155, 125, 111, 89, 103, 101, 118, 130],
    [138, 152, 125, 116, 100, 126, 104, 114, 100],
    [122, 118, 107, 113, 120, 141, 115, 114, 112],
    [102, 107, 102, 122, 128, 157, 115, 112, 85],
];
const BOX_5: [[u8; W]; H] = [
    [128, 115, 127, 136, 154, 136, 135, 127, 128],
    [154, 136, 143, 143, 153, 127, 124, 111, 112],
    [156, 133, 132, 139, 137, 116, 124, 117, 111],
    [145, 118, 122, 128, 131, 108, 130, 118, 114],
    [146, 120, 119, 126, 113, 99, 122, 112, 102],
    [146, 119, 120, 131, 121, 101, 125, 107, 99],
    [127, 102, 105, 128, 122, 108, 139, 125, 112],
];
const GAUSS_1: [[u8; W]; H] = [
    [116, 129, 156, 162, 161, 169, 151, 109, 87],
    [116, 126, 146, 148, 149, 155, 143, 113, 100],
    [131, 137, 138, 123, 119, 123, 117, 114, 115],
    [151, 155, 142, 108, 103, 107, 100, 107, 117],
    [132, 140, 130, 103, 110, 122, 110, 111, 121],
    [111, 119, 116, 106, 126, 137, 119, 109, 114],
    [111, 117, 113, 110, 134, 139, 116, 103, 104],
];
const GAUSS_1_U8: [[u8; W]; H] = [
    [116, 129, 155, 162, 162, 169, 152, 109, 87],
    [116, 126, 146, 148, 149, 155, 143, 113, 100],
    [131, 137, 138, 123, 119, 123, 117, 114, 115],
    [151, 155, 141, 108, 103, 106, 100, 107, 117],
    [133, 140, 130, 103, 110, 121, 110, 111, 121],
    [111, 119, 116, 106, 126, 137, 119, 109, 113],
    [111, 117, 113, 110, 134, 139, 117, 103, 104],
];
const FILTER2D: [[u8; W]; H] = [
    [72, 139, 104, 197, 141, 200, 183, 171, 43],
    [104, 78, 134, 162, 100, 127, 178, 113, 110],
    [188, 164, 177, 188, 93, 141, 118, 74, 65],
    [126, 72, 157, 114, 75, 107, 144, 111, 157],
    [124, 138, 167, 122, 74, 145, 96, 115, 57],
    [156, 96, 189, 80, 61, 178, 138, 35, 187],
    [48, 88, 102, 102, 115, 175, 145, 155, 53],
];
const SOBEL_X_REPLICATE: [[u8; W]; H] = [
    [87, 255, 255, 0, 104, 239, 0, 0, 0],
    [0, 255, 237, 0, 158, 141, 0, 0, 56],
    [0, 192, 0, 0, 197, 0, 0, 122, 201],
    [109, 247, 0, 0, 255, 0, 0, 255, 255],
    [255, 255, 0, 0, 255, 0, 0, 255, 255],
    [255, 255, 0, 60, 255, 0, 0, 197, 255],
    [255, 49, 0, 255, 255, 0, 0, 47, 0],
];
const BOX_3_REPLICATE: [[u8; W]; H] = [
    [105, 142, 167, 174, 171, 174, 158, 101, 50],
    [101, 129, 137, 152, 145, 155, 144, 120, 99],
    [149, 161, 130, 139, 111, 128, 108, 112, 113],
    [129, 155, 125, 111, 89, 103, 101, 118, 153],
    [122, 152, 125, 116, 100, 126, 104, 114, 138],
    [93, 118, 107, 113, 120, 141, 115, 114, 134],
    [113, 121, 108, 122, 130, 142, 102, 89, 93],
];
