//! `convert_rgb_to_gray` (and the slice forms `rgb_to_gray_u8` /
//! `bgr_to_gray_u8`) must give `cv2.cvtColor(img, COLOR_RGB2GRAY /
//! COLOR_BGR2GRAY)` for every 8-bit colour.
//!
//! OpenCV 4.13 converts 8-bit colour to grey in 15-bit fixed point:
//! `(9798 R + 19235 G + 3735 B + 2^14) >> 15` - checked against `cv2` on all
//! 2^24 colours, both orders, with zero mismatches. (The 14-bit
//! `4899/9617/1868` table of older OpenCV releases mismatches 43,864 of them.)
//! The previous implementation evaluated `0.299 R + 0.587 G + 0.114 B` in f32
//! and rounded half away from zero: 21,247 of the 2^24 colours (0.127%) came
//! out one grey level off, e.g. `(0, 0, 250)` gave 29 where OpenCV gives 28.
//!
//! Reference values: OpenCV 4.13 on the 4096x4096 image holding every colour
//! once (pixel `i` is `(i >> 16, (i >> 8) & 255, i & 255)`), summarised by a
//! weighted checksum, plus 80 individual colours: the first 40 the float path
//! got wrong and 40 spread evenly over all 21,247 it got wrong.

use cv_imgproc::{bgr_to_gray_u8, convert_rgb_to_gray, rgb_to_gray_u8};
use image::RgbImage;

/// `sum(gray[i] * ((i * 2654435761) mod 2^32))` mod 2^64 over OpenCV's output.
const OPENCV_WEIGHTED_CHECKSUM: u64 = 4593675103442398182;
/// `sum(gray[i])` over OpenCV's output.
const OPENCV_SUM: u64 = 2139096404;

fn checksums(gray: &[u8]) -> (u64, u64) {
    let mut weighted = 0u64;
    let mut sum = 0u64;
    for (i, &g) in gray.iter().enumerate() {
        let w = (i as u64).wrapping_mul(2654435761) & 0xffff_ffff;
        weighted = weighted.wrapping_add(g as u64 * w);
        sum += g as u64;
    }
    (weighted, sum)
}

fn every_colour_rgb() -> Vec<u8> {
    (0..1u32 << 24)
        .flat_map(|i| [(i >> 16) as u8, (i >> 8) as u8, i as u8])
        .collect()
}

#[test]
fn every_rgb_colour_matches_opencv() {
    let raw = every_colour_rgb();
    let img = RgbImage::from_raw(4096, 4096, raw).unwrap();
    let gray = convert_rgb_to_gray(&img);
    assert_eq!(
        checksums(gray.as_raw()),
        (OPENCV_WEIGHTED_CHECKSUM, OPENCV_SUM)
    );
}

#[test]
fn every_bgr_colour_matches_opencv() {
    // Same colours, stored B, G, R: OpenCV's BGR2GRAY gives the same grey.
    let mut raw = every_colour_rgb();
    for px in raw.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    let mut gray = vec![0u8; 1 << 24];
    bgr_to_gray_u8(&raw, &mut gray);
    assert_eq!(checksums(&gray), (OPENCV_WEIGHTED_CHECKSUM, OPENCV_SUM));
}

#[test]
fn individual_colours_match_opencv() {
    for (rgb, want) in CASES {
        let mut g = [0u8];
        rgb_to_gray_u8(&rgb, &mut g);
        assert_eq!(g[0], want, "rgb {rgb:?}");
        let mut g = [0u8];
        bgr_to_gray_u8(&[rgb[2], rgb[1], rgb[0]], &mut g);
        assert_eq!(g[0], want, "bgr of rgb {rgb:?}");
        let img = RgbImage::from_raw(1, 1, rgb.to_vec()).unwrap();
        assert_eq!(convert_rgb_to_gray(&img).as_raw()[0], want, "image {rgb:?}");
    }
}

const CASES: [([u8; 3], u8); 80] = [
    ([0, 0, 250], 28),
    ([0, 1, 201], 23),
    ([0, 2, 152], 18),
    ([0, 4, 168], 21),
    ([0, 5, 119], 16),
    ([0, 8, 86], 14),
    ([0, 9, 230], 31),
    ([0, 11, 246], 34),
    ([0, 12, 197], 29),
    ([0, 14, 213], 32),
    ([0, 15, 164], 27),
    ([0, 18, 131], 25),
    ([0, 19, 82], 20),
    ([0, 22, 49], 18),
    ([0, 22, 242], 40),
    ([0, 23, 193], 35),
    ([0, 25, 209], 38),
    ([0, 26, 160], 33),
    ([0, 28, 176], 36),
    ([0, 29, 127], 31),
    ([0, 32, 94], 29),
    ([0, 33, 238], 46),
    ([0, 35, 254], 49),
    ([0, 36, 205], 44),
    ([0, 38, 221], 47),
    ([0, 39, 172], 42),
    ([0, 42, 139], 40),
    ([0, 43, 90], 35),
    ([0, 46, 57], 33),
    ([0, 46, 250], 55),
    ([0, 47, 201], 50),
    ([0, 49, 217], 53),
    ([0, 50, 168], 48),
    ([0, 52, 184], 51),
    ([0, 53, 135], 46),
    ([0, 56, 102], 44),
    ([0, 57, 246], 61),
    ([0, 60, 213], 59),
    ([0, 62, 229], 62),
    ([0, 63, 180], 57),
    ([0, 0, 250], 28),
    ([4, 54, 172], 52),
    ([8, 140, 245], 112),
    ([13, 22, 243], 44),
    ([17, 152, 221], 119),
    ([22, 88, 169], 77),
    ([27, 46, 109], 47),
    ([32, 26, 155], 42),
    ([37, 49, 234], 66),
    ([42, 108, 189], 97),
    ([47, 244, 37], 162),
    ([53, 109, 155], 97),
    ([59, 29, 174], 54),
    ([64, 254, 169], 187),
    ([71, 1, 199], 44),
    ([77, 27, 102], 50),
    ([83, 105, 132], 101),
    ([90, 13, 140], 50),
    ([96, 221, 255], 187),
    ([103, 207, 221], 177),
    ([110, 252, 6], 182),
    ([118, 27, 205], 74),
    ([125, 130, 244], 141),
    ([132, 233, 90], 187),
    ([140, 135, 214], 145),
    ([148, 95, 13], 102),
    ([156, 17, 227], 82),
    ([164, 39, 198], 94),
    ([171, 244, 10], 196),
    ([179, 164, 15], 152),
    ([187, 72, 73], 107),
    ([194, 199, 120], 189),
    ([202, 77, 236], 132),
    ([209, 142, 111], 159),
    ([216, 215, 15], 193),
    ([223, 174, 6], 170),
    ([230, 119, 34], 143),
    ([236, 241, 162], 231),
    ([243, 159, 215], 191),
    ([249, 240, 19], 218),
];
