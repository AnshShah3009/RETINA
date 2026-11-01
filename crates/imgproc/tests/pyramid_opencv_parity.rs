//! `pyr_down` / `pyr_up` against `cv2.pyrDown` / `cv2.pyrUp` (OpenCV 4.13).
//!
//! Both used to be a bilinear resize to `w / 2` (`w * 2`): pyr_down dropped
//! the odd row/column (OpenCV keeps `(w + 1) / 2`), did not low-pass with the
//! 5-tap [1 4 6 4 1] / 16 kernel, and returned a black 1x1 for a 1x10 input.
//! Reference tables are cv2 output, row-major, hex.

use cv_imgproc::{build_pyramid, pyr_down, pyr_up};
use image::GrayImage;

fn rand_img(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut s = seed;
    GrayImage::from_fn(w, h, |_, _| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        image::Luma([(s >> 24) as u8])
    })
}

fn unhex(chunks: &[&str]) -> Vec<u8> {
    let s: String = chunks.concat();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn check(name: &str, out: &GrayImage, exp: (u32, u32, &[&str])) {
    assert_eq!(out.dimensions(), (exp.0, exp.1), "{name}: size");
    let e = unhex(exp.2);
    let n_diff = out.as_raw().iter().zip(&e).filter(|(a, b)| a != b).count();
    let max_diff = out
        .as_raw()
        .iter()
        .zip(&e)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        n_diff, 0,
        "{name}: {n_diff} px differ from cv2, max |diff| {max_diff}"
    );
}

#[test]
fn pyr_down_matches_opencv() {
    check(
        "down 40x30",
        &pyr_down(&rand_img(40, 30, 31)),
        DOWN_RAND_40X30,
    );
    check(
        "down 37x29",
        &pyr_down(&rand_img(37, 29, 32)),
        DOWN_RAND_37X29,
    );
    check("down 3x2", &pyr_down(&rand_img(3, 2, 33)), DOWN_RAND_3X2);
    check("down 1x7", &pyr_down(&rand_img(1, 7, 34)), DOWN_RAND_1X7);
}

#[test]
fn pyr_up_matches_opencv() {
    check("up 13x9", &pyr_up(&rand_img(13, 9, 35)), UP_RAND_13X9);
    check("up 1x5", &pyr_up(&rand_img(1, 5, 36)), UP_RAND_1X5);
    check("up 6x1", &pyr_up(&rand_img(6, 1, 37)), UP_RAND_6X1);
    check("up 1x1", &pyr_up(&rand_img(1, 1, 38)), UP_RAND_1X1);
}

#[test]
fn thin_constant_images_keep_their_value() {
    // cv2.pyrDown(full((1, 10), 200)) is 5x1 of 200, and (10, 1) is 1x5.
    for (w, h, ew, eh) in [(10, 1, 5, 1), (1, 10, 1, 5)] {
        let out = pyr_down(&GrayImage::from_pixel(w, h, image::Luma([200])));
        assert_eq!(out.dimensions(), (ew, eh));
        assert!(out.pixels().all(|p| p[0] == 200));
    }
}

#[test]
fn zero_sized_input_gives_zero_sized_output() {
    for (w, h) in [(0, 0), (0, 4), (4, 0)] {
        let img = GrayImage::new(w, h);
        assert_eq!(pyr_down(&img).dimensions(), (w.div_ceil(2), h.div_ceil(2)));
        assert_eq!(pyr_up(&img).dimensions(), (w * 2, h * 2));
    }
}

#[test]
fn build_pyramid_halves_with_rounding_up() {
    let p = build_pyramid(&rand_img(37, 29, 1), 4);
    let dims: Vec<(u32, u32)> = p.iter().map(|i| i.dimensions()).collect();
    assert_eq!(dims, vec![(37, 29), (19, 15), (10, 8), (5, 4)]);
}

// DOWN_RAND_40X30: cv2.pyrDown(rand_img(40, 30, 31)) -> 20x15
const DOWN_RAND_40X30: (u32, u32, &[&str]) = (
    20,
    15,
    &[
        "8a7f76746d87a0849395765c8fb390a384766b70958f83788873727c8890726a88ab9f997e82614d75839778908d6b5d",
        "877a6a8a848e8d8576876e6ea0887f918785787b73576f7f7377766278a4918083898d8a919890939879848b88838c8b",
        "5f6a6877959c898a867d928da38d969c84716c7f736b60737d998f8376546f6564768e79778570708f928a7b77938e71",
        "573d4558646b64578d8a6c5a618aa9a6677c8a83707273616250686f886b658690a3bb7f7b8da88479708a63687e9868",
        "5f69617b938f908898919b7b9b75678c999e9c82557172747f957b8059769975a4827b8d978e7d8d5e698b8d6e769264",
        "6f687f6c886972898a6c638c7f747b8279527d60696f76698084786c7e66868569758386736073718a9c794c62675b4e",
        "7c767ba29f9695a5858eb087",
    ],
);
// DOWN_RAND_37X29: cv2.pyrDown(rand_img(37, 29, 32)) -> 19x15
const DOWN_RAND_37X29: (u32, u32, &[&str]) = (
    19,
    15,
    &[
        "738b947a7e859e78898e819c97677b828f765f7b7d7260747e83767e7e768a9678837c72785d9196777785635c876665",
        "7e7567808593657b739d989a94526975987f6d8c745c6d8ead7d87908a9a938356546c8d938e749baca29a8470839684",
        "8184696d737383a1966b8193a2917b5c8a9d877b88988584838591895d5e8080969a7888ad7a6ca0b1836c686a717a6e",
        "669c847b8e716a8f9d8aa2a2756571798b8e7c6e84716d7f726c8875707f827e6b66637d797b7562445b84786e947a88",
        "8a7f83846a669898846462524b888e8f8c7682958385807d7c7e9888899f765a9d8d8f5c898b81908380988a878a668f",
        "7c5584a6a49772796c727d6e7c938192836c7f796d8b887c858c806567877564988a737c7e767d9b997e72827e",
    ],
);
// DOWN_RAND_3X2: cv2.pyrDown(rand_img(3, 2, 33)) -> 2x1
const DOWN_RAND_3X2: (u32, u32, &[&str]) = (2, 1, &["6b5d"]);
// DOWN_RAND_1X7: cv2.pyrDown(rand_img(1, 7, 34)) -> 1x4
const DOWN_RAND_1X7: (u32, u32, &[&str]) = (1, 4, &["49605467"]);
// UP_RAND_13X9: cv2.pyrUp(rand_img(13, 9, 35)) -> 26x18
const UP_RAND_13X9: (u32, u32, &[&str]) = (
    26,
    18,
    &[
        "536992b1c3c4bfbdbfc5c1a58f99a180647a9eadaca09aacc1c85979acc4c2a995a5bfcbc6af9ba2a27e6380aab8ae90",
        "7e97b8c47094c9d7c18a678cc1cec3ad9da3a3837095c1c7af83667fa6b39da7b8c5c09b83a3c9c2a38373859c9d9eba",
        "d2c7ae9a8e8e9293c4b4a0a7b1a8a2b4bfa57a51446a9cb5c2d2d7c4b3bdc2a27d70c8bea995857c7c888d7857393974",
        "b1b9b0bccac9c7d4d09f6856bec3bd95664b434b544e423a4983b5ab939cb0bdc7d3cd9d6958b7c1c5ab7e4c2a2b3a3d",
        "42556b7b86909695908a92b3cbb6958aabb6c5c2a5642b2a3d424f7792775d7da39b7e65638ebabeb2ae8fa0bcc8b77a",
        "43475d5c6491b2967587a3a39281777f8a8b8887718cb4c5b7855c6b847b779ec2b59c9aa2a6a7a396775b5252526286",
        "b9bfa67b627f9f90809ab8b2a4a6aba1959187684b484f51628cc1b9926a5b7c9d8c778ba49e95a4af9a807b7a6c5d5b",
        "5e5f749bc8b6875f4d5e71615678988b798ba19a8d939e9f947a61587995b4a17a5b4a4d53494a76998264738f999da8",
        "b6c4be926353545f6e6b625f64727e7c7f9daf906d718592958b879baa927267282526354d6682a2b9bfbfc9c7a37d77",
        "808b866248608288817f19110e2345698db2cdd5d5d8cfa983797f888155334c75858687",
    ],
);
// UP_RAND_1X5: cv2.pyrUp(rand_img(1, 5, 36)) -> 2x10
const UP_RAND_1X5: (u32, u32, &[&str]) = (2, 10, &["535366667f7f82826b6b3c3c1b1b303054546060"]);
// UP_RAND_6X1: cv2.pyrUp(rand_img(6, 1, 37)) -> 12x2
const UP_RAND_6X1: (u32, u32, &[&str]) =
    (12, 2, &["5971866533180d0a132a41495971866533180d0a132a4149"]);
// UP_RAND_1X1: cv2.pyrUp(rand_img(1, 1, 38)) -> 2x2
const UP_RAND_1X1: (u32, u32, &[&str]) = (2, 2, &["40404040"]);
