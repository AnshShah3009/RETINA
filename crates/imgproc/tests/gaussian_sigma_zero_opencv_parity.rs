//! A non-positive Gaussian sigma, as OpenCV treats it.
//!
//! * **Kernel size given** (`gaussian_kernel_1d(sigma, size)`,
//!   `gaussian_kernel(sigma, size)`): OpenCV's `getGaussianKernel(size,
//!   sigma <= 0)` derives the kernel from the size - the fixed binomial-like
//!   tables for sizes 1, 3, 5, 7 and 9, otherwise
//!   `sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8`. `cv2.GaussianBlur(img, (k, k),
//!   0)` blurs with that kernel. These used to divide by `sigma² = 0` and
//!   return a kernel of NaNs.
//! * **No kernel size** (`gaussian_blur(img, sigma)`, which sizes the kernel
//!   from sigma): `cv2.GaussianBlur(img, (0, 0), 0)` raises (`ksize.width > 0`
//!   assertion). With no error channel, `gaussian_blur` returns the image
//!   unchanged - what OpenCV gives for the 1x1 kernel - and now does so by an
//!   explicit check rather than by a NaN kernel happening to round back to
//!   `[1.0]`.
//!
//! Reference values: OpenCV 4.13, `cv2.getGaussianKernel(k, 0)` and
//! `cv2.GaussianBlur(img, (k, k), 0)` with `cv2.setNumThreads(1)`. (With
//! OpenCV's default thread pool, 8-bit `GaussianBlur` on this 9x7 image is
//! not deterministic: repeated calls return different bytes in row 1. The
//! single-threaded result equals the rounded f64 reference.)

use cv_imgproc::{
    BorderMode, gaussian_blur, gaussian_kernel, gaussian_kernel_1d, separable_convolve,
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
    GrayImage::from_raw(W as u32, H as u32, lcg(W * H, 3)).unwrap()
}

#[test]
fn zero_sigma_kernels_are_opencvs_tables() {
    let tables: [&[f32]; 5] = [
        &[1.0],
        &[0.25, 0.5, 0.25],
        &[0.0625, 0.25, 0.375, 0.25, 0.0625],
        &[
            0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
        ],
        &[
            0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875,
            0.05078125, 0.015625,
        ],
    ];
    for table in tables {
        for sigma in [0.0f32, -1.0, f32::NAN] {
            assert_eq!(
                gaussian_kernel_1d(sigma, table.len()),
                table,
                "size {} sigma {sigma}",
                table.len()
            );
        }
    }
}

#[test]
fn zero_sigma_derives_sigma_from_a_larger_size() {
    // The last tap absorbs the f32 normalisation residue (see
    // `gaussian_kernel_1d`), so it sits ~2e-7 off the f64 reference.
    let k = gaussian_kernel_1d(0.0, 11);
    for (i, (&got, &want)) in k.iter().zip(KERNEL_11.iter()).enumerate() {
        assert!(
            (got as f64 - want).abs() < 5e-7,
            "tap {i}: {got} vs cv2 {want}"
        );
    }
    let sum: f32 = k.iter().sum();
    assert_eq!(sum, 1.0);
}

#[test]
fn zero_sigma_2d_kernel_is_the_outer_product() {
    let k1 = gaussian_kernel_1d(0.0, 5);
    let k2 = gaussian_kernel(0.0, 5);
    assert_eq!((k2.width, k2.height), (5, 5));
    for y in 0..5 {
        for x in 0..5 {
            assert_eq!(k2.get(x, y), k1[x] * k1[y], "({x}, {y})");
        }
    }
    assert!(gaussian_kernel(0.0, 11).data.iter().all(|v| v.is_finite()));
}

#[test]
fn zero_sigma_with_a_size_blurs_like_opencv() {
    let img = input();
    for (k, want) in [(5usize, &BLUR_5), (7, &BLUR_7), (9, &BLUR_9)] {
        let kernel = gaussian_kernel_1d(0.0, k);
        let got = separable_convolve(&img, &kernel, BorderMode::Reflect101);
        assert_eq!(
            got.as_raw().as_slice(),
            want.as_flattened(),
            "cv2.GaussianBlur(img, ({k}, {k}), 0)"
        );
    }
}

#[test]
fn zero_sigma_without_a_size_returns_the_image() {
    let img = input();
    for sigma in [0.0f32, -2.0, f32::NAN] {
        assert_eq!(
            gaussian_blur(&img, sigma).as_raw(),
            img.as_raw(),
            "sigma {sigma}"
        );
    }
}

const BLUR_5: [[u8; W]; H] = [
    [115, 128, 152, 161, 163, 167, 150, 111, 90],
    [117, 128, 145, 148, 149, 153, 141, 114, 100],
    [134, 139, 138, 125, 120, 122, 118, 113, 113],
    [150, 151, 137, 111, 103, 106, 104, 109, 117],
    [136, 139, 128, 107, 110, 118, 111, 111, 117],
    [116, 119, 115, 110, 125, 134, 119, 110, 111],
    [111, 113, 111, 112, 132, 138, 120, 106, 104],
];
const BLUR_7: [[u8; W]; H] = [
    [127, 132, 142, 149, 151, 148, 135, 118, 110],
    [130, 133, 140, 143, 144, 141, 131, 117, 111],
    [136, 137, 135, 131, 127, 125, 120, 115, 112],
    [140, 138, 131, 121, 116, 114, 113, 112, 112],
    [133, 131, 124, 117, 115, 116, 114, 112, 112],
    [122, 121, 118, 116, 120, 122, 118, 113, 111],
    [118, 118, 116, 117, 123, 125, 119, 112, 109],
];
const BLUR_9: [[u8; W]; H] = [
    [132, 135, 140, 143, 143, 140, 131, 120, 116],
    [133, 135, 138, 139, 139, 135, 128, 119, 116],
    [135, 135, 134, 132, 129, 126, 121, 116, 114],
    [135, 134, 130, 124, 121, 119, 116, 114, 113],
    [130, 129, 125, 120, 118, 117, 115, 113, 113],
    [124, 123, 121, 119, 119, 120, 116, 113, 113],
    [120, 120, 119, 118, 120, 121, 117, 114, 113],
];
const KERNEL_11: [f64; 11] = [
    0.008812229292562288,
    0.027143577143479373,
    0.06511405659938269,
    0.12164907301380962,
    0.1769983568313557,
    0.20056541423882085,
    0.1769983568313557,
    0.12164907301380962,
    0.06511405659938269,
    0.027143577143479373,
    0.008812229292562288,
];
