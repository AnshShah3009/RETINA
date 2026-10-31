//! `separable_convolve` must compute the same thing as the 2-D
//! `convolve_with_border` with the outer product of the same 1-D kernel - the
//! separable form is an optimisation of the 2-D convolution, not a different
//! filter.
//!
//! Regression: for `BorderMode::Constant(v)` the vertical pass substituted the
//! raw constant `v` for an out-of-range source *row*. A row outside the image
//! is horizontally convolved too, so its value is `v * Σkx`; substituting `v`
//! scaled the border contribution by `1 / Σkx`. Kernels that sum to 1 (a
//! normalised Gaussian) agreed, which is why this survived: `[1, 1, 1]` (Σ=3),
//! `[1, 2, 1]` (Σ=4) and the derivative `[-1, 0, 1]` (Σ=0) did not.

use cv_imgproc::convolve::{gaussian_kernel_1d, BorderMode, Kernel};
use image::{GrayImage, Luma};

/// Low-dynamic-range pseudo-random image so that the `u8` clamp cannot mask a
/// difference between the two paths.
fn low_range_image(width: u32, height: u32, seed: u32) -> GrayImage {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state & 3) as u8
    };
    GrayImage::from_fn(width, height, |_, _| Luma([next()]))
}

/// The outer product `kx ⊗ ky` as a square `Kernel`, laid out exactly like
/// [`Kernel::get`] expects (`data[ky * width + kx] != 0`).
fn outer_product(kx: &[f32], ky: &[f32]) -> Kernel {
    assert_eq!(kx.len(), ky.len());
    let mut data = Vec::with_capacity(kx.len() * ky.len());
    for &y in ky {
        for &x in kx {
            data.push(x * y);
        }
    }
    Kernel::new(data, kx.len(), ky.len())
}

const BORDERS: &[BorderMode] = &[
    BorderMode::Constant(0),
    BorderMode::Constant(1),
    BorderMode::Constant(3),
    BorderMode::Constant(200),
    BorderMode::Replicate,
    BorderMode::Reflect,
    BorderMode::Reflect101,
    BorderMode::Wrap,
];

#[test]
fn hand_computable_constant_border_row() {
    // A 2x1 all-zero image with `[1, 2, 1]` (Σ = 4) and BORDER_CONSTANT(10).
    // All nine taps of the 2-D kernel are out of range except the two with
    // ky == 0, so the 2-D result is
    //   10 * (Σkx·Σky − kx[0]·ky[0]·... ) = 10 * (16 − 6) = 100.
    // Substituting the raw constant for the out-of-range rows gives
    // 10 * Σky + (horizontal result) = 40, so the two paths disagree by 60.
    let img = GrayImage::new(2, 1);
    let k = vec![1.0f32, 2.0, 1.0];
    let two_d =
        cv_imgproc::convolve_with_border(&img, &outer_product(&k, &k), BorderMode::Constant(10));
    let sep = cv_imgproc::separable_convolve(&img, &k, BorderMode::Constant(10));

    assert_eq!(two_d.as_raw(), &[100, 100], "2-D reference (hand computed)");
    assert_eq!(sep.as_raw(), two_d.as_raw(), "separable path disagrees");
}

#[test]
fn separable_matches_full_2d_convolution() {
    // Kernels whose sum is not 1 are the ones that exposed the defect; the
    // normalised Gaussian is the control that always worked.
    let kernels: &[(&str, Vec<f32>)] = &[
        ("box3", vec![1.0, 1.0, 1.0]),
        ("binomial3", vec![1.0, 2.0, 1.0]),
        ("derivative", vec![-1.0, 0.0, 1.0]),
        ("unnormalised5", vec![1.0, 4.0, 6.0, 4.0, 1.0]),
        ("half", vec![0.25, 0.5, 0.25]),
        ("gauss3", gaussian_kernel_1d(1.0, 3)),
    ];

    for width in 1..40u32 {
        for height in [1u32, 2, 3, 5, 8, 9, 16] {
            let img = low_range_image(width, height, width * 7 + height);
            for (name, k) in kernels {
                let kernel = outer_product(k, k);
                for &border in BORDERS {
                    let two_d = cv_imgproc::convolve_with_border(&img, &kernel, border);
                    let sep = cv_imgproc::separable_convolve(&img, k, border);

                    let max_diff = sep
                        .as_raw()
                        .iter()
                        .zip(two_d.as_raw())
                        .map(|(a, b)| (*a as i32 - *b as i32).abs())
                        .max()
                        .unwrap();

                    // A kernel that sums to exactly 1 (and the Gaussian, whose
                    // sum is 1 up to f32 rounding) can differ by the last bit
                    // of the u8 truncation; everything else must match exactly.
                    let tolerance = if name.starts_with("gauss") { 1 } else { 0 };
                    assert!(
                        max_diff <= tolerance,
                        "{name} @ {width}x{height} {border:?}: separable differs from the \
                         2-D convolution by {max_diff} (kernel sum = {})",
                        k.iter().sum::<f32>()
                    );
                }
            }
        }
    }
}

#[test]
fn sobel_ex_matches_the_2d_reference_for_a_constant_border() {
    // End-to-end: `sobel_ex` is documented as a Sobel derivative with a
    // selectable border, so it must equal the same linear transform applied to
    // the 2-D convolution with the Sobel kernel. It is routed through
    // `separable_convolve`, which used to under-weight the constant border for
    // exactly this kernel (Σ = 0 for the derivative pair).
    let deriv: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let smooth: Vec<f32> = vec![1.0, 2.0, 1.0];
    let (scale, delta) = (2.0f32, 5.0f32);

    for width in 1..24u32 {
        for height in [1u32, 2, 3, 5, 9] {
            let img = low_range_image(width, height, width * 31 + height);
            for border in [BorderMode::Constant(0), BorderMode::Constant(77)] {
                let two_d =
                    cv_imgproc::convolve_with_border(&img, &outer_product(&deriv, &smooth), border);
                let expected: Vec<u8> = two_d
                    .as_raw()
                    .iter()
                    .map(|&v| (v as f32 * scale + delta).clamp(0.0, 255.0) as u8)
                    .collect();

                let got = cv_imgproc::sobel_ex(&img, 1, 0, 3, scale, delta, border).unwrap();
                assert_eq!(
                    got.as_raw(),
                    &expected,
                    "sobel_ex {width}x{height} {border:?} does not match the 2-D reference"
                );
            }
        }
    }
}
