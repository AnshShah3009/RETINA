//! Zero-width / zero-height images must not panic.
//!
//! Row-parallel loops split buffers with `par_chunks_mut(width)`, which
//! panics with "chunk size must not be zero" when `width == 0` (and several
//! kernels index `height - 1`). Every public entry point below used to panic
//! on at least one of 0x0, 0xN or Nx0; each must now return an image of the
//! documented output size (empty when the input is empty).

use cv_imgproc::*;
use image::{GrayImage, RgbImage};
use std::panic::{AssertUnwindSafe, catch_unwind};

const SHAPES: [(u32, u32); 3] = [(0, 0), (0, 7), (7, 0)];

/// Run every case, collect the ones that panic or return the wrong size.
fn run(
    cases: Vec<(&str, Box<dyn Fn(u32, u32) -> (u32, u32)>)>,
    expect: fn(u32, u32) -> (u32, u32),
) {
    let mut bad = Vec::new();
    for (name, f) in &cases {
        for (w, h) in SHAPES {
            match catch_unwind(AssertUnwindSafe(|| f(w, h))) {
                Ok(dims) if dims == expect(w, h) => {}
                Ok(dims) => bad.push(format!("{name} {w}x{h}: got {dims:?}")),
                Err(_) => bad.push(format!("{name} {w}x{h}: panicked")),
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn gray(w: u32, h: u32) -> GrayImage {
    GrayImage::new(w, h)
}

fn same(w: u32, h: u32) -> (u32, u32) {
    (w, h)
}

#[test]
fn threshold_and_histogram() {
    run(
        vec![
            (
                "threshold",
                Box::new(|w, h| threshold(&gray(w, h), 9, 255, ThresholdType::Binary).dimensions()),
            ),
            (
                "threshold_otsu",
                Box::new(|w, h| {
                    threshold_otsu(&gray(w, h), 255, ThresholdType::Binary)
                        .1
                        .dimensions()
                }),
            ),
            (
                "adaptive_threshold mean",
                Box::new(|w, h| {
                    adaptive_threshold(
                        &gray(w, h),
                        255,
                        AdaptiveMethod::MeanC,
                        ThresholdType::Binary,
                        3,
                        0.0,
                    )
                    .dimensions()
                }),
            ),
            (
                "adaptive_threshold gaussian",
                Box::new(|w, h| {
                    adaptive_threshold(
                        &gray(w, h),
                        255,
                        AdaptiveMethod::GaussianC,
                        ThresholdType::Binary,
                        3,
                        0.0,
                    )
                    .dimensions()
                }),
            ),
            (
                "local_threshold",
                Box::new(|w, h| {
                    local_threshold(
                        &gray(w, h),
                        255,
                        LocalThresholdMethod::Sauvola,
                        ThresholdType::Binary,
                        3,
                        0.2,
                        128.0,
                    )
                    .unwrap()
                    .dimensions()
                }),
            ),
            (
                "histogram_equalization",
                Box::new(|w, h| histogram_equalization(&gray(w, h)).dimensions()),
            ),
            (
                "clahe",
                Box::new(|w, h| clahe(&gray(w, h), 2.0, (8, 8)).dimensions()),
            ),
        ],
        same,
    );
}

#[test]
fn morphology() {
    let k = create_morph_kernel(MorphShape::Rectangle, 3, 3);
    let k1 = k.clone();
    let k2 = k.clone();
    let k3 = k.clone();
    let k4 = k.clone();
    run(
        vec![
            (
                "dilate",
                Box::new(move |w, h| dilate(&gray(w, h), &k1, 1).dimensions()),
            ),
            (
                "erode",
                Box::new(move |w, h| erode(&gray(w, h), &k2, 1).dimensions()),
            ),
            (
                "opening",
                Box::new(move |w, h| opening(&gray(w, h), &k3, 1).dimensions()),
            ),
            (
                "erode_with_border",
                Box::new(move |w, h| {
                    erode_with_border(&gray(w, h), &k4, 1, BorderMode::Replicate).dimensions()
                }),
            ),
            (
                "morphological_gradient",
                Box::new(|w, h| morphological_gradient(&gray(w, h), 3).dimensions()),
            ),
            (
                "top_hat",
                Box::new(|w, h| top_hat(&gray(w, h), 3).dimensions()),
            ),
            (
                "black_hat",
                Box::new(|w, h| black_hat(&gray(w, h), 3).dimensions()),
            ),
            (
                "skeleton",
                Box::new(|w, h| skeleton(&gray(w, h)).dimensions()),
            ),
            (
                "hit_or_miss",
                Box::new(|w, h| {
                    hit_or_miss(&gray(w, h), &[0, 1, 0, 1, 1, 1, 0, 1, 0], 3, 3).dimensions()
                }),
            ),
        ],
        same,
    );
}

#[test]
fn convolution() {
    run(
        vec![
            (
                "gaussian_blur",
                Box::new(|w, h| gaussian_blur(&gray(w, h), 1.0).dimensions()),
            ),
            (
                "box_blur",
                Box::new(|w, h| box_blur(&gray(w, h), 3).dimensions()),
            ),
            (
                "sharpen",
                Box::new(|w, h| sharpen(&gray(w, h), 1.0).dimensions()),
            ),
            (
                "convolve",
                Box::new(|w, h| convolve(&gray(w, h), &box_kernel(3)).dimensions()),
            ),
            (
                "separable_convolve",
                Box::new(|w, h| {
                    separable_convolve(&gray(w, h), &[0.25, 0.5, 0.25], BorderMode::Reflect101)
                        .dimensions()
                }),
            ),
        ],
        same,
    );
}

#[test]
fn edges() {
    run(
        vec![
            ("sobel", Box::new(|w, h| sobel(&gray(w, h)).0.dimensions())),
            (
                "scharr",
                Box::new(|w, h| scharr(&gray(w, h)).1.dimensions()),
            ),
            (
                "sobel_ex",
                Box::new(|w, h| {
                    sobel_ex(&gray(w, h), 1, 0, 3, 1.0, 0.0, BorderMode::Reflect101)
                        .unwrap()
                        .dimensions()
                }),
            ),
            (
                "sobel_magnitude",
                Box::new(|w, h| sobel_magnitude(&gray(w, h), &gray(w, h)).dimensions()),
            ),
            (
                "laplacian",
                Box::new(|w, h| laplacian(&gray(w, h)).dimensions()),
            ),
            (
                "canny",
                Box::new(|w, h| canny(&gray(w, h), 50, 150).dimensions()),
            ),
        ],
        same,
    );
}

#[test]
fn geometry_and_resize() {
    let ident = nalgebra::Matrix3::<f32>::identity();
    let aff = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    // Empty source warped/resized into an empty destination.
    run(
        vec![
            (
                "warp_affine",
                Box::new(move |w, h| warp_affine(&gray(w, h), aff, w, h).dimensions()),
            ),
            (
                "warp_perspective",
                Box::new(move |w, h| warp_perspective(&gray(w, h), &ident, w, h).dimensions()),
            ),
            (
                "remap",
                Box::new(|w, h| {
                    let n = (w * h) as usize;
                    remap(
                        &gray(w, h),
                        &vec![0.0; n],
                        &vec![0.0; n],
                        w,
                        h,
                        Interpolation::Linear,
                        BorderMode::Replicate,
                    )
                    .dimensions()
                }),
            ),
            (
                "remap_identity",
                Box::new(|w, h| {
                    remap_identity(&gray(w, h), Interpolation::Linear, BorderMode::Replicate)
                        .dimensions()
                }),
            ),
            (
                "resize",
                Box::new(|w, h| resize(&gray(w, h), w, h, Interpolation::Linear).dimensions()),
            ),
            (
                "resize_rgb",
                Box::new(|w, h| {
                    resize_rgb(&RgbImage::new(w, h), w, h, Interpolation::Linear).dimensions()
                }),
            ),
        ],
        same,
    );
    // Non-empty source, empty destination.
    run(
        vec![
            (
                "warp_affine to empty",
                Box::new(move |w, h| warp_affine(&gray(5, 5), aff, w, h).dimensions()),
            ),
            (
                "warp_perspective to empty",
                Box::new(move |w, h| warp_perspective(&gray(5, 5), &ident, w, h).dimensions()),
            ),
            (
                "resize to empty",
                Box::new(|w, h| resize(&gray(5, 5), w, h, Interpolation::Linear).dimensions()),
            ),
            (
                "resize_rgb to empty",
                Box::new(|w, h| {
                    resize_rgb(&RgbImage::new(5, 5), w, h, Interpolation::Cubic).dimensions()
                }),
            ),
        ],
        same,
    );
    // Empty source into a non-empty destination: zeros, no panic.
    run(
        vec![
            (
                "warp_affine from empty",
                Box::new(move |w, h| warp_affine(&gray(w, h), aff, 4, 3).dimensions()),
            ),
            (
                "resize from empty",
                Box::new(|w, h| resize(&gray(w, h), 4, 3, Interpolation::Linear).dimensions()),
            ),
        ],
        |_, _| (4, 3),
    );
}

#[test]
fn bilateral_and_template() {
    run(
        vec![
            (
                "bilateral_filter_depth",
                Box::new(|w, h| {
                    let out = bilateral_filter_depth(&[], w, h, BilateralFilterParams::default());
                    assert!(out.is_empty());
                    (w, h)
                }),
            ),
            (
                "bilateral_filter_rgb",
                Box::new(|w, h| {
                    let out = bilateral_filter_rgb(&[], w, h, BilateralFilterParams::default());
                    assert!(out.is_empty());
                    (w, h)
                }),
            ),
            (
                "joint_bilateral_filter",
                Box::new(|w, h| {
                    let out =
                        joint_bilateral_filter(&[], &[], w, h, BilateralFilterParams::default());
                    assert!(out.is_empty());
                    (w, h)
                }),
            ),
            (
                "match_template",
                Box::new(|w, h| {
                    let r = match_template(
                        &gray(w, h),
                        &gray(1, 1),
                        cv_hal::context::TemplateMatchMethod::Ccorr,
                    );
                    assert!(r.data.is_empty());
                    (w, h)
                }),
            ),
        ],
        same,
    );
}
