//! `resize` uses **half-pixel** (`(x+0.5)·w/nw − 0.5`), matching OpenCV. Pin
//! that here.
//!
//! # Why this needs a test rather than a comment
//!
//! The two coordinate conventions - this one and align-corners
//! (`x·(w−1)/(nw−1)`) - are indistinguishable on same-size and even-width
//! inputs, which is most of them. An impulse at an **odd** width is the
//! discriminating input: one lit pixel, one number, no hypothesis about phase.
//!
//! # The measurement
//!
//! A 16x16 image with a single `255` at the centre, downscaled to 8x8:
//!
//! ```text
//! align-corners  fx = x*(w-1)/(nw-1)     = 8.5714  -> 255*(1-0.5714)^2 = 46.84
//! half-pixel     fx = (x+0.5)*w/nw - 0.5 = 8.5000  -> 255*(1-0.5)^2     = 63.75
//!
//! OpenCV LINEAR  :  64 = 255/4   (half-pixel, rounded)
//! ```
//!
//! `resize` used to return 47 - align-corners - while every OpenCV-based caller
//! expects 64. **The convention was switched to half-pixel deliberately** (an
//! explicit contract decision, recorded in `docs/PARITY.md`): the library is
//! positioned as an OpenCV replacement, and silently choosing the opposite
//! convention hands a user compositing `resize` with `warp_affine` or a camera
//! matrix a half-pixel displacement they cannot see. Align-corners remains
//! defensible in the abstract - it pins the corner samples, so a downscale and a
//! matching upscale are exact inverses at the endpoints - but interchangeability
//! with `cv2.resize` is worth more here.
//!
//! **Scope, and what was actually checked.** `cv_hal`'s CPU resize implements
//! the mapping and this test covers the value the caller observes. The GPU path
//! is a separate implementation - `hal/shaders/resize_f32.wgsl` and its f16,
//! bf16 and multichannel variants, reached through `gpu_kernels::resize` - and
//! it is **not** covered by a test here, because that needs an adapter. It was
//! checked by reading:
//!
//! ```wgsl
//! src_width_f  = f32(params.src_w);
//! dst_width_f  = max(f32(params.dst_w), 1.0);
//! src_x_f      = max((f32(x_dst) + 0.5) * src_width_f / dst_width_f - 0.5, 0.0);
//! ```
//!
//! which is half-pixel, matching the CPU path. So the two backends agree with
//! each other and with OpenCV together. **If the convention is ever changed,
//! all of `imgproc`, `hal`'s CPU resize, and the WGSL shaders must change
//! together**, or `resize` will disagree with itself depending on which backend
//! ran it. That requirement is recorded here because the GPU side has no test
//! guarding it.

#![forbid(unsafe_code)]

use cv_imgproc::resize::{resize, Interpolation};
use image::{GrayImage, Luma};

/// `half-pixel`: output pixel `x` samples source `(x + 0.5)·w/nw − 0.5`.
fn half_pixel(x: usize, w: usize, nw: usize) -> f64 {
    (x as f64 + 0.5) * w as f64 / nw as f64 - 0.5
}

/// The 2-D bilinear weight of a single source pixel at `(sx, sy)` for one output
/// pixel, under half-pixel. This is the whole test: the impulse response is
/// `255 · weight`, and comparing that number distinguishes the conventions.
fn impulse_weight(
    ox: usize,
    oy: usize,
    sx: usize,
    sy: usize,
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
) -> f64 {
    let axis = |o: usize, s: usize, a: usize, b: usize| -> f64 {
        // Positions below 0 clamp to the edge sample, as the resamplers do.
        let f = half_pixel(o, a, b).max(0.0);
        let x0 = (f as usize).min(a.saturating_sub(1));
        let x1 = (x0 + 1).min(a.saturating_sub(1));
        let d = if x0 == x1 {
            0.0
        } else {
            (f - x0 as f64).clamp(0.0, 1.0)
        };
        if s == x0 {
            1.0 - d
        } else if s == x1 {
            d
        } else {
            0.0
        }
    };
    axis(ox, sx, w, nw) * axis(oy, sy, h, nh)
}

fn impulse_image(s: u32) -> GrayImage {
    let mut img = GrayImage::new(s, s);
    img.put_pixel(s / 2, s / 2, Luma([255]));
    img
}

/// The convention, stated as an assertion.
///
/// The impulse response is 64 - `255/4`, exactly what OpenCV's `INTER_LINEAR`
/// produces on this input. Under align-corners it is 47. If this number moves,
/// the sampling convention changed again - which is a contract decision, not a
/// bug fix, and it must change in imgproc, hal's CPU resize AND hal's GPU
/// resize together.
#[test]
fn resize_uses_half_pixel_sampling() {
    let (w, h) = (16u32, 16u32);
    let (nw, nh) = (8usize, 8usize);
    let img = impulse_image(w);

    let out = resize(&img, nw as u32, nh as u32, Interpolation::Linear);
    let got = out.get_pixel(4, 4)[0];

    let want = (255.0 * impulse_weight(4, 4, 8, 8, w as usize, h as usize, nw, nh) + 0.5) as u8;

    assert_eq!(
        got, want,
        "the impulse response is {got} but half-pixel predicts {want}. If this \\
         changed, the sampling convention did - which is a contract decision, not \\
         a bug fix, and it must change in imgproc, hal's CPU resize AND hal's GPU \\
         resize together."
    );

    // And the number itself, so a reader does not have to re-derive it:
    //   half-pixel fx = (4 + 0.5) * 2 - 0.5 = 8.5, dx = 0.5
    //   2-D weight on the source pixel = (1 - 0.5)^2 = 0.25
    //   255 * 0.25 = 63.75 -> 64   (OpenCV's value)
    assert_eq!(got, 64, "the documented value for this configuration");
}

/// CONTROL: `Nearest` agrees with OpenCV exactly - it floors `x / (nw/w)`, the
/// scale as an f32 reciprocal (cv2's arithmetic, measured over 293 samples of
/// 12 (w, nw) pairs; every multiply form fails one tie case and half-pixel
/// fails 101). Nearest is not half-pixel, and no smooth-kernel convention
/// affects it.
#[test]
fn nearest_neighbour_agrees_with_opencv() {
    let img = impulse_image(16);
    let out = resize(&img, 8, 8, Interpolation::Nearest);
    let lit: Vec<(u32, u32, u8)> = (0..8)
        .flat_map(|y| (0..8).map(move |x| (y, x)))
        .filter(|&(y, x)| out.get_pixel(x, y)[0] != 0)
        .map(|(y, x)| (x, y, out.get_pixel(x, y)[0]))
        .collect();
    assert_eq!(
        lit,
        vec![(4, 4, 255)],
        "OpenCV INTER_NEAREST on this input gives [(4, 4, 255)]"
    );
}

/// At `w == nw` the two conventions coincide exactly (`x*(w-1)/(w-1) = x` and
/// `(x+0.5)*w/w - 0.5 = x`), which is why every earlier parity attempt that
/// used same-size or well-chosen even widths failed to see the difference.
/// Asserted so the choice of test input above is not mistaken for arbitrary.
#[test]
fn same_size_sampling_coincides_with_align_corners() {
    fn align_corners(x: usize, w: usize) -> f64 {
        x as f64 * (w - 1) as f64 / (w - 1) as f64
    }
    for w in [16usize, 48, 64, 100] {
        for x in 0..w {
            assert!(
                (align_corners(x, w) - half_pixel(x, w, w)).abs() < 1e-12,
                "at equal width both mappings are x = {x}; w={w}"
            );
        }
    }
}

/// CONTROL: a resize to the same size is the identity under half-pixel too,
/// because `(x+0.5)*w/w - 0.5 = x` exactly. A direct check of the mapping.
#[test]
fn resizing_to_the_same_size_is_the_identity() {
    let mut src = GrayImage::new(16, 16);
    for y in 0..16u32 {
        for x in 0..16u32 {
            src.put_pixel(x, y, Luma([((x * 16 + y * 3) % 256) as u8]));
        }
    }
    let out = resize(&src, 16, 16, Interpolation::Linear);
    let diff: i32 = (0..16u32)
        .flat_map(|y| (0..16u32).map(move |x| (x, y)))
        .map(|(x, y)| (out.get_pixel(x, y)[0] as i32 - src.get_pixel(x, y)[0] as i32).abs())
        .sum();
    assert_eq!(diff, 0, "a same-size bilinear resize must be the identity");
}
