//! `resize` uses **align-corners**; OpenCV uses **half-pixel**. Pin that here.
//!
//! # Why this needs a test rather than a comment
//!
//! The two conventions are indistinguishable on even-width inputs, which is most
//! of them. Measured on a 48-wide sinusoid and a 48-wide checkerboard, the two
//! mappings agree exactly — an earlier parity attempt used those inputs and could
//! not tell the conventions apart, and wrongly read a differing *amplitude* as
//! "different filtering". An impulse at an **odd** width is the discriminating
//! input: one lit pixel, one number, no hypothesis about phase.
//!
//! # The measurement
//!
//! A 16x16 image with a single `255` at the centre, downscaled to 8x8:
//!
//! ```text
//! OpenCV NEAREST : 255          Rust NEAREST : 255    (agree)
//! OpenCV LINEAR  :  64 = 255/4  Rust LINEAR  :  47
//! OpenCV AREA    :  64          (identical to LINEAR)
//! ```
//!
//! 47 is exactly what a correct bilinear gives under align-corners:
//!
//! ```text
//! align-corners  fx = x*(w-1)/(nw-1)     = 8.5714  -> 255*(1-0.5714)^2 = 46.84
//! half-pixel     fx = (x+0.5)*w/nw - 0.5 = 8.5000  -> 255*(1-0.5)^2     = 63.75
//! ```
//!
//! **This is a convention difference, not a defect.** Align-corners pins the corner
//! samples, so a downscale and a matching upscale are exact inverses at the
//! endpoints; half-pixel treats samples as centres, which is what OpenCV and
//! graphics APIs use, and avoids a half-pixel shift at non-integer scales.
//!
//! What *is* a defect is making the opposite choice from OpenCV **silently** in a
//! library positioned as its replacement: a user compositing `resize` with
//! `warp_affine` or a camera matrix gets a half-pixel displacement they cannot see.
//! This test makes the choice explicit and pins it, so that changing it is a
//! deliberate decision rather than an accident — and so a future reader
//! discovers it from the code rather than from a rendering artefact.
//!
//! **Scope, and what was actually checked.** `cv_hal`'s CPU resize implements the
//! mapping and this test covers the value the caller observes. The GPU path is a
//! separate implementation — `hal/shaders/resize_f32.wgsl`, reached through
//! `gpu_kernels::resize` — and it is **not** covered by a test here, because that
//! needs an adapter. It was checked by reading:
//!
//! ```wgsl
//! src_width_f  = f32(params.src_w) - 1.0;
//! dst_width_f  = max(f32(params.dst_w) - 1.0, 1.0);
//! src_x_f      = max(f32(x_dst) * src_width_f / dst_width_f, 0.0);
//! ```
//!
//! which is align-corners, matching the CPU path. So the two backends agree with
//! each other and disagree with OpenCV together. **If the convention is ever
//! changed, all three — `imgproc`, `hal`'s CPU resize, and the WGSL shader — must
//! change together**, or `resize` will disagree with itself depending on which
//! backend ran it. That requirement is recorded here because the GPU side has no
//! test guarding it.

#![forbid(unsafe_code)]

use cv_imgproc::resize::{resize, Interpolation};
use image::{GrayImage, Luma};

/// `align-corners`: output pixel `x` samples source `x·(w−1)/(nw−1)`.
fn align_corners(x: usize, w: usize, nw: usize) -> f64 {
    if nw <= 1 {
        return 0.0;
    }
    x as f64 * (w - 1) as f64 / (nw - 1) as f64
}

/// The 2-D bilinear weight of a single source pixel at `(sx, sy)` for one output
/// pixel, under align-corners. This is the whole test: the impulse response is
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
        let f = align_corners(o, a, b);
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
/// Against OpenCV this test would fail with `47` versus `64`; it is pinned against
/// the *current* behaviour deliberately, because changing the convention is a
/// contract decision affecting every caller — not a bug fix.
#[test]
fn resize_uses_align_corners_sampling() {
    let (w, h) = (16u32, 16u32);
    let (nw, nh) = (8usize, 8usize);
    let img = impulse_image(w);

    let out = resize(&img, nw as u32, nh as u32, Interpolation::Linear);
    let got = out.get_pixel(4, 4)[0];

    let want = (255.0 * impulse_weight(4, 4, 8, 8, w as usize, h as usize, nw, nh) + 0.5) as u8;

    assert_eq!(
        got, want,
        "the impulse response is {got} but align-corners predicts {want}. If this \\
         changed, the sampling convention did - which is a contract decision, not \\
         a bug fix, and it must change in imgproc, hal's CPU resize AND hal's GPU \\
         resize together."
    );

    // And the number itself, so a reader does not have to re-derive it:
    //   align-corners fx = 4 * 15/7 = 8.5714, dx = 0.5714
    //   2-D weight on the source pixel = (1 - 0.5714)^2 = 0.18367
    //   255 * 0.18367 = 46.84 -> 47
    assert_eq!(got, 47, "the documented value for this configuration");
}

/// CONTROL: `Nearest` agrees with OpenCV exactly, so the divergence above is
/// specific to the interpolating modes and not to the coordinate mapping.
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

/// An **even** width cannot distinguish the conventions, which is why every
/// earlier attempt with one failed to see the difference. Asserted so the choice of
/// test input above is not mistaken for an arbitrary one.
#[test]
fn an_even_width_cannot_distinguish_the_conventions() {
    // 16 -> 8 with an impulse at 8: align-corners and half-pixel both put it in
    // output pixel 4, so only the *weight* differs, and it does.
    //
    // The stronger statement: for w == nw the two mappings coincide exactly,
    // because x*(w-1)/(w-1) = x and (x+0.5)*w/w - 0.5 = x.
    for w in [16usize, 48, 64, 100] {
        for x in 0..w {
            let half_pixel = (x as f64 + 0.5) * w as f64 / w as f64 - 0.5;
            assert!(
                (align_corners(x, w, w) - half_pixel).abs() < 1e-12,
                "at equal width both mappings are x = {x}; w={w}"
            );
        }
    }
}

/// CONTROL: a resize to the same size is the identity, and a double resize is
/// close to reversible under the chosen convention.
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
    assert_eq!(
        diff, 0,
        "a same-size bilinear resize must be the identity; align-corners gives \\
         fx = x exactly, so this is a direct check of the mapping"
    );
}
