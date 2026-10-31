//! Numerical-parity number generator: filters (`cv-imgproc`) vs OpenCV.
//!
//! Prints one value per line, no assertions. The companion Python reference
//! side lives at `parity/parity_filters.py`.
//!
//! Output protocol (one record per line, `#`-prefixed so a line is never a
//! bare number and both sides agree on framing):
//!
//! ```text
//! #IM <name> <width> <height> <v0> <v1> ...   # image, row-major, values as f64
//! #V  <name> <value>                          # scalar, {:.17e}
//! #M  <name> <kernel_radius>                  # case metadata: border width to exclude
//! ```
//!
//! The Python side re-derives every input from the closed-form definitions
//! below and asserts byte-identity against the `#IM` records, so a drifting
//! input generator fails loudly instead of silently producing a meaningless
//! comparison.
//!
//! No randomness anywhere: every input is a closed-form function of (x, y).

use cv_imgproc::{
    create_morph_kernel, dilate_with_border, erode_with_border, gaussian_blur_with_border,
    gaussian_kernel_1d, sobel_ex, BorderMode, MorphShape,
};
use image::{GrayImage, Luma};

const W: u32 = 64;
const H: u32 = 48;
const TAU: f64 = std::f64::consts::PI * 2.0;

// ── deterministic inputs (shared definition with the Python side) ───────────

fn input_constant() -> GrayImage {
    GrayImage::from_pixel(W, H, Luma([128]))
}

fn input_impulse() -> GrayImage {
    let mut img = GrayImage::from_pixel(W, H, Luma([0]));
    img.put_pixel(W / 2, H / 2, Luma([255]));
    img
}

/// Monotone ramp with unit slope in both axes: `f(x,y) = x + y`, max 110 so
/// nothing saturates and both derivatives are strictly non-negative (which
/// makes the u8 Sobel output comparable against an OpenCV CV_64F reference).
fn input_ramp() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            img.put_pixel(x, y, Luma([(x + y) as u8]));
        }
    }
    img
}

/// `127 + 100*sin(2*pi*(3x/W + 2y/H))`, rounded to nearest u8.
fn input_sinusoid() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let theta = TAU * (3.0 * x as f64 / W as f64 + 2.0 * y as f64 / H as f64);
            let v = (127.0 + 100.0 * theta.sin()).round();
            img.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
        }
    }
    img
}

fn inputs() -> Vec<(&'static str, GrayImage)> {
    vec![
        ("const", input_constant()),
        ("impulse", input_impulse()),
        ("ramp", input_ramp()),
        ("sinusoid", input_sinusoid()),
    ]
}

// ── emitters ────────────────────────────────────────────────────────────────

fn emit_image(name: &str, img: &GrayImage) {
    let mut line = format!("#IM {} {} {}", name, img.width(), img.height());
    for p in img.as_raw() {
        line.push_str(&format!(" {:.1}", *p as f64));
    }
    println!("{line}");
}

/// Record the half-width of the border band a case's border policy can affect.
/// The Python harness excludes exactly this band when reporting the "interior"
/// deviation, and reports the full-field deviation alongside it.
fn emit_meta(name: &str, radius: usize) {
    println!("#M {} {}", name, radius);
}

/// Emit the float32 convolution sum *before* the round-to-u8 step, for one
/// row of one case. A normalised blur of a constant must give exactly that
/// constant in f32; if the pre-rounding value is off by more than a few ulps
/// the kernel normalisation is wrong, and if it is correct then a u8
/// discrepancy is purely the `as u8` truncation below.
fn emit_f32_row(name: &str, values: &[f32]) {
    let mut line = format!("#FR {} {}", name, values.len());
    for v in values {
        line.push_str(&format!(" {:.17e}", v));
    }
    println!("{line}");
}

fn main() {
    let all = inputs();

    // Inputs, echoed so the reference side can verify input identity.
    for (name, img) in &all {
        emit_image(&format!("input_{name}"), img);
    }

    // Structuring elements, as masks. OpenCV's `getStructuringElement`
    // produces the canonical shapes; this is where an implementation that
    // derives the ellipse analytically can diverge.
    for (tag, shape) in [
        ("rect", MorphShape::Rectangle),
        ("ellipse", MorphShape::Ellipse),
    ] {
        let ks = 5i32;
        let pts = create_morph_kernel(shape, ks as u32, ks as u32);
        let mut mask = GrayImage::new(ks as u32, ks as u32);
        for &(dx, dy) in &pts {
            mask.put_pixel((dx + ks / 2) as u32, (dy + ks / 2) as u32, Luma([255]));
        }
        emit_image(&format!("kernel_{tag}_5x5"), &mask);
    }

    // ── gaussian blur ──────────────────────────────────────────────────────
    // OpenCV's `GaussianBlur(ksize=(0,0), sigma)` derives
    //   ksize = 2*cvRound(3*sigma) + 1,
    // while this repo's `gaussian_blur_ctx` uses `(ceil(6*sigma)) | 1`.
    // Those agree for every sigma where 6*sigma is an exact integer (and also
    // at sigma=1.5 by rounding), which is why only such sigmas are compared:
    // otherwise the kernels have different supports and the comparison would
    // measure the window, not the filter.
    let sigmas = [0.75f32, 1.5, 2.0];
    let borders = [
        ("reflect101", BorderMode::Reflect101),
        ("replicate", BorderMode::Replicate),
    ];

    for (name, img) in &all {
        for &sigma in &sigmas {
            for (bname, border) in borders {
                let radius = (((sigma * 6.0).ceil() as usize) | 1) / 2;
                let case = format!("gauss_{name}_s{sigma:?}_{bname}");
                emit_meta(&case, radius);
                let out = gaussian_blur_with_border(img, sigma, border);
                emit_image(&case, &out);
            }
        }
    }

    // Kernel probe: the exact 1-D kernel coefficients the library builds, so
    // the Python side can recompute the convolution in f64 and attribute any
    // u8 discrepancy between the two sides to either a different kernel or
    // only to the round-to-nearest vs truncate-on-cast quantisation step.
    for &sigma in &sigmas {
        let size = (((sigma * 6.0).ceil() as usize) | 1);
        let k = gaussian_kernel_1d(sigma, size);
        emit_f32_row(&format!("kernel1d_s{sigma:?}"), &k);
        println!("#FK s{sigma:?} {:.17e} {}", k.iter().sum::<f32>(), k.len());
    }

    // ── erode / dilate ─────────────────────────────────────────────────────
    // Rust's `dilate`/`erode` wrappers use `BorderMode::Replicate`; OpenCV's
    // morphology default border value is +DBL_MAX for dilate and -DBL_MAX for
    // erode, which is boundary *replication* in effect. Comparable.
    let rect5 = create_morph_kernel(MorphShape::Rectangle, 5, 5);
    let ellipse5 = create_morph_kernel(MorphShape::Ellipse, 5, 5);
    for (name, img) in &all {
        for (ktag, kernel) in [("rect5", &rect5), ("ellipse5", &ellipse5)] {
            let d = dilate_with_border(img, kernel, 1, BorderMode::Replicate);
            emit_meta(&format!("dilate_{name}_{ktag}"), 2);
            emit_image(&format!("dilate_{name}_{ktag}"), &d);

            let e = erode_with_border(img, kernel, 1, BorderMode::Replicate);
            emit_meta(&format!("erode_{name}_{ktag}"), 2);
            emit_image(&format!("erode_{name}_{ktag}"), &e);
        }
    }

    // ── Sobel ──────────────────────────────────────────────────────────────
    // The Rust result is a `GrayImage` (u8, clamped to 0..255); OpenCV's
    // CV_64F result is signed and unquantised. The Python side therefore also
    // reports `round()` of its own CV_64F output, which is the only
    // like-for-like quantity, plus the maximum magnitude of the reference
    // (which bounds the cost of the sign loss).
    for (name, img) in &all {
        for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
            for ksize in [3usize, 5] {
                let case = format!("sobel_{name}_dx{dx}_dy{dy}_k{ksize}");
                emit_meta(&case, ksize / 2);
                let out = sobel_ex(img, dx, dy, ksize, 1.0, 0.0, BorderMode::Reflect101)
                    .expect("sobel_ex");
                emit_image(&case, &out);
            }
        }
    }
}
