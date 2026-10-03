//! Numerical-parity number generator: edge-preserving tone mapping / denoise
//! (`cv-photo` bilateral, `cv-imgproc` CLAHE + histogram equalisation)
//! vs OpenCV. Companion reference side: `parity/parity_photo.py`.
//!
//! Protocol:
//! ```text
//! #IM <name> <width> <height> <v0> ...   # image, row-major
//! #B  <name> <d> <sigma_color> <sigma_space>
//! #C  <name> <clip_limit> <tiles_x> <tiles_y>
//! #M  <name> <border_exclusion_radius>
//! ```

use cv_imgproc::{clahe, histogram_equalization};
use cv_photo::bilateral_filter;
use image::{GrayImage, Luma};

const W: u32 = 64;
const H: u32 = 64;
const TAU: f64 = std::f64::consts::PI * 2.0;

fn emit_image(name: &str, img: &GrayImage) {
    let mut line = format!("#IM {} {} {}", name, img.width(), img.height());
    for p in img.as_raw() {
        line.push_str(&format!(" {:.1}", *p as f64));
    }
    println!("{line}");
}

fn emit_bilateral(name: &str, d: i32, sc: f32, ss: f32) {
    println!("#B {} {} {:.17e} {:.17e}", name, d, sc, ss);
}

fn emit_clahe(name: &str, clip: f32, tx: u32, ty: u32) {
    println!("#C {} {:.17e} {} {}", name, clip, tx, ty);
}

fn emit_meta(name: &str, radius: usize) {
    println!("#M {} {}", name, radius);
}

/// Low-contrast scene: a smooth luminance ramp confined to ~60 grey levels.
/// This is the case CLAHE exists for - a global histogram equalisation is
/// nearly linear across such a narrow range and CLAHE is not.
fn input_low_contrast() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = (100.0 + 60.0 * x as f64 / (W as f64 - 1.0)).round();
            img.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
        }
    }
    img
}

/// Bimodal scene: a bright left region and a dark right region with a soft
/// transition, so per-tile statistics genuinely differ across the grid.
fn input_bimodal() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let fx = x as f64 / (W as f64 - 1.0);
            let base = if fx < 0.5 { 30.0 } else { 200.0 };
            let wobble = 20.0 * (TAU * (2.0 * y as f64 / H as f64)).sin();
            let v = (base + wobble).round();
            img.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
        }
    }
    img
}

/// Smooth sinusoid plus a *deterministic* high-frequency pattern; used for
/// bilateral filtering where the reference uses the same u8 image, so there
/// is no quantisation mismatch to explain.
fn input_photo_bilateral() -> GrayImage {
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

fn main() {
    let low = input_low_contrast();
    let bi = input_bimodal();
    let bi_f = input_photo_bilateral();
    emit_image("input_low_contrast", &low);
    emit_image("input_bimodal", &bi);
    emit_image("input_bilateral", &bi_f);

    // ── histogram equalisation ─────────────────────────────────────────────
    // The Rust LUT is the textbook (cdf - cdf_min) / (total - cdf_min) * 255,
    // rounded. OpenCV's `equalizeHist` uses exactly that formula.
    for (name, img) in [("low_contrast", &low), ("bimodal", &bi)] {
        let case = format!("equalize_{name}");
        emit_meta(&case, 0);
        let out = histogram_equalization(img);
        emit_image(&case, &out);
    }

    // ── CLAHE ──────────────────────────────────────────────────────────────
    for (name, img) in [("low_contrast", &low), ("bimodal", &bi)] {
        for &clip in &[2.0f32, 4.0] {
            let case = format!("clahe_{name}_c{clip:?}_8x8");
            emit_clahe(&case, clip, 8, 8);
            let out = clahe(img, clip, (8, 8));
            emit_image(&case, &out);
        }
    }

    // ── bilateral ──────────────────────────────────────────────────────────
    // `d` is the full kernel diameter; radius = d/2 on both sides.
    for &d in &[5i32, 9] {
        for &(sc, ss) in &[(20.0f32, 10.0f32), (50.0, 50.0)] {
            let case = format!("bilateral_d{d}_sc{sc:?}_ss{ss:?}");
            emit_bilateral(&case, d, sc, ss);
            emit_meta(&case, (d / 2) as usize);
            let out = bilateral_filter(&bi_f, d, sc, ss);
            emit_image(&case, &out);
        }
    }
}
