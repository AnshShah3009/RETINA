//! Numerical-parity number generator: resampling & geometry (`cv-imgproc`)
//! vs OpenCV. Companion reference side: `parity/parity_geometry.py`.
//!
//! Same line protocol as `parity_filters.rs`, plus
//!
//! ```text
//! #W <name> <width> <height> <v0> ...   # warped/resampled image (row-major)
//! ```

use cv_imgproc::{remap_ex, resize, warp_affine_ex, BorderMode, Interpolation};
use image::{GrayImage, Luma};

const W: u32 = 48;
const H: u32 = 36;
const TAU: f64 = std::f64::consts::PI * 2.0;

fn emit_image(name: &str, img: &GrayImage) {
    let mut line = format!("#IM {} {} {}", name, img.width(), img.height());
    for p in img.as_raw() {
        line.push_str(&format!(" {:.1}", *p as f64));
    }
    println!("{line}");
}

fn emit_warp(name: &str, img: &GrayImage) {
    let mut line = format!("#W {} {} {}", name, img.width(), img.height());
    for p in img.as_raw() {
        line.push_str(&format!(" {:.1}", *p as f64));
    }
    println!("{line}");
}

fn emit_meta(name: &str, radius: usize) {
    println!("#M {} {}", name, radius);
}

fn emit_f32(name: &str, v: f32) {
    println!("#F {} {:.17e}", name, v);
}

/// Smooth closed-form content: nothing saturates, no hard edges to alias.
fn input_smooth() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let theta = TAU * (2.0 * x as f64 / W as f64 + 1.0 * y as f64 / H as f64);
            let v = (127.0 + 100.0 * theta.sin()).round();
            img.put_pixel(x, y, Luma([v.clamp(0.0, 255.0) as u8]));
        }
    }
    img
}

/// Alternating checkerboard, the adversarial input for any resampler: it
/// aliases maximally under downscale and any off-by-half-pixel in the source
/// coordinate mapping shows up immediately as a large deviation.
fn input_checker() -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = if (x + y) % 2 == 0 { 40 } else { 200 };
            img.put_pixel(x, y, Luma([v]));
        }
    }
    img
}

/// Forward (src -> dst) affine matrix built the OpenCV way, from
/// `cv::getRotationMatrix2D` composed with a translation. Returned as the
/// 2x3 form that the Rust entry point takes, which is the same matrix
/// `cv::warpAffine` takes.
fn affine_rotate_translate(angle_deg: f64, tx: f64, ty: f64) -> [[f32; 3]; 2] {
    let a = angle_deg * std::f64::consts::PI / 180.0;
    let (c, s) = (a.cos(), a.sin());
    let cx = (W as f64 - 1.0) / 2.0;
    let cy = (H as f64 - 1.0) / 2.0;
    // getRotationMatrix2D(center, angle, 1.0)
    let m00 = c;
    let m01 = s;
    let m02 = (1.0 - c) * cx + s * cy;
    let m10 = -s;
    let m11 = c;
    let m12 = s * cx - (1.0 - c) * cy;
    [
        [(m00) as f32, (m01) as f32, (m02 + tx) as f32],
        [(m10) as f32, (m11) as f32, (m12 + ty) as f32],
    ]
}

fn main() {
    let src = input_smooth();
    let checker = input_checker();
    emit_image("input_smooth", &src);
    emit_image("input_checker", &checker);

    // ── resize ─────────────────────────────────────────────────────────────
    // Up- and down-scale, one case per interpolation mode.
    let targets = [(24u32, 18u32), (96, 72), (13, 40)];
    let modes = [
        ("nearest", Interpolation::Nearest),
        ("linear", Interpolation::Linear),
        ("cubic", Interpolation::Cubic),
        ("lanczos", Interpolation::Lanczos),
    ];
    for (iname, img) in [("smooth", &src), ("checker", &checker)] {
        for (tw, th) in targets {
            for (mname, mode) in modes {
                let case = format!("resize_{iname}_{mname}_{tw}x{th}");
                // Radius = half the widest kernel support in the mode
                // (Lanczos-3 -> 3), so the reported interior is guaranteed
                // kernel-free even though the Rust resamplers edge-clamp.
                let radius = match mode {
                    Interpolation::Nearest | Interpolation::Linear => 1,
                    Interpolation::Cubic => 2,
                    Interpolation::Lanczos => 3,
                };
                emit_meta(&case, radius);
                let out = resize(img, tw, th, mode);
                emit_image(&case, &out);
            }
        }
    }

    // ── warpAffine ─────────────────────────────────────────────────────────
    // Linear / BORDER_CONSTANT(0) is OpenCV's documented default pair.
    let m = affine_rotate_translate(20.0, 4.0, -3.0);
    emit_f32("warpM00", m[0][0]);
    emit_f32("warpM01", m[0][1]);
    emit_f32("warpM02", m[0][2]);
    emit_f32("warpM10", m[1][0]);
    emit_f32("warpM11", m[1][1]);
    emit_f32("warpM12", m[1][2]);

    for (iname, img) in [("smooth", &src), ("checker", &checker)] {
        for (bname, border) in [
            ("c0", BorderMode::Constant(0)),
            ("replicate", BorderMode::Replicate),
        ] {
            let case = format!("warpaffine_{iname}_linear_{bname}");
            emit_meta(&case, 1);
            let out = warp_affine_ex(img, m, W, H, Interpolation::Linear, border);
            emit_warp(&case, &out);
        }
        // Nearest sampling isolates the coordinate transform from the
        // interpolation kernel: any residual is the transform itself.
        let case = format!("warpaffine_{iname}_nearest_c0");
        emit_meta(&case, 0);
        let out = warp_affine_ex(
            img,
            m,
            W,
            H,
            Interpolation::Nearest,
            BorderMode::Constant(0),
        );
        emit_warp(&case, &out);
    }

    // ── remap ──────────────────────────────────────────────────────────────
    // Smooth analytic displacement field: a pure translation plus a radial
    // quadratic. Known in closed form on both sides, and its gradients stay
    // below 1 so no folding occurs.
    let (dw, dh) = (W, H);
    let mut mx = vec![0f32; (dw * dh) as usize];
    let mut my = vec![0f32; (dw * dh) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let idx = (y * dw + x) as usize;
            let fx = x as f64;
            let fy = y as f64;
            let dx = 3.0 + 0.5 * fy * fy / (H as f64);
            let dy = -2.0 + 0.5 * fx * fx / (W as f64);
            mx[idx] = (fx + dx) as f32;
            my[idx] = (fy + dy) as f32;
        }
    }
    for (iname, img) in [("smooth", &src), ("checker", &checker)] {
        for (mname, mode) in modes {
            let case = format!("remap_{iname}_{mname}_c0");
            let radius = match mode {
                Interpolation::Nearest | Interpolation::Linear => 1,
                Interpolation::Cubic => 2,
                Interpolation::Lanczos => 3,
            };
            emit_meta(&case, radius);
            let out = remap_ex(img, &mx, &my, dw, dh, mode, BorderMode::Constant(0));
            emit_warp(&case, &out);
        }
    }
}
