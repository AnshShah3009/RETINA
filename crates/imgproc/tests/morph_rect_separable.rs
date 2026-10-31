//! Rectangular-kernel erode / dilate take a separable min/max path; it must
//! return exactly the bytes of the per-tap loop it bypasses, for every border
//! mode. `dense` below is that loop, verbatim apart from packaging.

use cv_imgproc::{
    create_morph_kernel, dilate_with_border_into, erode_with_border_into, BorderMode, MorphShape,
};
use image::GrayImage;
use std::time::Instant;

fn map_coord(coord: i32, len: i32, border: BorderMode) -> Option<i32> {
    match border {
        BorderMode::Constant(_) => (coord >= 0 && coord < len).then_some(coord),
        BorderMode::Replicate => Some(coord.clamp(0, len - 1)),
        BorderMode::Wrap => Some(coord.rem_euclid(len)),
        BorderMode::Reflect => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len;
            let c = coord.rem_euclid(period);
            Some(if c >= len { period - c - 1 } else { c })
        }
        BorderMode::Reflect101 => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * len - 2;
            let c = coord.rem_euclid(period);
            Some(if c >= len { period - c } else { c })
        }
    }
}

/// The pre-existing per-tap loop (one iteration).
fn dense(src: &GrayImage, kernel: &[(i32, i32)], border: BorderMode, is_max: bool) -> GrayImage {
    let (w, h) = (src.width() as i32, src.height() as i32);
    let data = src.as_raw();
    let mut out = GrayImage::new(src.width(), src.height());
    for y in 0..h {
        for x in 0..w {
            let mut v = if is_max { 0u8 } else { 255u8 };
            for &(kx, ky) in kernel {
                let p = match (map_coord(x + kx, w, border), map_coord(y + ky, h, border)) {
                    (Some(ix), Some(iy)) => data[(iy * w + ix) as usize],
                    _ => match border {
                        BorderMode::Constant(c) => c,
                        _ if is_max => 0,
                        _ => 255,
                    },
                };
                v = if is_max { v.max(p) } else { v.min(p) };
            }
            out.as_mut()[(y * w + x) as usize] = v;
        }
    }
    out
}

fn noise(w: u32, h: u32, seed: u64) -> GrayImage {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    GrayImage::from_fn(w, h, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        image::Luma([(s >> 24) as u8])
    })
}

fn rect(x0: i32, x1: i32, y0: i32, y1: i32) -> Vec<(i32, i32)> {
    (y0..=y1)
        .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
        .collect()
}

#[test]
fn rectangle_morphology_is_byte_identical_to_per_tap_loop() {
    let borders = [
        BorderMode::Constant(0),
        BorderMode::Constant(77),
        BorderMode::Constant(255),
        BorderMode::Replicate,
        BorderMode::Reflect,
        BorderMode::Reflect101,
        BorderMode::Wrap,
    ];
    let mut kernels: Vec<Vec<(i32, i32)>> =
        [(1, 1), (3, 3), (2, 4), (5, 1), (1, 7), (17, 3), (21, 21)]
            .iter()
            .map(|&(kw, kh)| create_morph_kernel(MorphShape::Rectangle, kw, kh))
            .collect();
    // Off-centre rectangles, one wider than the image, and a duplicate tap.
    kernels.push(rect(2, 4, -3, -1));
    kernels.push(rect(-40, 3, 0, 1));
    let mut dup = rect(-1, 1, -1, 1);
    dup.push((0, 0));
    kernels.push(dup);

    for (i, &(w, h)) in [(1u32, 1u32), (1, 9), (9, 1), (2, 2), (31, 17), (70, 23)]
        .iter()
        .enumerate()
    {
        let img = noise(w, h, i as u64);
        for kernel in &kernels {
            for &border in &borders {
                let mut out = GrayImage::new(w, h);
                dilate_with_border_into(&img, &mut out, kernel, 1, border);
                assert_eq!(
                    out.as_raw(),
                    dense(&img, kernel, border, true).as_raw(),
                    "dilate {w}x{h} {border:?} kernel {kernel:?}"
                );
                erode_with_border_into(&img, &mut out, kernel, 1, border);
                assert_eq!(
                    out.as_raw(),
                    dense(&img, kernel, border, false).as_raw(),
                    "erode {w}x{h} {border:?} kernel {kernel:?}"
                );
                // Iterations chain the same single step.
                erode_with_border_into(&img, &mut out, kernel, 2, border);
                let twice = dense(&dense(&img, kernel, border, false), kernel, border, false);
                assert_eq!(out.as_raw(), twice.as_raw());
            }
        }
    }
}

/// `cargo test --release -p cv-imgproc --test morph_rect_separable -- --ignored --nocapture`
#[test]
#[ignore]
fn rectangle_morphology_timing() {
    let img = noise(1920, 1080, 3);
    for k in [3u32, 7, 15, 31] {
        let kernel = create_morph_kernel(MorphShape::Rectangle, k, k);
        let mut out = GrayImage::new(1920, 1080);
        let reps = 5;
        erode_with_border_into(&img, &mut out, &kernel, 1, BorderMode::Replicate);
        let t = Instant::now();
        for _ in 0..reps {
            erode_with_border_into(&img, &mut out, &kernel, 1, BorderMode::Replicate);
        }
        let t_new = t.elapsed().as_secs_f64() / reps as f64;
        // The old path is the per-tap loop, which the general (non-rectangle)
        // code still runs: a rectangle with its last tap removed costs the
        // same per-tap work minus one tap, so time that as "before".
        let mut not_rect = kernel.clone();
        not_rect.pop();
        let mut slow = GrayImage::new(1920, 1080);
        erode_with_border_into(&img, &mut slow, &not_rect, 1, BorderMode::Replicate);
        let t = Instant::now();
        for _ in 0..reps {
            erode_with_border_into(&img, &mut slow, &not_rect, 1, BorderMode::Replicate);
        }
        let t_old = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "erode 1920x1080 rect {k:2}x{k:<2} replicate: per-tap {:8.2} ms, separable {:6.2} ms ({:.0}x)",
            t_old * 1e3,
            t_new * 1e3,
            t_old / t_new
        );
    }
}
