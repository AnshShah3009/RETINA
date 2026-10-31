//! OpenCV-exact Canny on 8-bit images, shared by every backend.
//!
//! `cv::Canny` (aperture 3) on a `CV_8U` image: 3x3 Sobel with
//! BORDER_REPLICATE, integer L1 magnitude `|dx| + |dy|` or squared L2
//! `dx² + dy²` (thresholds squared to match), non-maximum suppression strict
//! towards left/up (`>`) and non-strict towards right/down (`>=`), strict on
//! both diagonal neighbours, magnitude outside the image 0, thresholds strict
//! (`m > low`, `m > high`), 8-connected hysteresis. Everything is integer
//! arithmetic, so the GPU kernel (`shaders/canny.wgsl`) reproduces it bit for
//! bit; it shares [`canny_thresholds`] and [`canny_hysteresis`] with this
//! reference.

use rayon::prelude::*;

/// Fixed-point shift of OpenCV's `tan(22.5°)` NMS test.
pub const CANNY_TG_SHIFT: i64 = 15;

/// `round(tan(22.5°) * 2^15)`, as OpenCV computes it.
pub fn canny_tg22() -> i64 {
    (0.414_213_562_373_095_05 * (1i64 << CANNY_TG_SHIFT) as f64 + 0.5) as i64
}

/// Integer thresholds `(low, high)` as `cv::Canny` derives them: swapped when
/// `low > high`; with `l2` clamped to 32767 and squared (positive values
/// only); then floored.
pub fn canny_thresholds(low: f64, high: f64, l2: bool) -> (i64, i64) {
    let (mut lo, mut hi) = (low, high);
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    if l2 {
        lo = lo.min(32767.0);
        hi = hi.min(32767.0);
        if lo > 0.0 {
            lo *= lo;
        }
        if hi > 0.0 {
            hi *= hi;
        }
    }
    (lo.floor() as i64, hi.floor() as i64)
}

/// 3x3 Sobel with BORDER_REPLICATE, as `cv::Canny` computes it.
fn sobel3_replicate(data: &[u8], w: usize, h: usize) -> (Vec<i32>, Vec<i32>) {
    let mut dx = vec![0i32; w * h];
    let mut dy = vec![0i32; w * h];
    dx.par_chunks_mut(w)
        .zip(dy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (dx_row, dy_row))| {
            let r0 = &data[y.saturating_sub(1) * w..][..w];
            let r1 = &data[y * w..][..w];
            let r2 = &data[(y + 1).min(h - 1) * w..][..w];
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let p = |r: &[u8], i: usize| r[i] as i32;
                dx_row[x] =
                    (p(r0, xp) - p(r0, xm)) + 2 * (p(r1, xp) - p(r1, xm)) + (p(r2, xp) - p(r2, xm));
                dy_row[x] =
                    (p(r2, xm) + 2 * p(r2, x) + p(r2, xp)) - (p(r0, xm) + 2 * p(r0, x) + p(r0, xp));
            }
        });
    (dx, dy)
}

/// 8-connected hysteresis over an NMS state map (0 = none, 1 = weak,
/// 2 = strong); returns the 0/255 edge mask. The result is the set of weak
/// pixels connected to a strong one, independent of visiting order.
pub fn canny_hysteresis(state: &mut [u8], w: usize, h: usize) -> Vec<u8> {
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| state[i] == 2).collect();
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let n = ny * w + nx;
                if state[n] == 1 {
                    state[n] = 2;
                    stack.push(n);
                }
            }
        }
    }
    state
        .par_iter()
        .map(|&s| if s == 2 { 255 } else { 0 })
        .collect()
}

/// `cv2.Canny(src, low, high, L2gradient=l2)` on a row-major `w x h` 8-bit
/// image; returns the 0/255 edge mask.
pub fn canny_u8(src: &[u8], w: usize, h: usize, low: f64, high: f64, l2: bool) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    assert_eq!(src.len(), w * h, "canny_u8: buffer is not w * h");
    let (lo, hi) = canny_thresholds(low, high, l2);

    let (dx, dy) = sobel3_replicate(src, w, h);
    let mag: Vec<i64> = dx
        .par_iter()
        .zip(dy.par_iter())
        .map(|(&gx, &gy)| {
            let (gx, gy) = (gx as i64, gy as i64);
            if l2 {
                gx * gx + gy * gy
            } else {
                gx.abs() + gy.abs()
            }
        })
        .collect();
    let at = |x: isize, y: isize| -> i64 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            0
        } else {
            mag[y as usize * w + x as usize]
        }
    };

    let tg22 = canny_tg22();
    let mut state = vec![0u8; w * h];
    state.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let yi = y as isize;
        for (x, st) in row.iter_mut().enumerate() {
            let j = y * w + x;
            let m = mag[j];
            if m <= lo {
                continue;
            }
            let xi = x as isize;
            let (xs, ys) = (dx[j] as i64, dy[j] as i64);
            let ax = xs.abs();
            let ay = ys.abs() << CANNY_TG_SHIFT;
            let tg22x = ax * tg22;
            let is_max = if ay < tg22x {
                m > at(xi - 1, yi) && m >= at(xi + 1, yi)
            } else {
                let tg67x = tg22x + (ax << (CANNY_TG_SHIFT + 1));
                if ay > tg67x {
                    m > at(xi, yi - 1) && m >= at(xi, yi + 1)
                } else {
                    let s: isize = if (xs ^ ys) < 0 { -1 } else { 1 };
                    m > at(xi - s, yi - 1) && m > at(xi + s, yi + 1)
                }
            };
            if is_max {
                *st = if m > hi { 2 } else { 1 };
            }
        }
    });

    canny_hysteresis(&mut state, w, h)
}

/// Quantise an arbitrary float image to the 8-bit input `cv::Canny`
/// requires: round half to even, saturate to 0..=255 (NaN -> 0).
pub fn quantize_u8(v: f32) -> u8 {
    if v.is_nan() {
        0
    } else {
        v.round_ties_even().clamp(0.0, 255.0) as u8
    }
}
