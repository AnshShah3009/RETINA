//! SAD block matching in O(W * H * D) via per-disparity running box sums.
//!
//! The dense matcher in `CpuBackend::stereo_match` evaluates, for every pixel
//! and disparity, a `B x B` sum of absolute differences: O(W * H * D * B^2).
//! For a fixed disparity `d` that cost is a box filter of the absolute
//! difference image `A_d(r, c) = |L(r, c) - R(r, c - d)|`, so it can be
//! maintained with running sums: per-column sums over the `B` rows of the
//! window (slid down one row at a time) and a running sum along the row.
//!
//! The running sums add and subtract, so they only reproduce the dense loop's
//! float sums exactly when every partial sum is an exactly representable
//! integer. [`integer_pixels`] checks that precondition (integer-valued
//! pixels, and `B^2 * 2 * max|v|` within the type's exact-integer range); the
//! caller keeps the dense loop for anything else. Under it both computations
//! are exact, so the costs, their `<` comparisons in disparity order and the
//! chosen disparities are identical.

use cv_core::Float;
use rayon::prelude::*;

/// Largest power of two `M` such that every integer in `[-M, M]` is exact in `T`.
fn exact_integer_limit<T: Float>() -> f64 {
    let mut limit = 1.0f64;
    for k in 1..=52 {
        let probe = 2f64.powi(k) + 1.0;
        if Float::to_f64(T::from_f64(probe)) != probe {
            break;
        }
        limit = 2f64.powi(k + 1);
    }
    limit
}

/// Convert both images to `i32` if all pixels are integers small enough that
/// every SAD partial sum over a `block x block` window is exact in `T` (and in
/// `u32`). `None` means the dense float loop must run.
pub fn integer_pixels<T: Float>(
    left: &[T],
    right: &[T],
    block: usize,
) -> Option<(Vec<i32>, Vec<i32>)> {
    const MAX_ABS: f64 = (1u32 << 20) as f64;
    let mut max_abs = 0.0f64;
    let mut convert = |src: &[T]| -> Option<Vec<i32>> {
        let mut out = Vec::with_capacity(src.len());
        for v in src {
            let f = Float::to_f64(*v);
            // `!(a <= b)` also rejects NaN.
            if !(f.abs() <= MAX_ABS) || f.trunc() != f {
                return None;
            }
            max_abs = max_abs.max(f.abs());
            out.push(f as i32);
        }
        Some(out)
    };
    let l = convert(left)?;
    let r = convert(right)?;
    let worst = (block * block) as f64 * 2.0 * max_abs;
    (worst <= exact_integer_limit::<T>() && worst <= u32::MAX as f64).then_some((l, r))
}

/// Best disparity per pixel (`None` = no candidate tested), with the dense
/// loop's semantics: window rows/cols `[y - B/2, y - B/2 + B)`, pixel valid when
/// `B/2 <= x` and `x + B/2 < w` (same for `y`, and for `rx = x - disp`),
/// disparities tried in increasing order and replaced only on strictly
/// smaller cost. `block >= 1`.
pub fn block_match_sad(
    left: &[i32],
    right: &[i32],
    w: usize,
    h: usize,
    block: usize,
    min_disp: i32,
    num_disp: i32,
) -> Vec<Option<i32>> {
    assert!(block >= 1);
    assert_eq!(left.len(), w * h);
    assert_eq!(right.len(), w * h);
    let half = block / 2;
    let mut out = vec![None; w * h];
    if num_disp <= 0 || h < 2 * half + 1 || w < 2 * half + 1 {
        return out;
    }
    // Valid output rows/cols: half <= v && v + half < n (hi exclusive).
    let (y_lo, y_hi) = (half, h - half);
    let (x_lo, x_hi) = (half as i64, (w - half) as i64);
    let nd = num_disp as usize;
    fn row(img: &[i32], w: usize, r: usize) -> &[i32] {
        &img[r * w..(r + 1) * w]
    }

    const BAND: usize = 16;
    out[y_lo * w..y_hi * w]
        .par_chunks_mut(w * BAND)
        .enumerate()
        .for_each(|(band, out_band)| {
            let y0 = y_lo + band * BAND;
            let rows = out_band.len() / w;
            // colsum[d * w + c] = sum over the window rows of A_d(r, c), kept
            // for the columns where both c and c - disp lie inside the image.
            let mut colsum = vec![0u32; nd * w];
            let mut best = vec![u64::MAX; w];
            for dy in 0..rows {
                let y = y0 + dy;
                let out_row = &mut out_band[dy * w..(dy + 1) * w];
                best.fill(u64::MAX);
                for d in 0..nd {
                    let disp = min_disp as i64 + d as i64;
                    // Columns c with 0 <= c < w and 0 <= c - disp < w.
                    let c_lo = disp.clamp(0, w as i64) as usize;
                    let c_hi = (w as i64 + disp).clamp(c_lo as i64, w as i64) as usize;
                    if c_lo == c_hi {
                        // |disp| >= w: no column pairs, so no testable pixel.
                        continue;
                    }
                    let r_lo = (c_lo as i64 - disp) as usize;
                    let r_hi = (c_hi as i64 - disp) as usize;
                    let cs = &mut colsum[d * w..(d + 1) * w];
                    if dy == 0 {
                        cs.fill(0);
                        for r in y - half..y - half + block {
                            let l = &row(left, w, r)[c_lo..c_hi];
                            let rr = &row(right, w, r)[r_lo..r_hi];
                            for ((c, &a), &b) in cs[c_lo..c_hi].iter_mut().zip(l).zip(rr) {
                                *c += a.abs_diff(b);
                            }
                        }
                    } else {
                        let (add, sub) = (y - half + block - 1, y - half - 1);
                        let la = &row(left, w, add)[c_lo..c_hi];
                        let ra = &row(right, w, add)[r_lo..r_hi];
                        let ls = &row(left, w, sub)[c_lo..c_hi];
                        let rs = &row(right, w, sub)[r_lo..r_hi];
                        for ((((c, &la), &ra), &ls), &rs) in
                            cs[c_lo..c_hi].iter_mut().zip(la).zip(ra).zip(ls).zip(rs)
                        {
                            *c = *c + la.abs_diff(ra) - ls.abs_diff(rs);
                        }
                    }
                    // Pixels x with x and rx = x - disp both valid.
                    let xa = x_lo.max(x_lo + disp);
                    let xb = x_hi.min(x_hi + disp);
                    if xa >= xb {
                        continue;
                    }
                    let (xa, xb) = (xa as usize, xb as usize);
                    let mut s: u64 = cs[xa - half..xa - half + block]
                        .iter()
                        .map(|&v| v as u64)
                        .sum();
                    for x in xa..xb {
                        if x > xa {
                            s += cs[x - half + block - 1] as u64;
                            s -= cs[x - half - 1] as u64;
                        }
                        if s < best[x] {
                            best[x] = s;
                            out_row[x] = Some(disp as i32);
                        }
                    }
                }
            }
        });
    out
}
