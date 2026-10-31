//! Separable erode / dilate for rectangular structuring elements.
//!
//! A rectangle `[x0, x1] x [y0, y1]` of offsets is the product of a row
//! segment and a column segment, and min / max are associative, so
//!
//! ```text
//! ext_{dy, dx} P(y + dy, x + dx) = ext_dy ( ext_dx P(y + dy, x + dx) )
//! ```
//!
//! for any padded image `P`. With border handling expressed as per-axis index
//! maps (`P(py, px) = src[row(py)][col(px)]`, or `fill` when either map has no
//! source index) this is exactly the per-tap definition, so the result is
//! bit-identical to the dense O(kw * kh) loop for every border mode. The
//! horizontal pass uses the van Herk / Gil-Werman algorithm (three min/max per
//! pixel, independent of `kw`) for wide windows and contiguous slice min/max
//! (auto-vectorised) for narrow ones; the vertical pass combines whole rows
//! with slice min/max.

use rayon::prelude::*;

/// Window width from which the horizontal pass switches from `kw` shifted
/// slice passes to van Herk / Gil-Werman.
const VAN_HERK_MIN_WIDTH: usize = 16;

#[inline(always)]
fn ext(a: u8, b: u8, is_max: bool) -> u8 {
    if is_max {
        a.max(b)
    } else {
        a.min(b)
    }
}

/// `acc[i] = ext(acc[i], src[i])` over equal-length slices.
#[inline]
fn ext_into(acc: &mut [u8], src: &[u8], is_max: bool) {
    if is_max {
        for (a, &s) in acc.iter_mut().zip(src) {
            *a = (*a).max(s);
        }
    } else {
        for (a, &s) in acc.iter_mut().zip(src) {
            *a = (*a).min(s);
        }
    }
}

/// Sliding-window extremum of width `k` over `buf`: `out[x] = ext(buf[x..x + k])`
/// for `x in 0..out.len()`, `buf.len() == out.len() + k - 1`.
fn sliding_ext(
    buf: &[u8],
    k: usize,
    out: &mut [u8],
    is_max: bool,
    g: &mut Vec<u8>,
    h: &mut Vec<u8>,
) {
    let w = out.len();
    debug_assert_eq!(buf.len(), w + k - 1);
    if k < VAN_HERK_MIN_WIDTH {
        out.copy_from_slice(&buf[..w]);
        for s in 1..k {
            ext_into(out, &buf[s..s + w], is_max);
        }
        return;
    }
    // van Herk / Gil-Werman: block-wise prefix (g) and suffix (h) extrema;
    // a window of width k straddles at most one block boundary, so
    // ext(buf[x..x + k]) = ext(h[x], g[x + k - 1]).
    let n = buf.len();
    g.clear();
    g.resize(n, 0);
    h.clear();
    h.resize(n, 0);
    for start in (0..n).step_by(k) {
        let end = (start + k).min(n);
        let mut acc = buf[start];
        for j in start..end {
            acc = ext(acc, buf[j], is_max);
            g[j] = acc;
        }
        let mut acc = buf[end - 1];
        for j in (start..end).rev() {
            acc = ext(acc, buf[j], is_max);
            h[j] = acc;
        }
    }
    for x in 0..w {
        out[x] = ext(h[x], g[x + k - 1], is_max);
    }
}

/// Erode (`is_max == false`) or dilate (`is_max == true`) one `w x h` plane by
/// the rectangle of offsets `[x0, x1] x [y0, y1]` (inclusive, `x0 <= x1`,
/// `y0 <= y1`).
///
/// The rectangle is given by its size `kw x kh` and the border maps: output
/// pixel `(x, y)` combines padded columns `x..x + kw` and padded rows
/// `y..y + kh`, where padded column `j` stands for image column `j + x0` and
/// reads source column `col_map[j]` (`col_map.len() == w + kw - 1`), and
/// likewise `row_map` for rows. `None` means "outside the image": the tap
/// reads `fill`.
#[allow(clippy::too_many_arguments)]
pub fn rect_extremum(
    src: &[u8],
    w: usize,
    h: usize,
    dst: &mut [u8],
    kw: usize,
    kh: usize,
    col_map: &[Option<usize>],
    row_map: &[Option<usize>],
    fill: u8,
    is_max: bool,
) {
    assert_eq!(src.len(), w * h);
    assert_eq!(dst.len(), w * h);
    assert!(kw >= 1 && kh >= 1);
    assert_eq!(col_map.len(), w + kw - 1);
    assert_eq!(row_map.len(), h + kh - 1);
    if w == 0 || h == 0 {
        return;
    }

    // Horizontal pass over every source row.
    let mut hpass = vec![0u8; w * h];
    hpass
        .par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each_init(
            || (vec![0u8; w + kw - 1], Vec::new(), Vec::new()),
            |(buf, g, hh), (out_row, row)| {
                for (b, m) in buf.iter_mut().zip(col_map) {
                    *b = match m {
                        Some(c) => row[*c],
                        None => fill,
                    };
                }
                sliding_ext(buf, kw, out_row, is_max, g, hh);
            },
        );

    // Vertical pass: combine kh horizontally-reduced rows. A padded row with
    // no source row is `fill` at every column.
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let rows = &row_map[y..y + kh];
        let mut any_fill = false;
        let mut first = true;
        for r in rows {
            match r {
                Some(r) => {
                    let s = &hpass[r * w..(r + 1) * w];
                    if first {
                        out_row.copy_from_slice(s);
                        first = false;
                    } else {
                        ext_into(out_row, s, is_max);
                    }
                }
                None => any_fill = true,
            }
        }
        if first {
            out_row.fill(fill);
        } else if any_fill {
            for v in out_row.iter_mut() {
                *v = ext(*v, fill, is_max);
            }
        }
    });
}

/// If `offsets` is exactly the set of points of an axis-aligned rectangle
/// (duplicates allowed), return `(x0, x1, y0, y1)`.
pub fn as_rectangle(offsets: &[(i32, i32)]) -> Option<(i32, i32, i32, i32)> {
    let (&(fx, fy), _) = offsets.split_first()?;
    let (mut x0, mut x1, mut y0, mut y1) = (fx, fx, fy, fy);
    for &(x, y) in offsets {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let bw = (x1 as i64 - x0 as i64 + 1) as usize;
    let bh = (y1 as i64 - y0 as i64 + 1) as usize;
    let area = bw.checked_mul(bh)?;
    if area > offsets.len() {
        return None;
    }
    let mut seen = vec![false; area];
    let mut distinct = 0usize;
    for &(x, y) in offsets {
        let i = (y - y0) as usize * bw + (x - x0) as usize;
        if !seen[i] {
            seen[i] = true;
            distinct += 1;
        }
    }
    (distinct == area).then_some((x0, x1, y0, y1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(
        src: &[u8],
        w: usize,
        h: usize,
        x0: i32,
        x1: i32,
        y0: i32,
        y1: i32,
        is_max: bool,
    ) -> Vec<u8> {
        let fill = if is_max { 0 } else { 255 };
        let mut out = vec![0u8; w * h];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let mut v = fill;
                for dy in y0..=y1 {
                    for dx in x0..=x1 {
                        let (sx, sy) = (x + dx, y + dy);
                        let p = if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                            fill
                        } else {
                            src[sy as usize * w + sx as usize]
                        };
                        v = ext(v, p, is_max);
                    }
                }
                out[y as usize * w + x as usize] = v;
            }
        }
        out
    }

    #[test]
    fn matches_dense_identity_border() {
        let (w, h) = (45usize, 23usize);
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 7919) % 251) as u8).collect();
        for &(x0, x1, y0, y1) in &[
            (0, 0, 0, 0),
            (-1, 1, -1, 1),
            (-2, 1, 0, 3),
            (-9, 10, -3, 2),
            (-30, 30, -1, 0),
            (3, 5, -6, -2),
        ] {
            for is_max in [false, true] {
                let fill = if is_max { 0 } else { 255 };
                let kw = (x1 - x0 + 1) as usize;
                let kh = (y1 - y0 + 1) as usize;
                let map = |n: usize, lo: i32, k: usize| -> Vec<Option<usize>> {
                    (0..n + k - 1)
                        .map(|j| {
                            let c = j as i32 + lo;
                            (c >= 0 && c < n as i32).then_some(c as usize)
                        })
                        .collect()
                };
                let mut out = vec![0u8; w * h];
                rect_extremum(
                    &src,
                    w,
                    h,
                    &mut out,
                    kw,
                    kh,
                    &map(w, x0, kw),
                    &map(h, y0, kh),
                    fill,
                    is_max,
                );
                assert_eq!(out, dense(&src, w, h, x0, x1, y0, y1, is_max));
            }
        }
    }

    #[test]
    fn rectangle_detection() {
        let rect: Vec<(i32, i32)> = (-1..=1)
            .flat_map(|y| (-2..=0).map(move |x| (x, y)))
            .collect();
        assert_eq!(as_rectangle(&rect), Some((-2, 0, -1, 1)));
        let mut dup = rect.clone();
        dup.push((0, 0));
        assert_eq!(as_rectangle(&dup), Some((-2, 0, -1, 1)));
        let cross = vec![(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)];
        assert_eq!(as_rectangle(&cross), None);
        assert_eq!(as_rectangle(&[]), None);
        assert_eq!(as_rectangle(&[(0, 0), (5, 5)]), None);
    }
}
