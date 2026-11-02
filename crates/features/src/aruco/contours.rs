//! Contour extraction for the ArUco candidate search: OpenCV's
//! `findContours(RETR_LIST, CHAIN_APPROX_NONE)`, `approxPolyDP` on a closed
//! curve and `isContourConvex`, on integer pixel coordinates.
//!
//! `cv_imgproc::find_contours` only starts a hole border on a pixel that no
//! outer border has visited, so the inner border of a ring one or two pixels
//! thick (the black frame of a small marker) is never traced, and
//! `cv_imgproc::approx_poly_dp` keeps the contour's first point as a vertex,
//! which turns most quadrilaterals into pentagons. Both are ported here from
//! OpenCV instead.

/// Chain-code steps, counter-clockwise on screen starting east (OpenCV's
/// `icvCodeDeltas`): `x` right, `y` down.
const STEPS: [(isize, isize); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Every border of the non-zero regions of `binary` (`w` x `h`, row-major),
/// outer borders and hole borders alike, as the 8-connected sequence of
/// border pixels (Suzuki-Abe border following, as OpenCV's `findContours`
/// with `RETR_LIST` and `CHAIN_APPROX_NONE`). Like OpenCV, the one-pixel frame
/// of the image is treated as background.
pub(super) fn find_contours_list(binary: &[u8], w: usize, h: usize) -> Vec<Vec<(i32, i32)>> {
    if w < 3 || h < 3 || binary.len() != w * h {
        return Vec::new();
    }
    // 0 background, 1 unvisited foreground, 2 border pixel already followed,
    // -2 border pixel whose east neighbour is background ("right bound").
    // RETR_LIST builds no hierarchy, so one border number is enough.
    let mut img: Vec<i8> = binary.iter().map(|&v| i8::from(v != 0)).collect();
    for x in 0..w {
        img[x] = 0;
        img[(h - 1) * w + x] = 0;
    }
    for y in 0..h {
        img[y * w] = 0;
        img[y * w + w - 1] = 0;
    }
    let mut out = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w {
            let p = img[y * w + x];
            let prev = img[y * w + x - 1];
            if p == 1 && prev == 0 {
                out.push(follow_border(&mut img, w, x, y, false));
            } else if p == 0 && prev >= 1 {
                out.push(follow_border(&mut img, w, x - 1, y, true));
            }
        }
    }
    out
}

/// Follow one border starting at `(x0, y0)` (OpenCV's `icvFetchContour`),
/// marking it in `img`. The start pixel never touches the zero frame, so no
/// neighbour lookup leaves the image.
fn follow_border(img: &mut [i8], w: usize, x0: usize, y0: usize, hole: bool) -> Vec<(i32, i32)> {
    let w = w as isize;
    let at = |x: isize, y: isize| (y * w + x) as usize;
    let (x0, y0) = (x0 as isize, y0 as isize);

    // Look clockwise around the start pixel, beginning next to the
    // background pixel that triggered the border (west for an outer border,
    // east for a hole).
    let first = if hole { 0 } else { 4 };
    let mut s = first;
    let mut found = None;
    for _ in 0..8 {
        s = (s + 7) % 8;
        if s == first {
            break;
        }
        if img[at(x0 + STEPS[s].0, y0 + STEPS[s].1)] != 0 {
            found = Some(s);
            break;
        }
    }
    let Some(mut s) = found else {
        // Isolated pixel.
        img[at(x0, y0)] = -2;
        return vec![(x0 as i32, y0 as i32)];
    };
    let (x1, y1) = (x0 + STEPS[s].0, y0 + STEPS[s].1);

    let mut points = Vec::new();
    let (mut x3, mut y3) = (x0, y0);
    loop {
        // Counter-clockwise from the pixel we came from; that pixel itself is
        // the last one examined, so the search always succeeds.
        let mut k = s;
        let mut east_is_background = false;
        let (x4, y4) = loop {
            k += 1;
            let d = k % 8;
            let (nx, ny) = (x3 + STEPS[d].0, y3 + STEPS[d].1);
            if img[at(nx, ny)] != 0 {
                break (nx, ny);
            }
            if d == 0 {
                east_is_background = true;
            }
        };
        s = k % 8;
        let here = at(x3, y3);
        if east_is_background {
            img[here] = -2;
        } else if img[here] == 1 {
            img[here] = 2;
        }
        points.push((x3 as i32, y3 as i32));
        if (x4, y4) == (x0, y0) && (x3, y3) == (x1, y1) {
            break;
        }
        (x3, y3) = (x4, y4);
        s = (s + 4) % 8;
    }
    points
}

/// OpenCV's `approxPolyDP(curve, epsilon, closed = true)`: Douglas-Peucker
/// starting from two approximately farthest points of the curve (so the
/// curve's first point is not forced to be a vertex), followed by OpenCV's
/// clean-up pass that drops vertices lying on an almost straight line.
pub(super) fn approx_poly_dp_closed(src: &[(i32, i32)], epsilon: f64) -> Vec<(i32, i32)> {
    let count = src.len();
    if count == 0 {
        return Vec::new();
    }
    let eps = epsilon * epsilon;
    let pt = |i: usize| (src[i].0 as f64, src[i].1 as f64);
    let mut dst: Vec<(i32, i32)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    // 1. Approximately the two farthest points of the curve.
    let mut pos = 0usize;
    let mut right_start = 0usize;
    let mut le_eps = false;
    let mut start = 0usize;
    for _ in 0..3 {
        pos = (pos + right_start) % count;
        start = pos;
        let (sx, sy) = pt(start);
        let mut max_dist = 0.0;
        for j in 1..count {
            let (x, y) = pt((start + j) % count);
            let d = (x - sx) * (x - sx) + (y - sy) * (y - sy);
            if d > max_dist {
                max_dist = d;
                right_start = j;
            }
        }
        le_eps = max_dist <= eps;
    }
    if le_eps {
        dst.push(src[start]);
    } else {
        let far = (right_start + start) % count;
        stack.push((far, start));
        stack.push((start, far));
    }

    // 2. Split each slice at its farthest point from the chord.
    while let Some((s0, s1)) = stack.pop() {
        let (ex, ey) = pt(s1);
        let (sx, sy) = pt(s0);
        let mut p = (s0 + 1) % count;
        let mut split = s0;
        let le = if p != s1 {
            let (dx, dy) = (ex - sx, ey - sy);
            let mut max_dist = 0.0;
            while p != s1 {
                let (x, y) = pt(p);
                // A chord whose ends coincide (a curve passing a pixel twice)
                // falls back to the distance from that point.
                let d = if dx == 0.0 && dy == 0.0 {
                    (x - sx).hypot(y - sy)
                } else {
                    ((y - sy) * dx - (x - sx) * dy).abs()
                };
                if d > max_dist {
                    max_dist = d;
                    split = p;
                }
                p = (p + 1) % count;
            }
            let norm = if dx == 0.0 && dy == 0.0 {
                1.0
            } else {
                dx * dx + dy * dy
            };
            max_dist * max_dist <= eps * norm
        } else {
            true
        };
        if le {
            dst.push(src[s0]);
        } else {
            stack.push((split, s1));
            stack.push((s0, split));
        }
    }

    // 3. OpenCV's clean-up: remove vertices on an almost straight line,
    // reading ahead in place exactly as `approxPolyDP_` does.
    let count = dst.len();
    let mut new_count = count;
    let f = |p: (i32, i32)| (p.0 as f64, p.1 as f64);
    let mut pos = count - 1;
    let read = |pos: &mut usize, dst: &[(i32, i32)]| {
        let p = dst[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };
    let mut start_pt = read(&mut pos, &dst);
    let mut wpos = pos;
    let mut cur = read(&mut pos, &dst);
    let mut i = 0;
    while i < count && new_count > 2 {
        let end_pt = read(&mut pos, &dst);
        let (sx, sy) = f(start_pt);
        let (px, py) = f(cur);
        let (ex, ey) = f(end_pt);
        let (dx, dy) = (ex - sx, ey - sy);
        let dist = ((px - sx) * dy - (py - sy) * dx).abs();
        let successive_inner_product = (px - sx) * (ex - px) + (py - sy) * (ey - py);
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy)
            && dx != 0.0
            && dy != 0.0
            && successive_inner_product >= 0.0
        {
            new_count -= 1;
            start_pt = end_pt;
            dst[wpos] = end_pt;
            wpos = (wpos + 1) % count;
            cur = read(&mut pos, &dst);
            i += 2;
            continue;
        }
        start_pt = cur;
        dst[wpos] = cur;
        wpos = (wpos + 1) % count;
        cur = end_pt;
        i += 1;
    }
    dst.truncate(new_count);
    dst
}

/// OpenCV's `isContourConvex` on integer points: every turn has the same,
/// non-zero orientation.
pub(super) fn is_convex(p: &[(i32, i32)]) -> bool {
    let n = p.len();
    if n < 3 {
        return false;
    }
    let mut prev = p[n - 2];
    let mut cur = p[n - 1];
    let (mut dx0, mut dy0) = (cur.0 as i64 - prev.0 as i64, cur.1 as i64 - prev.1 as i64);
    let mut orientation = 0u8;
    for &q in p {
        prev = cur;
        cur = q;
        let (dx, dy) = (cur.0 as i64 - prev.0 as i64, cur.1 as i64 - prev.1 as i64);
        let (dxdy0, dydx0) = (dx * dy0, dy * dx0);
        orientation |= match dydx0.cmp(&dxdy0) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => 2,
            std::cmp::Ordering::Equal => 3,
        };
        if orientation == 3 {
            return false;
        }
        (dx0, dy0) = (dx, dy);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(rows: &[&str]) -> (Vec<u8>, usize, usize) {
        let h = rows.len();
        let w = rows[0].len();
        let data = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| u8::from(b == b'#')))
            .collect();
        (data, w, h)
    }

    /// A filled square gives one outer border, traced through every border
    /// pixel once, and no hole.
    #[test]
    fn filled_square_has_one_border() {
        let (img, w, h) = image(&[
            "......", //
            ".####.", //
            ".####.", //
            ".####.", //
            ".####.", //
            "......",
        ]);
        let c = find_contours_list(&img, w, h);
        assert_eq!(c.len(), 1);
        // OpenCV: 12 points from (1, 1), going down first.
        assert_eq!(c[0].len(), 12);
        assert_eq!(c[0][..3], [(1, 1), (1, 2), (1, 3)]);
    }

    /// A one-pixel ring: the outer border and the hole border are both
    /// reported, although the outer trace already visited every pixel of the
    /// hole border (which `cv_imgproc::find_contours` does not handle). The
    /// hole border skips the ring's corners, which touch the hole only
    /// diagonally. Lengths and start points as OpenCV's.
    #[test]
    fn thin_ring_has_outer_and_hole_border() {
        let (img, w, h) = image(&[
            ".......", //
            ".#####.", //
            ".#...#.", //
            ".#...#.", //
            ".#...#.", //
            ".#####.", //
            ".......",
        ]);
        let c = find_contours_list(&img, w, h);
        assert_eq!(c.len(), 2, "{c:?}");
        assert_eq!((c[0].len(), c[0][0]), (16, (1, 1)));
        assert_eq!((c[1].len(), c[1][0]), (12, (1, 2)));
    }

    /// The image frame counts as background, like OpenCV.
    #[test]
    fn frame_is_background() {
        let (img, w, h) = image(&["####", "####", "####", "####"]);
        let c = find_contours_list(&img, w, h);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 4);
    }

    /// The approximation of a rotated square's border finds its four
    /// corners, whichever border pixel the trace started on.
    #[test]
    fn approx_finds_the_four_corners_of_a_tilted_square() {
        let (w, h) = (60usize, 60usize);
        let corners = [(30.0, 8.0), (52.0, 30.0), (30.0, 52.0), (8.0, 30.0)];
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64, y as f64);
                let inside = (0..4).all(|k| {
                    let (ax, ay) = corners[k];
                    let (bx, by) = corners[(k + 1) % 4];
                    (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0.0
                });
                img[y * w + x] = u8::from(inside);
            }
        }
        let c = find_contours_list(&img, w, h);
        assert_eq!(c.len(), 1);
        let approx = approx_poly_dp_closed(&c[0], c[0].len() as f64 * 0.03);
        assert_eq!(approx.len(), 4, "{approx:?}");
        assert!(is_convex(&approx));
        for (cx, cy) in corners {
            assert!(
                approx
                    .iter()
                    .any(|&(x, y)| (x as f64 - cx).hypot(y as f64 - cy) <= 1.5),
                "corner ({cx}, {cy}) not in {approx:?}"
            );
        }
    }

    #[test]
    fn convexity() {
        assert!(is_convex(&[(0, 0), (10, 0), (10, 10), (0, 10)]));
        assert!(is_convex(&[(0, 0), (0, 10), (10, 10), (10, 0)]));
        assert!(!is_convex(&[(0, 0), (10, 0), (3, 3), (0, 10)]));
        assert!(!is_convex(&[(0, 0), (5, 0), (10, 0), (0, 10)]));
    }
}
