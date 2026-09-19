//! Contour approximation algorithms — ported from OpenCV 5.x
//!
//! Implements:
//! - `approxPolyDP`: Douglas-Peucker polygon simplification
//! - `approxPolyN`: N-point polygon approximation

use nalgebra::Point2;

/// Douglas-Peucker polygon simplification
///
/// Approximates a curve with a polygon with fewer vertices.
/// The epsilon parameter controls the approximation accuracy
/// (larger = fewer vertices, more approximation).
pub fn approx_poly_dp(curve: &[Point2<f64>], epsilon: f64, closed: bool) -> Vec<Point2<f64>> {
    if curve.len() < 3 {
        return curve.to_vec();
    }

    let n = curve.len();
    let mut result = Vec::new();

    if !closed {
        // Open curve: anchor the simplification at both endpoints.
        let mut mask = vec![false; n];
        mask[0] = true;
        mask[n - 1] = true;
        douglas_peucker(curve, 0, n - 1, epsilon, &mut mask);
        for (i, &m) in mask.iter().enumerate() {
            if m {
                result.push(curve[i]);
            }
        }
        return result;
    }

    // Closed curve: split at the two farthest-apart contour points and
    // simplify each arc against its chord (OpenCV semantics). Simplifying
    // [0..n-1] as an open chain never measured deviation across the
    // closing segment near the seam.
    let mut far_pair = (0usize, 0usize);
    let mut far_dist: f64 = -1.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = curve[i].x - curve[j].x;
            let dy = curve[i].y - curve[j].y;
            let d2 = dx * dx + dy * dy;
            if d2 > far_dist {
                far_dist = d2;
                far_pair = (i, j);
            }
        }
    }
    let (i0, i1) = far_pair;

    let mut mask = vec![false; n];
    mask[i0] = true;
    mask[i1] = true;
    if i0 < i1 {
        douglas_peucker(curve, i0, i1, epsilon, &mut mask);
        // Wrap-around arc: j..n plus 0..i — handled via a rotated copy.
        let mut arc: Vec<Point2<f64>> = Vec::with_capacity(i0 + n - i1 + 1);
        arc.extend_from_slice(&curve[i1..n]);
        arc.extend_from_slice(&curve[..=i0]);
        let mut arc_mask = vec![false; arc.len()];
        arc_mask[0] = true;
        arc_mask[arc.len() - 1] = true;
        douglas_peucker(&arc, 0, arc.len() - 1, epsilon, &mut arc_mask);
        for (k, &m) in arc_mask.iter().enumerate() {
            if m {
                let global = (i1 + k) % n;
                mask[global] = true;
            }
        }
    } else {
        // Degenerate duplicate extremes; fall back to open-chain behavior.
        douglas_peucker(curve, 0, n - 1, epsilon, &mut mask);
    }

    for (i, &m) in mask.iter().enumerate() {
        if m {
            result.push(curve[i]);
        }
    }

    result
}

fn douglas_peucker(
    curve: &[Point2<f64>],
    start: usize,
    end: usize,
    epsilon: f64,
    mask: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }

    let mut dmax = 0.0;
    let mut idx = start;

    for i in (start + 1)..end {
        let d = perpendicular_distance(&curve[i], &curve[start], &curve[end]);
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }

    if dmax > epsilon {
        douglas_peucker(curve, start, idx, epsilon, mask);
        douglas_peucker(curve, idx, end, epsilon, mask);
    } else {
        for i in (start + 1)..end {
            mask[i] = false;
        }
    }
}

/// Perpendicular distance from point to line segment
fn perpendicular_distance(pt: &Point2<f64>, line_start: &Point2<f64>, line_end: &Point2<f64>) -> f64 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 {
        return (pt.x - line_start.x).hypot(pt.y - line_start.y);
    }
    let num = ((line_end.y - line_start.y) * pt.x
        - (line_end.x - line_start.x) * pt.y
        + line_end.x * line_start.y
        - line_end.y * line_start.x)
        .abs();
    num / len_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_poly_dp_simple() {
        let curve = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.1),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        ];
        let approx = approx_poly_dp(&curve, 0.5, false);
        assert!(approx.len() <= curve.len());
        assert_eq!(approx[0], curve[0]);
        assert_eq!(*approx.last().unwrap(), *curve.last().unwrap());
    }
}
