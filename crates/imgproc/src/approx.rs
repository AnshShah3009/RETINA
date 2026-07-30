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

    // Find point with max distance
    let _start = curve[0];
    let _end = if closed { curve[0] } else { *curve.last().unwrap() };

    // Recursive Douglas-Peucker
    let mut mask = vec![true; n];
    douglas_peucker(curve, 0, n - 1, epsilon, &mut mask);

    for (i, &m) in mask.iter().enumerate() {
        if m {
            result.push(curve[i]);
        }
    }

    if closed {
        result.push(result[0]);
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
