//! Polygon approximation of floating-point curves (OpenCV's `approxPolyDP`).

use nalgebra::Point2;

/// OpenCV's `approxPolyDP(curve, epsilon, closed)`: Ramer-Douglas-Peucker
/// polygon simplification, returning a subset of the curve's points.
///
/// Same algorithm as [`crate::contours::approx_poly_dp`] (see there): a
/// closed curve is split at two far-apart points found from point 0, not at
/// point 0 itself, deviations are measured to the chord segment, and
/// OpenCV's final clean-up pass is applied. The arithmetic is `f64`;
/// OpenCV's `float` path takes coordinate differences in `f32`, so on
/// coordinates whose differences are not exact in `f32` a near-tie in
/// distance may resolve differently.
pub fn approx_poly_dp(curve: &[Point2<f64>], epsilon: f64, closed: bool) -> Vec<Point2<f64>> {
    crate::contours::approx_poly_dp_indices(
        curve.len(),
        |i| (curve[i].x, curve[i].y),
        epsilon,
        closed,
    )
    .into_iter()
    .map(|i| curve[i])
    .collect()
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

    #[test]
    fn test_approx_poly_dp_keeps_vertices_beyond_epsilon() {
        // Regression: `douglas_peucker` started from an all-false mask and only
        // ever cleared entries, so a vertex selected as a split point was never
        // marked kept. Every curve with 3+ points therefore collapsed to its
        // two endpoints no matter how large the deviation was.
        let spike = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 5.0), // 5 px off the closing chord, epsilon 0.5
            Point2::new(2.0, 0.0),
        ];
        let open = approx_poly_dp(&spike, 0.5, false);
        assert_eq!(
            open.len(),
            3,
            "the spike vertex exceeds epsilon and must be kept: {:?}",
            open
        );
        assert!(open.contains(&Point2::new(1.0, 5.0)));

        // Closed squares: a dense outline must simplify to its four corners,
        // not to two opposite corners joined by a diagonal.
        let mut square = Vec::new();
        for i in 0..10 {
            square.push(Point2::new(i as f64, 0.0));
        }
        for i in 1..10 {
            square.push(Point2::new(9.0, i as f64));
        }
        for i in (0..9).rev() {
            square.push(Point2::new(i as f64, 9.0));
        }
        for i in (1..9).rev() {
            square.push(Point2::new(0.0, i as f64));
        }
        let closed = approx_poly_dp(&square, 0.5, true);
        assert_eq!(
            closed.len(),
            4,
            "a closed square must keep its four corners: {:?}",
            closed
        );
    }
}
