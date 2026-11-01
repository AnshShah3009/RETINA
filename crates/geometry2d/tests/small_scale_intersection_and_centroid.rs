//! Scale regressions for segment intersection and polygon centroid.
//!
//! An absolute `EPS = 1e-10` on a length² cross product treated a real crossing
//! of segments a millionth of a unit long as parallel, and the same cutoff on
//! shoelace area replaced the centroid of a closed 1e-5 triangle with a vertex
//! average that counted the repeated closing vertex.

use cv_geometry2d::geometry2d::*;

fn p(x: f64, y: f64) -> Point2D {
    Point2D::new(x, y)
}

#[test]
fn tiny_crossing_segments_return_the_intersection_point() {
    // Same shape as the unit-scale case below, scaled by 1e-6.
    // Cross product is 2e-12, under the old absolute 1e-10 parallel cutoff,
    // so the point function returned None while `segments_intersect` was true.
    let a1 = p(0.0, 0.0);
    let a2 = p(1e-6, 0.0);
    let b1 = p(0.5e-6, -1e-6);
    let b2 = p(0.5e-6, 1e-6);
    assert!(segments_intersect(&a1, &a2, &b1, &b2), "the segments cross");
    let hit = segment_intersection_point(&a1, &a2, &b1, &b2)
        .expect("crossing segments must have an intersection point");
    assert!(
        (hit.x - 0.5e-6).abs() < 1e-18,
        "x: got {}, want {}",
        hit.x,
        0.5e-6
    );
    assert!(hit.y.abs() < 1e-18, "y: got {}, want 0", hit.y);
}

#[test]
fn unit_scale_crossing_is_unchanged() {
    let hit = segment_intersection_point(&p(0.0, 0.0), &p(1.0, 0.0), &p(0.5, -1.0), &p(0.5, 1.0))
        .expect("unit-scale cross");
    assert_eq!(hit.x, 0.5);
    assert_eq!(hit.y, 0.0);
}

#[test]
fn parallel_segments_still_have_no_point() {
    assert!(
        segment_intersection_point(&p(0.0, 0.0), &p(1.0, 0.0), &p(0.0, 1.0), &p(2.0, 1.0))
            .is_none()
    );
    assert!(
        segment_intersection_point(&p(0.0, 0.0), &p(1e-6, 0.0), &p(0.0, 1e-6), &p(1e-6, 1e-6),)
            .is_none()
    );
}

fn right_triangle(leg: f64, close: bool) -> Polygon {
    let mut ring = vec![p(0.0, 0.0), p(leg, 0.0), p(0.0, leg)];
    if close {
        ring.push(p(0.0, 0.0));
    }
    Polygon::new(ring, vec![])
}

#[test]
fn closed_tiny_right_triangle_centroid_is_the_vertex_mean() {
    // Area is 5e-11, under the old 1e-10 cutoff, so the centroid fell back to
    // averaging four vertices (the closing point repeated) and landed at
    // (leg/4, leg/4) = (2.5e-6, 2.5e-6) instead of (leg/3, leg/3).
    let leg = 1e-5;
    let c = right_triangle(leg, true).centroid();
    let expect = leg / 3.0;
    assert!(
        (c.x - expect).abs() < 1e-15,
        "x: got {}, want {}",
        c.x,
        expect
    );
    assert!(
        (c.y - expect).abs() < 1e-15,
        "y: got {}, want {}",
        c.y,
        expect
    );
}

#[test]
fn unit_right_triangle_centroid_stays_exact() {
    let c = right_triangle(1.0, true).centroid();
    assert!((c.x - 1.0 / 3.0).abs() < 1e-15);
    assert!((c.y - 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn zero_area_closed_ring_does_not_average_the_closing_vertex_twice() {
    // Collinear, explicitly closed. Averaging the repeated (0, 0) gives
    // (0.75, 0); the three distinct vertices average to (1, 0).
    let poly = Polygon::new(
        vec![p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0), p(0.0, 0.0)],
        vec![],
    );
    let c = poly.centroid();
    assert!((c.x - 1.0).abs() < 1e-15, "x: got {}", c.x);
    assert!(c.y.abs() < 1e-15, "y: got {}", c.y);
}
