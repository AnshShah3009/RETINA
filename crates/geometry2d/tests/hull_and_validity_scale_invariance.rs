//! `convex_hull`, `Polygon::is_valid` and the Douglas-Peucker distance must not
//! depend on the units of the coordinates.
//!
//! All three compared a length² quantity against an absolute `EPS = 1e-10`:
//! the hull of the 40 points of `common` scaled by 1e-5 or 1e-6 came back as the
//! two vertices `[3, 8]` (SciPy: `[3, 8, 9, 11, 16, 18, 34, 38]`), a square of side
//! 1e-6 was "not valid", and every chord shorter than 1e-5 measured distance
//! to its start point. Orientation signs now come from Shewchuk's adaptive
//! exact `orient2d` (the `robust` crate), and the zero-length test is exact.

mod common;

use common::*;
use cv_geometry2d::geometry2d::*;

fn hull_indices(pts: &[Point2D]) -> Vec<usize> {
    let hull = convex_hull(pts);
    let mut ring = hull.exterior.clone();
    if ring.len() > 1 && ring[0] == ring[ring.len() - 1] {
        ring.pop();
    }
    let mut idx: Vec<usize> = ring
        .iter()
        .map(|h| {
            pts.iter()
                .position(|p| p == h)
                .expect("hull vertex is an input point")
        })
        .collect();
    idx.sort_unstable();
    idx
}

#[test]
fn convex_hull_matches_scipy_at_every_scale() {
    for &s in &SCALES {
        let pts = scaled(s);
        assert_eq!(hull_indices(&pts), SCIPY_HULL.to_vec(), "scale {s}");
        let hull = convex_hull(&pts);
        let area = hull.area();
        let expected = SCIPY_HULL_AREA_UNIT * s * s;
        assert!(
            ((area - expected) / expected).abs() < 1e-12,
            "scale {s}: area {area}"
        );
        assert!(hull.is_valid(), "scale {s}: hull must be a valid polygon");
    }
}

#[test]
fn convex_hull_keeps_a_nearly_flat_but_not_flat_vertex() {
    // (0.5, 1e-12) is above the chord by 1e-12: qhull keeps it (hull area
    // 5e-13), an absolute-EPS test on the cross product dropped it.
    let pts = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.5, 1e-12),
    ];
    assert_eq!(hull_indices(&pts), vec![0, 1, 2]);
}

#[test]
fn is_valid_is_scale_free() {
    for &s in &[1.0, 1e-3, 1e-6, 1e-9, 1e6] {
        let sq = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(s, 0.0),
                Point2D::new(s, s),
                Point2D::new(0.0, s),
                Point2D::new(0.0, 0.0),
            ],
            vec![],
        );
        assert!(sq.is_valid(), "square of side {s} is valid");
        let flat = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(s, 0.0),
                Point2D::new(2.0 * s, 0.0),
                Point2D::new(3.0 * s, 0.0),
            ],
            vec![],
        );
        assert!(!flat.is_valid(), "collinear ring of scale {s} is not valid");
        let bowtie = Polygon::new(
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(s, s),
                Point2D::new(s, 0.0),
                Point2D::new(0.0, s),
            ],
            vec![],
        );
        assert!(!bowtie.is_valid(), "bow-tie of scale {s} self-intersects");
    }
}

#[test]
fn simplify_uses_the_perpendicular_distance_on_short_chords() {
    // Chord of length 1e-6; the middle point is 0.5e-6 off the chord and
    // 0.707e-6 from its start. With tolerance 0.6e-6 it must be dropped
    // (shapely: `LineString(...).simplify(6e-7)` keeps two points). The old
    // code fell back to the point-to-endpoint distance whenever the squared
    // chord length was below 1e-10, and kept it.
    let line = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(0.5e-6, 0.5e-6),
        Point2D::new(1e-6, 0.0),
    ];
    assert_eq!(simplify(&line, 0.6e-6).len(), 2);
    assert_eq!(simplify(&line, 0.4e-6).len(), 3);
}
