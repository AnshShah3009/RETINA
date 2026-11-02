//! `buffer_polygon` against shapely (GEOS 3.x, shapely 2.1.2), `quad_segs`
//! equal to `segments`.
//!
//! Three defects, each visible in one case below:
//!
//! * The raw offset ring was returned as the polygon. Where the offsets of a
//!   concavity cross - the inside of a U whose gap is narrower than twice the
//!   distance - that ring folds over itself: area 17.574 instead of 17.249 at
//!   distance 0.6, 25.705 instead of 24.092 at 1.0, and `is_valid` false.
//! * Convex corners were found with `cross > EPS` on a length^2 quantity, so
//!   below ~1e-5 every corner got a miter instead of a round join: a square
//!   of side 1e-6 buffered by half its side measured 3.5 s^2, not 3.784 s^2.
//! * Holes were dropped (and so was any negative-distance concave corner's
//!   rounding).

use cv_geometry2d::geometry2d::*;

fn ring(v: &[(f64, f64)]) -> Vec<Point2D> {
    let mut pts: Vec<Point2D> = v.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    pts.push(pts[0].clone());
    pts
}

fn poly(v: &[(f64, f64)]) -> Polygon {
    Polygon::new(ring(v), vec![])
}

fn u_shape() -> Polygon {
    poly(&[
        (0., 0.),
        (3., 0.),
        (3., 3.),
        (2., 3.),
        (2., 1.),
        (1., 1.),
        (1., 3.),
        (0., 3.),
    ])
}

fn assert_rel(got: f64, want: f64, what: &str) {
    assert!(
        ((got - want) / want).abs() < 1e-7,
        "{what}: area {got}, shapely {want}"
    );
}

#[test]
fn the_offset_of_a_concavity_does_not_fold_over_itself() {
    for (d, shapely) in [
        (0.1, 8.627048227358193),
        (0.4, 13.832771637731025),
        (0.6, 17.249060899502588),
        (1.0, 24.092330272151937),
    ] {
        let b = buffer_polygon(&u_shape(), d, 16);
        assert_rel(b.area(), shapely, &format!("U buffered by {d}"));
        assert!(b.is_valid(), "U buffered by {d} is not a valid polygon");
    }
}

#[test]
fn round_joins_do_not_depend_on_the_units() {
    for s in [1.0, 1e-3, 1e-6] {
        let sq = poly(&[(0., 0.), (s, 0.), (s, s), (0., s)]);
        let b = buffer_polygon(&sq, 0.5 * s, 16);
        assert_rel(
            b.area() / (s * s),
            3.784137122636485,
            &format!("square of side {s}, in units of s^2"),
        );
    }
}

#[test]
fn non_right_corners_use_quad_segs_per_quarter_turn() {
    let tri = poly(&[(0., 0.), (4., 0.), (1., 3.)]);
    assert_rel(
        buffer_polygon(&tri, 0.7, 8).area(),
        15.512926908198901,
        "triangle",
    );
}

#[test]
fn holes_shrink_and_close() {
    let holed = Polygon::new(
        ring(&[(0., 0.), (10., 0.), (10., 10.), (0., 10.)]),
        vec![ring(&[(4., 4.), (6., 4.), (6., 6.), (4., 6.)])],
    );
    let b = buffer_polygon(&holed, 0.5, 16);
    assert_eq!(b.holes.len(), 1, "a 2x2 hole buffered by 0.5 is a 1x1 hole");
    assert_rel(b.area(), 119.78413712263644, "square with hole, d = 0.5");

    let b = buffer_polygon(&holed, 1.5, 16);
    assert!(b.holes.is_empty(), "the hole closes at d = 1.5");
    assert_rel(b.area(), 167.05723410372843, "square with hole, d = 1.5");
}

#[test]
fn negative_buffers_round_reflex_corners_and_split() {
    assert_rel(
        buffer_polygon(&u_shape(), -0.3, 16).area(),
        2.5988553179254343,
        "U eroded by 0.3",
    );

    // Two 2x2 squares joined by a 0.2-wide bar: eroding by 0.2 removes the
    // bar and leaves two pieces.
    let dumbbell = poly(&[
        (0., 0.),
        (2., 0.),
        (2., 0.9),
        (3., 0.9),
        (3., 0.),
        (5., 0.),
        (5., 2.),
        (3., 2.),
        (3., 1.1),
        (2., 1.1),
        (2., 2.),
        (0., 2.),
    ]);
    let parts = buffer_polygon_parts(&dumbbell, -0.2, 16);
    assert_eq!(parts.len(), 2);
    for p in &parts {
        assert_rel(p.area(), 2.5617687287357604, "dumbbell half");
    }

    let unit = poly(&[(0., 0.), (1., 0.), (1., 1.), (0., 1.)]);
    assert_eq!(buffer_polygon(&unit, -0.6, 16).area(), 0.0);
}
