//! `convex_hull` must not panic, and must not invent vertices, on non-finite input.
//!
//! The point sort used `partial_cmp(..).unwrap()`, and `partial_cmp` returns
//! `None` for NaN — so a single NaN coordinate **panicked** the whole call:
//!
//! ```text
//! finite        -> ok, 7 vertices
//! one NaN x     -> *** PANICKED ***
//! a NaN point   -> *** PANICKED ***
//! ```
//!
//! Note the asymmetry with the sentinel-accumulation sites fixed elsewhere: those
//! were unreachable because Rust's `f32::min`/`max` return the *non-NaN* operand
//! (`f32::MAX.min(f64::NAN) == f32::MAX`), so a NaN coordinate was swallowed by
//! the running bounds. Here the NaN went straight into a comparison.
//!
//! Simply swapping in `total_cmp` stops the panic but returns a *wrong* hull: a
//! NaN has no position, so the cross-product test compares against NaN and every
//! such comparison is false. Measured, that gave nine vertices from eight points,
//! where the finite input gives seven. So non-finite points are now dropped -
//! the hull is computed over the points that actually have a position.

use cv_geometry2d::geometry2d::{convex_hull, Point2D};

fn point(x: f64, y: f64) -> Point2D {
    Point2D { x, y }
}

/// A square plus points inside it: the hull is the square, four vertices.
fn square_with_interior() -> Vec<Point2D> {
    vec![
        point(0.0, 0.0),
        point(10.0, 0.0),
        point(10.0, 10.0),
        point(0.0, 10.0),
        point(3.0, 3.0),
        point(7.0, 7.0),
    ]
}

#[test]
fn a_finite_cloud_gives_the_expected_hull() {
    let hull = convex_hull(&square_with_interior());
    // The four square corners. `exterior` is closed, so the first point repeats.
    let distinct = {
        let mut v = hull.exterior.clone();
        if v.len() > 1 && v[0] == v[v.len() - 1] {
            v.pop();
        }
        v
    };
    assert_eq!(
        distinct.len(),
        4,
        "control: the hull of a square with two interior points is the square, \
         got {:?}",
        hull.exterior
    );
}

#[test]
fn a_nan_coordinate_does_not_panic() {
    let mut pts = square_with_interior();
    pts[2].x = f64::NAN;
    // Before the fix this panicked inside `partial_cmp().unwrap()`.
    let hull = convex_hull(&pts);
    assert!(
        hull.exterior
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite()),
        "the hull must not contain a non-finite vertex: {:?}",
        hull.exterior
    );
}

#[test]
fn a_non_finite_point_does_not_change_the_hull_of_the_rest() {
    let good = square_with_interior();

    let mut with_nan = good.clone();
    with_nan.push(point(f64::NAN, 5.0));
    assert_eq!(
        convex_hull(&with_nan).exterior,
        convex_hull(&good).exterior,
        "a NaN point has no position, so it must not be inside or outside the \
         hull; including it made the cross-product test compare against NaN, \\
         where every comparison is false"
    );

    let mut with_inf = good.clone();
    with_inf.push(point(f64::INFINITY, 5.0));
    assert_eq!(
        convex_hull(&with_inf).exterior,
        convex_hull(&good).exterior,
        "an infinite point must be excluded the same way"
    );
}

/// A NaN point must not be silently turned into a vertex of its own.
#[test]
fn a_non_finite_point_never_becomes_a_vertex() {
    let pts = vec![
        point(0.0, 0.0),
        point(10.0, 0.0),
        point(10.0, 10.0),
        point(0.0, 10.0),
        point(f64::NAN, f64::NAN),
    ];
    let hull = convex_hull(&pts);
    assert!(
        !hull
            .exterior
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite()),
        "hull contains a non-finite vertex: {:?}",
        hull.exterior
    );
}

/// The empty and single-point cases, which sort an empty slice and so never
/// reach the comparison at all.
#[test]
fn degenerate_inputs_are_handled() {
    assert!(convex_hull(&[]).exterior.is_empty());
    let one = convex_hull(&[point(1.0, 2.0)]);
    assert!(one.exterior.len() <= 2, "a single point cannot form a hull");
}
