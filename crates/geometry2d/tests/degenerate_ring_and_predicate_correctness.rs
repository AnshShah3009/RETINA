//! Regression tests for the degenerate-input defects fixed in `geometry2d`.
//!
//! Five defects, each pinned with the number it produced before the fix and a
//! control that keeps the well-formed case working:
//!
//! 1. `polygons_intersect` dropped the wrap-around edge of an **open** ring, so
//!    two polygons touching only across that edge were reported disjoint.
//! 2. `polygon_contains_polygon` returned `true` for an **empty** `inner`: with
//!    the vertex loop vacuous, "contains nothing" read as "contains".
//! 3. `Polygon::is_valid` counted vertices instead of checking that the ring
//!    bounds a region: a ring of four collinear vertices was "valid", and a
//!    genuine triangle without its closing vertex was "invalid".
//! 4. `simplify` with a **negative tolerance** recursed until the stack
//!    overflowed - an abort, not even a catchable panic.
//! 5. `from_wkt` accepted `nan` / `inf` / `1e400`, producing a polygon whose
//!    area is `NaN` and whose bounding box silently omits the bad vertex.
//!
//! Where an external reference exists the expectation was cross-checked with
//! GEOS via `shapely` 2.1.2 rather than read back off this crate.

use cv_geometry2d::geometry2d::*;

fn p(x: f64, y: f64) -> Point2D {
    Point2D::new(x, y)
}

/// A ring with its closing vertex repeated - the form `Polygon`'s own docs ask
/// for ("should be closed: first == last").
fn closed(v: &[(f64, f64)]) -> Polygon {
    let mut pts: Vec<Point2D> = v.iter().map(|&(x, y)| p(x, y)).collect();
    pts.push(pts[0].clone());
    Polygon::new(pts, vec![])
}

// ── 1. polygons_intersect and the wrap-around edge of an open ring ──────────

#[test]
fn polygons_touching_across_an_open_rings_closing_edge_intersect() {
    // A: the unit square written *without* its repeated closing vertex, so the
    // only edge missing from the old sweep was (0,4)-(0,0) - the left edge.
    // `a.area()` is 16, i.e. the crate does treat this ring as closed.
    let open_square = Polygon::new(
        vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)],
        vec![],
    );
    assert!(
        (open_square.area() - 16.0).abs() < 1e-12,
        "precondition: the open ring must have the area of the closed square, got {}",
        open_square.area()
    );

    // A triangle whose whole contact with A is that omitted left edge: it spans
    // y = 1..3 and reaches exactly x = 0. Before the fix the edge sweep saw
    // nothing and the containment fallback (one vertex per polygon) said no:
    // `false`, while the closed forms said `true`.
    let touching = Polygon::new(vec![p(-1.0, 1.0), p(-1.0, 3.0), p(0.0, 2.0)], vec![]);
    assert!(
        polygons_intersect(&open_square, &touching),
        "a triangle meeting the square exactly across the ring's wrap-around \
         edge still intersects it (GEOS: True)"
    );
    // Symmetric order.
    assert!(polygons_intersect(&touching, &open_square));

    // Control: the same pair in closed form, plus the ordinary overlap, edge
    // sharing and far-apart cases, must be unaffected.
    let closed_square = closed(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
    assert!(polygons_intersect(&closed_square, &touching));
    let overlapping = closed(&[(3.0, 3.0), (6.0, 3.0), (6.0, 6.0), (3.0, 6.0)]);
    assert!(polygons_intersect(&closed_square, &overlapping), "overlap");
    let far = closed(&[
        (100.0, 100.0),
        (101.0, 100.0),
        (101.0, 101.0),
        (100.0, 101.0),
    ]);
    assert!(
        !polygons_intersect(&open_square, &far),
        "an open ring must not become a false positive for a distant polygon"
    );
}

#[test]
fn a_hole_ring_is_part_of_the_intersection_sweep() {
    // A second, independent way the old edge sweep lost contact: it iterated
    // `a.exterior` and `b.exterior` only, so a polygon's *holes* were never
    // compared against anything.
    //
    // The configuration is built so that nothing else can rescue it:
    //
    // * `big` is (0,0)-(10,10) with a horizontal slot for a hole, so every edge
    //   of `small` lies strictly inside `big`'s exterior extent - the two
    //   exteriors neither cross nor touch.
    // * `small` straddles the slot's upper edge y = 8 with its edge x = 3,
    //   which crosses the slot edge at (3, 8).
    // * `small.exterior[0]` is (1, 6.5), which lies *inside the slot* - a point
    //   in `big`'s exterior that is in `big`'s hole - so the one-vertex
    //   containment fallback reports `point_in_polygon((1, 6.5), big)` as
    //   `false` and cannot fire.
    // * `big.exterior[0]` is (0, 0), far outside `small`.
    //
    // So the containment fallbacks both say no, and the exterior edge sweep has
    // nothing to find: the only contact is between `small` and `big`'s hole.
    let small = closed(&[(1.0, 6.5), (3.0, 6.5), (3.0, 9.5), (1.0, 9.5)]);
    let slot_closed = |exterior: Vec<Point2D>| -> Polygon {
        Polygon::new(
            exterior,
            vec![closed(&[(-2.0, 6.0), (12.0, 6.0), (12.0, 8.0), (-2.0, 8.0)]).exterior],
        )
    };
    let big_closed =
        slot_closed(closed(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]).exterior);
    assert!(
        !point_in_polygon(&small.exterior[0], &big_closed),
        "precondition: small.exterior[0] = (1, 6.5) lies inside the slot, so it \
         is inside big's exterior but not inside big, and the containment \
         fallback cannot fire"
    );
    let big_open = Polygon::new(
        closed(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]).exterior,
        vec![vec![p(-2.0, 6.0), p(12.0, 6.0), p(12.0, 8.0), p(-2.0, 8.0)]],
    );

    for (name, big) in [("closed hole", &big_closed), ("open hole", &big_open)] {
        // Both argument orders, because the sweep pairs `a`'s rings against
        // `b`'s: a hole that is only consulted on one side would still be a
        // defect.
        assert!(
            polygons_intersect(&small, big),
            "{name}: small crosses the slot's edges y = 6 and y = 8"
        );
        assert!(
            polygons_intersect(big, &small),
            "{name}: the same pair in the other argument order"
        );
    }

    // Control: move `small` clear of the slot. The two polygons then share all
    // of `small`'s area (2 x 1 = 2), so they still intersect - and the
    // containment fallback agrees, which is exactly why the failing case above
    // had to be built to defeat it.
    let clear_of_slot = closed(&[(1.0, 8.5), (3.0, 8.5), (3.0, 9.5), (1.0, 9.5)]);
    assert!(
        polygons_intersect(&clear_of_slot, &big_closed),
        "still shared"
    );
    assert!(
        polygons_intersect(&clear_of_slot, &big_open),
        "still shared"
    );
    let shared: f64 = polygon_intersection(&clear_of_slot, &big_closed)
        .iter()
        .map(|p| p.area())
        .sum();
    assert!(
        (shared - 2.0).abs() < 1e-9,
        "shared area {shared}, expected 2.0"
    );
}

#[test]
fn a_hole_that_breaks_out_of_its_polygon_is_an_intersection() {
    // The omitted-edge defect reached through a hole ring written without its
    // closing vertex: the wrap-around edge is the only contact.
    let square = closed(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
    // An open hole ring pushed out through the left edge: (-1,1)-(2,3)-(0,3).
    // Its closing edge - (0,3) back to (-1,1) - is exactly the segment that
    // meets the square's left edge x = 0. Every other edge of the hole ring
    // lies outside the square, and the one `outer` vertex that falls inside the
    // hole, (0,0), is excluded by `point_in_ring`'s half-open edge rule, so
    // neither the sweep nor the containment fallback finds the contact.
    let open_hole = vec![p(-1.0, 1.0), p(2.0, 1.0), p(2.0, 3.0), p(0.0, 3.0)];
    let square_with_hole = Polygon::new(
        square.exterior.clone(),
        vec![Polygon::new(open_hole, vec![]).exterior],
    );
    assert!(
        polygons_intersect(&square, &square_with_hole),
        "a hole ring protruding past the exterior boundary overlaps it"
    );

    // The closed spelling of the same polygon was wrong too, its closing edge
    // being x = -1 and the hole rings being absent from the sweep entirely.
    let closed_hole = Polygon::new(
        square.exterior.clone(),
        vec![closed(&[(-1.0, 1.0), (2.0, 1.0), (2.0, 3.0), (-1.0, 3.0)]).exterior],
    );
    assert!(polygons_intersect(&square, &closed_hole), "closed spelling");

    // Control: a hole entirely inside still overlaps the plain square, and the
    // overlap is exactly the holed polygon's own area - 16 - 1 = 15, not the
    // full 16 (GEOS agrees on both the intersection and its area).
    let clear = Polygon::new(
        square.exterior.clone(),
        vec![closed(&[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]).exterior],
    );
    assert!(
        polygons_intersect(&square, &clear),
        "the hole's area is shared"
    );
    let shared: f64 = polygon_intersection(&square, &clear)
        .iter()
        .map(|p| p.area())
        .sum();
    assert!(
        (shared - 15.0).abs() < 1e-9,
        "shared area {shared}, expected 15.0",
    );

    // Control: genuinely disjoint polygons stay disjoint, open or closed.
    let far = closed(&[(50.0, 50.0), (54.0, 50.0), (54.0, 54.0), (50.0, 54.0)]);
    assert!(!polygons_intersect(&square, &far));
    let far_open = Polygon::new(
        vec![p(50.0, 50.0), p(54.0, 50.0), p(54.0, 54.0), p(50.0, 54.0)],
        vec![],
    );
    assert!(!polygons_intersect(&square, &far_open));
}

// ── 2. containment of an empty polygon ─────────────────────────────────────

#[test]
fn an_empty_polygon_is_contained_by_nothing() {
    let big = closed(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
    let empty = Polygon::new(vec![], vec![]);

    assert!(
        !polygon_contains_polygon(&big, &empty),
        "before the fix this was `true`: the vertex loop was vacuous and nothing \
         else could reject it, so every polygon was reported to contain the empty \
         ring - and the empty ring contained itself"
    );
    assert!(!polygon_contains_polygon(&empty, &empty));
    assert!(
        !polygon_contains_polygon(&empty, &big),
        "an empty polygon contains nothing either"
    );

    // Control: a real polygon is still contained.
    let inside = closed(&[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]);
    assert!(polygon_contains_polygon(&big, &inside));
    // A collapsed two-vertex ring is not a polygon (`is_valid` is false, its
    // area is 0) but every one of its points lies inside `big`, so it is
    // contained in the "covered by" sense - GEOS reports `True` as well. The
    // distinction that matters here is *bounded* versus *unbounded* input: a
    // ring with nothing in it is no conclusion, a degenerate one is a zero-area
    // one.
    let collapsed = Polygon::new(vec![p(1.0, 1.0), p(2.0, 2.0), p(1.0, 1.0)], vec![]);
    assert!(
        !collapsed.is_valid(),
        "precondition: a collapsed ring is not a valid polygon"
    );
    assert!(polygon_contains_polygon(&big, &collapsed));
}

// ── 3. Polygon::is_valid ───────────────────────────────────────────────────

#[test]
fn a_ring_of_collinear_vertices_is_not_a_valid_polygon() {
    // Four collinear vertices plus the closing vertex: five positions, so the
    // old `len() < 4` count let it through as `valid` even though it encloses
    // no area at all (its `area()` is 0.0). GEOS agrees:
    // `Polygon([(0,0),(1,0),(2,0),(3,0)]).is_valid` is `False`.
    let flat = closed(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    assert_eq!(flat.area(), 0.0, "precondition: the ring encloses nothing");
    assert!(
        !flat.is_valid(),
        "a ring of collinear vertices bounds no region and must not be valid"
    );

    // Control: a real square is still valid, and so is the bowtie's opposite.
    assert!(closed(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]).is_valid());
}

#[test]
fn validity_does_not_depend_on_whether_the_ring_repeats_its_first_vertex() {
    // The docs say "at least 3 *distinct* vertices". The old check demanded
    // `len() >= 4`, so the very same triangle was reported invalid when it was
    // written without the repeated closing vertex - a form `area` and
    // `perimeter` both accept and report the full area for.
    let open_triangle = Polygon::new(vec![p(0.0, 0.0), p(4.0, 0.0), p(0.0, 3.0)], vec![]);
    let closed_triangle = closed(&[(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]);

    assert!(
        (open_triangle.area() - closed_triangle.area()).abs() < 1e-12,
        "precondition: both spellings of the same ring measure the same area"
    );
    assert!(
        open_triangle.is_valid(),
        "three distinct vertices with area {open_triangle_area} is a triangle; \
         requiring a fourth position rejected the un-closed spelling",
        open_triangle_area = open_triangle.area()
    );
    assert!(closed_triangle.is_valid());

    // Control: genuinely degenerate rings are still rejected, and the
    // self-intersecting bowtie (which the old check caught) is still caught.
    assert!(!Polygon::new(vec![], vec![]).is_valid());
    assert!(!Polygon::new(vec![p(0.0, 0.0), p(1.0, 1.0)], vec![]).is_valid());
    let bowtie = closed(&[(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)]);
    assert!(!bowtie.is_valid(), "self-intersecting ring");
}

// ── 4. simplify with a negative tolerance ──────────────────────────────────

#[test]
fn simplify_with_a_negative_tolerance_terminates() {
    // Before the fix this aborted the process:
    //   thread '...' has overflowed its stack
    //   fatal runtime error: stack overflow, aborting
    // which `catch_unwind` cannot intercept. The recursion starts at index 1,
    // yet with a negative tolerance even an entirely collinear input satisfies
    // `max_dist > tolerance` at the initial index 0, so the "split" returned
    // the whole slice and the next level recursed on it forever.
    let coords: Vec<Point2D> = (0..6).map(|i| p(i as f64, i as f64 * 0.5)).collect();
    let result = simplify(&coords, -1.0);
    assert!(
        result.len() >= 2,
        "a negative tolerance must still return the two endpoints, got {} points",
        result.len()
    );
    assert_eq!(result[0], coords[0]);
    assert_eq!(result[result.len() - 1], coords[coords.len() - 1]);

    // Control: the non-negative path is unchanged, including the boundary.
    let flat: Vec<Point2D> = (0..5).map(|i| p(i as f64, 0.0)).collect();
    assert_eq!(simplify(&flat, 0.0), vec![flat[0].clone(), flat[4].clone()]);
    assert_eq!(
        simplify(&flat, -1.0),
        simplify(&flat, 0.0),
        "clamping a negative tolerance must land on the zero-tolerance result"
    );
    // Control: a deviation strictly smaller than the tolerance is *not* an
    // exact test, which is what distinguishes clamping at 0 from folding a
    // negative tolerance to its magnitude: `(-1).abs() == 1.0`, so an `abs()`
    // clamp would drop this point while a clamp at 0 keeps it (its distance
    // from the chord is 1e-9, which is > 0 but far below 1.0).
    let epsilon_dev = vec![p(0.0, 0.0), p(1.0, 1e-9), p(2.0, 0.0)];
    assert_eq!(
        simplify(&epsilon_dev, 0.0).len(),
        3,
        "at zero tolerance every deviation at all is significant"
    );
    assert_eq!(
        simplify(&epsilon_dev, 1.0).len(),
        2,
        "a tolerance above the deviation collapses to the endpoints"
    );
    assert_eq!(
        simplify(&epsilon_dev, -1.0),
        simplify(&epsilon_dev, 0.0),
        "a negative tolerance must behave exactly like zero, not like its magnitude"
    );
    assert_eq!(
        simplify(&epsilon_dev, -1.0).len(),
        3,
        "control for the assertion above: an `abs()` clamp would report 2 here"
    );

    let bent = vec![p(0.0, 0.0), p(1.0, 5.0), p(2.0, 0.0)];
    assert_eq!(simplify(&bent, 0.1).len(), 3, "a real deviation is kept");
    assert_eq!(simplify(&bent, 0.1), simplify(&bent, -0.1));
}

// ── 5. WKT with a non-finite coordinate ────────────────────────────────────

#[test]
fn wkt_with_a_non_finite_coordinate_is_rejected() {
    // `f64::from_str` accepts `nan`, `inf`, `infinity` and overflows `1e400`
    // to `+inf`. All four parsed `Ok` before the fix, and the result was a
    // polygon whose `area()` is `NaN` and whose `bbox()` quietly dropped the
    // offending vertex: `POLYGON((nan 0, 1 0, 1 1, 0 1, 0 0))` reported
    // `area() = NaN, bbox() = (0, 0, 1, 1)`.
    for (name, wkt) in [
        ("nan", "POLYGON((nan 0, 1 0, 1 1, 0 1, 0 0))"),
        ("NaN", "POLYGON((0 NaN, 1 0, 1 1, 0 1, 0 0))"),
        ("inf", "POLYGON((inf 0, 1 0, 1 1, 0 1, 0 0))"),
        ("-infinity", "POLYGON((-infinity 0, 1 0, 1 1, 0 1, 0 0))"),
        ("1e400 overflow", "POLYGON((1e400 0, 1 0, 1 1, 0 1, 0 0))"),
        ("-1e400 overflow", "POLYGON((-1e400 0, 1 0, 1 1, 0 1, 0 0))"),
        (
            "non-finite in a hole",
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 1, 3 nan, 1 3, 1 1))",
        ),
    ] {
        let parsed = from_wkt(wkt);
        assert!(
            parsed.is_err(),
            "{name}: a coordinate with no position must be rejected, got {:?}",
            parsed.as_ref().ok().map(|p| (p.area(), p.bbox()))
        );
    }

    // Control: ordinary rings, holes and negative-but-finite coordinates all
    // parse exactly as before.
    let ok = from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))").expect("unit square");
    assert!((ok.area() - 1.0).abs() < 1e-12);
    let negatives = from_wkt("POLYGON((-1 -1, 1 -1, 1 1, -1 1, -1 -1))").expect("negative box");
    assert!(
        (negatives.area() - 4.0).abs() < 1e-12,
        "{}",
        negatives.area()
    );
    let holed =
        from_wkt("POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1,3 1,3 3,1 3,1 1))").expect("ring with a hole");
    assert_eq!(holed.holes.len(), 1);
    assert!((holed.area() - 12.0).abs() < 1e-12, "{}", holed.area());
}
