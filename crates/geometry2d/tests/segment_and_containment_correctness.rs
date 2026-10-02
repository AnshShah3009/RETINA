//! Correctness regressions for `segments_intersect`, `polygon_contains_polygon`,
//! ring perimeter and the STR-tree packing.
//!
//! Every case below was cross-checked against GEOS via `shapely` (`LineString
//! .intersects`, `Polygon.contains`) so the expectations are external, not
//! reverse-engineered from this crate.

use cv_geometry2d::geometry2d::*;

fn p(x: f64, y: f64) -> Point2D {
    Point2D::new(x, y)
}

fn ring(v: &[(f64, f64)]) -> Polygon {
    let mut pts: Vec<Point2D> = v.iter().map(|&(x, y)| p(x, y)).collect();
    if let (Some(f), Some(l)) = (pts.first().cloned(), pts.last().cloned()) {
        if f.x != l.x || f.y != l.y {
            pts.push(f);
        }
    }
    Polygon::new(pts, vec![])
}

// ── Segment intersection ─────────────────────────────────────────────────────

#[test]
fn touching_segments_are_reported_as_intersecting() {
    // The doc contract is "intersect (including touching / collinear)".
    // Verified with shapely: all of these are `True`.
    let cases: [(f64, f64, f64, f64, f64, f64, f64, f64); 4] = [
        // shared endpoint
        (0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0),
        // T-touch: endpoint of the vertical segment lies on the horizontal one
        (5.0, 0.0, 5.0, 1.0, 0.0, 0.0, 10.0, 0.0),
        // interior T-touch
        (-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        // collinear overlap
        (0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0),
    ];
    for (a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y) in cases {
        assert!(
            segments_intersect(&p(a1x, a1y), &p(a2x, a2y), &p(b1x, b1y), &p(b2x, b2y)),
            "segments ({a1x},{a1y})-({a2x},{a2y}) and ({b1x},{b1y})-({b2x},{b2y}) \
             share a point, so they must intersect"
        );
    }
}

#[test]
fn disjoint_segments_are_not_reported_as_intersecting() {
    // shapely: all `False`.
    let cases: [(&str, (f64, f64, f64, f64, f64, f64, f64, f64)); 3] = [
        (
            // a1 is collinear with b, and b1 inside a's bounding box, but the
            // segments are far apart: this is the false positive the swapped
            // `on_segment` arguments produced.
            "collinear endpoint, far apart",
            (0.0, 0.0, 6.0, 3.0, 5.0, 0.0, 10.0, 0.0),
        ),
        ("parallel", (0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0)),
        (
            "collinear, disjoint",
            (0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0),
        ),
    ];
    for (name, (a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y)) in cases {
        assert!(
            !segments_intersect(&p(a1x, a1y), &p(a2x, a2y), &p(b1x, b1y), &p(b2x, b2y)),
            "{name}: segments ({a1x},{a1y})-({a2x},{a2y}) and ({b1x},{b1y})-({b2x},{b2y}) \
             are disjoint"
        );
    }
}

#[test]
fn properly_crossing_segments_still_intersect() {
    // Control for the two tests above: the fix must not reject everything.
    assert!(segments_intersect(
        &p(0.0, 0.0),
        &p(2.0, 2.0),
        &p(0.0, 2.0),
        &p(2.0, 0.0)
    ));
}

// ── polygons_intersect ───────────────────────────────────────────────────────

#[test]
fn disjoint_polygons_are_not_reported_as_intersecting() {
    // Thin triangle (area 3) and a square (area 5) that sit above/below each
    // other: shapely `Polygon.intersects` -> False. The triangle's vertex (0,0)
    // is collinear with the square's bottom edge y=0, which is exactly the
    // configuration the swapped `on_segment` arguments turned into a hit.
    let a = ring(&[(0.0, 0.0), (6.0, 3.0), (6.0, 2.0)]);
    let b = ring(&[(5.0, 0.0), (10.0, 0.0), (10.0, 1.0), (5.0, 1.0)]);
    assert!((a.area() - 3.0).abs() < 1e-9);
    assert!((b.area() - 5.0).abs() < 1e-9);
    assert!(
        !polygons_intersect(&a, &b),
        "disjoint polygons reported as intersecting: a.bbox={:?} b.bbox={:?}",
        a.bbox(),
        b.bbox()
    );
    // and the boolean op agrees
    assert!(polygon_intersection(&a, &b).is_empty());
}

#[test]
fn overlapping_and_touching_polygons_still_intersect() {
    // Controls for the test above.
    let a = ring(&[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]);
    let overlapping = ring(&[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]);
    let edge_touching = ring(&[(2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0)]);
    let far = ring(&[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)]);
    assert!(polygons_intersect(&a, &overlapping), "overlap");
    assert!(polygons_intersect(&a, &edge_touching), "shared edge");
    assert!(!polygons_intersect(&a, &far), "far apart");
}

// ── polygon_contains_polygon ─────────────────────────────────────────────────

#[test]
fn containment_is_not_concluded_from_vertices_alone() {
    // 10x10 square with a 4x4 hole; the "inner" ring spans (1,1)-(9,9), so all
    // four of its vertices are inside `outer` and outside the hole, but it
    // covers the hole. shapely: outer.contains(inner) -> False.
    let outer = Polygon::new(
        ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]).exterior,
        vec![ring(&[(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]).exterior],
    );
    let covers_hole = ring(&[(1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (1.0, 9.0)]);
    assert!(
        !polygon_contains_polygon(&outer, &covers_hole),
        "a ring that covers the hole of `outer` is not contained in it"
    );

    // Control: a ring that really is inside must still be reported contained.
    let inside = ring(&[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]);
    assert!(polygon_contains_polygon(&outer, &inside));
}

#[test]
fn containment_allows_boundary_touching() {
    // C-shaped ring; the small square [1,3]x[1,3] touches the C's re-entrant
    // corner at (3,3). shapely: contains -> True, covers -> True.
    let c_shape = ring(&[
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 3.0),
        (3.0, 3.0),
        (3.0, 7.0),
        (10.0, 7.0),
        (10.0, 10.0),
        (0.0, 10.0),
    ]);
    let touching = ring(&[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]);
    let interior = ring(&[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]);
    let upper_arm = ring(&[(4.0, 7.5), (6.0, 7.5), (6.0, 9.5), (4.0, 9.5)]);
    assert!(polygon_contains_polygon(&c_shape, &touching));
    assert!(polygon_contains_polygon(&c_shape, &interior));
    assert!(polygon_contains_polygon(&c_shape, &upper_arm));
    // Control: the converse must stay false, and a ring that straddles the
    // mouth of the C (edge crossing the concavity) is not contained.
    assert!(!polygon_contains_polygon(&touching, &c_shape));
    let straddling = ring(&[(2.0, 2.0), (8.0, 8.0), (2.0, 8.0)]);
    assert!(
        !polygon_contains_polygon(&c_shape, &straddling),
        "a triangle reaching across the C's mouth is not contained"
    );
}

// ── Ring perimeter ───────────────────────────────────────────────────────────

#[test]
fn perimeter_of_an_open_ring_includes_its_closing_edge() {
    let open = Polygon::new(
        vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)],
        vec![],
    );
    let closed = ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    assert!((open.area() - closed.area()).abs() < 1e-12);
    assert!(
        (open.perimeter() - 4.0).abs() < 1e-12,
        "unit square perimeter, got {}",
        open.perimeter()
    );
    assert!((open.perimeter() - closed.perimeter()).abs() < 1e-12);

    // Control: a 3-4-5 triangle, both windings, and a hole contribution.
    let tri = Polygon::new(vec![p(0.0, 0.0), p(3.0, 0.0), p(0.0, 4.0)], vec![]);
    assert!(
        (tri.perimeter() - 12.0).abs() < 1e-12,
        "{}",
        tri.perimeter()
    );
    let holed = Polygon::new(
        closed.exterior.clone(),
        vec![vec![p(0.2, 0.2), p(0.2, 0.8), p(0.8, 0.8), p(0.8, 0.2)]],
    );
    assert!((holed.perimeter() - (4.0 + 0.6 * 4.0)).abs() < 1e-12);
}

// ── STR-tree ─────────────────────────────────────────────────────────────────

#[test]
fn strtree_tolerates_non_finite_bounding_boxes() {
    // More than the node capacity (8) so the y-centre sort inside `str_build`
    // runs; a NaN ordinate makes that centre NaN and the sort used to panic
    // inside `partial_cmp(..).unwrap()`.
    let mut items: Vec<(usize, f64, f64, f64, f64)> = (0..12)
        .map(|i| (i, i as f64, 0.0, i as f64 + 1.0, 1.0))
        .collect();
    items.push((99, 4.0, f64::NAN, 6.0, 2.0));

    let tree = STRtree::new(&items);
    // The finite items are still queryable.
    let hits = tree.query(3.0, 0.0, 5.0, 1.0);
    assert!(hits.contains(&3) && hits.contains(&4), "hits={hits:?}");
}
