//! `voronoi_diagram` must not depend on the units of the coordinates.
//!
//! `clip_by_halfplane` compared the half-plane dot product - a length²
//! quantity, `n · (v - mid)` - against the absolute `EPS = 1e-10`. On two sites
//! one micron apart inside a micron-sized box that product is ~1e-12, i.e.
//! below the cutoff for *every* vertex, so every vertex read as "inside", no
//! cell was ever clipped, and each of the two cells came back as the whole
//! bounding box:
//!
//! ```text
//! scale   sum of cell areas / box area
//! 1       1.0
//! 1e-2    1.0
//! 1e-4    1.0
//! 1e-6    2.0   (both cells are the whole box - they overlap completely)
//! ```
//!
//! The zero band is now relative to `|n| |v - mid|`, the natural scale of the
//! dot product, exactly as `segment_intersection_point` scales its parallel
//! cutoff. At unit scale the old threshold is unchanged.

use cv_geometry2d::geometry2d::*;

fn p(x: f64, y: f64) -> Point2D {
    Point2D::new(x, y)
}

/// Two sites at (0,0) and (s,0) in the box `[-s,2s] x [-s,s]`. The bisector is
/// x = s/2, so each cell is exactly half the box.
fn two_site_cells(s: f64) -> (Polygon, Polygon, f64) {
    let pts = vec![p(0.0, 0.0), p(s, 0.0)];
    let bounds = (-s, -s, 2.0 * s, s);
    let cells = voronoi_diagram(&pts, bounds);
    assert_eq!(cells.len(), 2, "one cell per site at scale {s}");
    let box_area = (bounds.2 - bounds.0) * (bounds.3 - bounds.1);
    (cells[0].clone(), cells[1].clone(), box_area)
}

#[test]
fn cells_of_two_close_sites_partition_the_bounding_box() {
    for &s in &[1.0, 1e-3, 1e-6] {
        let (a, b, box_area) = two_site_cells(s);
        let total = a.area() + b.area();
        assert!(
            ((total - box_area) / box_area).abs() < 1e-9,
            "scale {s}: the cells cover {total:e} but the box is {box_area:e} \
             (with an absolute cutoff no vertex is ever clipped, so each cell \
             is the whole box)"
        );
        assert!(
            ((a.area() - box_area / 2.0) / box_area).abs() < 1e-9,
            "scale {s}: cell 0 should be half the box, got {:e}",
            a.area()
        );
        assert!(
            ((a.bbox().2 - s / 2.0) / s).abs() < 1e-9,
            "scale {s}: cell 0 must stop at the bisector x = {}, got max_x = {}",
            s / 2.0,
            a.bbox().2
        );
        assert!(
            ((b.bbox().0 - s / 2.0) / s).abs() < 1e-9,
            "scale {s}: cell 1 must start at the bisector, got min_x = {}",
            b.bbox().0
        );

        // A point plainly on site 0's side belongs to cell 0 only.
        let near_0 = p(-s / 4.0, 0.0);
        assert!(
            point_in_polygon(&near_0, &a),
            "scale {s}: cell 0 owns {near_0}"
        );
        assert!(
            !point_in_polygon(&near_0, &b),
            "scale {s}: cell 1 must not overlap cell 0"
        );
    }
}

#[test]
fn unit_scale_three_site_cells_stay_disjoint_and_cover_the_box() {
    // Control: three sites at unit scale tile the box with no overlap.
    let pts = vec![p(0.0, 0.0), p(10.0, 0.0), p(5.0, 10.0)];
    let bounds = (-10.0, -10.0, 20.0, 20.0);
    let box_area = (bounds.2 - bounds.0) * (bounds.3 - bounds.1);
    let cells = voronoi_diagram(&pts, bounds);
    assert_eq!(cells.len(), 3);
    let total: f64 = cells.iter().map(|c| c.area()).sum();
    assert!(
        ((total - box_area) / box_area).abs() < 1e-9,
        "covered {total}, box {box_area}"
    );
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            let shared: f64 = polygon_intersection(&cells[i], &cells[j])
                .iter()
                .map(|c| c.area())
                .sum();
            assert!(
                shared / box_area < 1e-9,
                "cells {i} and {j} overlap by {shared:e}"
            );
        }
    }
}
