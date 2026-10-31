//! `delaunay_triangulation` must not depend on the units of the coordinates.
//!
//! The in-circle determinant (units of length⁴) was compared against an
//! absolute `EPS = 1e-10`, and hull edges were closed by a finite `20 x`
//! super-triangle. On the 40 points of `common` (hull area 0.80260 · scale²):
//!
//! ```text
//! scale   triangles   covered area / scale²
//! 1       70          0.80260   (mixed clockwise / counter-clockwise output)
//! 1e-3    6           1.09285   (overlapping triangles)
//! 1e-5    0           0
//! 1e-6    0           0
//! ```
//!
//! (The review's own 40-point set also lost hull triangles at unit scale to the
//! super-triangle vertices: covered 0.733651 of a 0.734003 hull.)
//!
//! Now: Shewchuk's adaptive exact `incircle` / `orient2d` (the `robust` crate
//! `geo` already uses) and ghost triangles through a vertex at infinity, so
//! the result tiles the convex hull exactly. SciPy reference (qhull): 70
//! simplices = 2n - h - 2 at scales 1, 1e-3, 1e-6 and 1e6.

mod common;

use common::*;
use cv_geometry2d::geometry2d::*;

fn tri_area(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0
}

#[test]
fn delaunay_matches_scipy_at_every_scale() {
    for &s in &SCALES {
        let pts = scaled(s);
        let tris = delaunay_triangulation(&pts);
        // Euler: 2n - h - 2 triangles for n points with h hull vertices.
        let expected = 2 * pts.len() - SCIPY_HULL.len() - 2;
        assert_eq!(expected, SCIPY_SIMPLICES);
        assert_eq!(tris.len(), SCIPY_SIMPLICES, "scale {s}: triangle count");

        let mut area = 0.0;
        for t in &tris {
            let a = tri_area(&pts[t[0]], &pts[t[1]], &pts[t[2]]);
            assert!(a > 0.0, "scale {s}: triangle {t:?} not counter-clockwise");
            area += a;
        }
        let hull_area = SCIPY_HULL_AREA_UNIT * s * s;
        assert!(
            ((area - hull_area) / hull_area).abs() < 1e-12,
            "scale {s}: covered area {area} vs hull area {hull_area}"
        );

        // Empty-circle property, checked in scale-free form: the in-circle
        // determinant relative to its magnitude bound.
        for t in &tris {
            let (a, b, c) = (&pts[t[0]], &pts[t[1]], &pts[t[2]]);
            for (i, d) in pts.iter().enumerate() {
                if t.contains(&i) {
                    continue;
                }
                let (adx, ady) = (a.x - d.x, a.y - d.y);
                let (bdx, bdy) = (b.x - d.x, b.y - d.y);
                let (cdx, cdy) = (c.x - d.x, c.y - d.y);
                let al = adx * adx + ady * ady;
                let bl = bdx * bdx + bdy * bdy;
                let cl = cdx * cdx + cdy * cdy;
                let det = adx * (bdy * cl - bl * cdy) - ady * (bdx * cl - bl * cdx)
                    + al * (bdx * cdy - bdy * cdx);
                let bound = (adx.abs() + ady.abs())
                    * (bdx.abs() + bdy.abs())
                    * (cdx.abs() + cdy.abs())
                    * (al + bl + cl);
                assert!(
                    det <= 1e-12 * bound,
                    "scale {s}: point {i} inside circumcircle of {t:?}"
                );
            }
        }
    }
}

#[test]
fn delaunay_is_translation_invariant() {
    // Far from the origin the absolute coordinates are ~1e6 but the spacing is
    // unit-sized; the predicates must see the spacing, not the offset.
    let pts: Vec<Point2D> = PTS
        .iter()
        .map(|p| Point2D::new(p[0] + 1e6, p[1] - 3e6))
        .collect();
    let tris = delaunay_triangulation(&pts);
    assert_eq!(tris.len(), SCIPY_SIMPLICES);
}

#[test]
fn delaunay_skips_duplicates_and_collinear_input() {
    let line: Vec<Point2D> = (0..5)
        .map(|i| Point2D::new(i as f64, 2.0 * i as f64))
        .collect();
    assert!(delaunay_triangulation(&line).is_empty());

    let mut pts = scaled(1.0);
    pts.push(pts[5].clone());
    pts.push(pts[17].clone());
    let tris = delaunay_triangulation(&pts);
    assert_eq!(tris.len(), SCIPY_SIMPLICES, "duplicates add no triangles");
    assert!(tris.iter().all(|t| t.iter().all(|&i| i < 40)));
}
