//! `Polygon::is_valid` rejects rings that touch themselves, as GEOS does.
//!
//! Expectations are shapely 2.1.2 `Polygon(ring).is_valid` for each ring.
//! The self-intersection test counted only *proper* crossings, so every
//! `false` case below except the bow-tie came back valid.

use cv_geometry2d::geometry2d::*;

fn poly(v: &[(f64, f64)]) -> Polygon {
    let mut pts: Vec<Point2D> = v.iter().map(|&(x, y)| Point2D::new(x, y)).collect();
    pts.push(pts[0].clone());
    Polygon::new(pts, vec![])
}

#[test]
fn self_touching_rings_are_invalid() {
    let cases: [(&str, &[(f64, f64)], bool); 8] = [
        // Two lobes meeting at the vertex (1, 1), visited twice.
        (
            "figure eight",
            &[(0., 0.), (2., 0.), (1., 1.), (2., 2.), (0., 2.), (1., 1.)],
            false,
        ),
        // The last vertex lies on the first edge.
        (
            "vertex on edge",
            &[(0., 0.), (2., 0.), (2., 2.), (1., 0.)],
            false,
        ),
        // A spike that runs out along y = 0 and back.
        (
            "spike",
            &[(0., 0.), (2., 0.), (3., 0.), (2., 0.), (2., 2.), (0., 2.)],
            false,
        ),
        // A notch whose bottom runs back along the first edge.
        (
            "collinear overlap",
            &[
                (0., 0.),
                (3., 0.),
                (3., 1.),
                (2., 1.),
                (2., 0.),
                (1., 0.),
                (1., -1.),
                (0., -1.),
            ],
            false,
        ),
        // Controls.
        ("bow-tie", &[(0., 0.), (2., 2.), (2., 0.), (0., 2.)], false),
        (
            "repeated vertex",
            &[(0., 0.), (1., 0.), (1., 0.), (1., 1.), (0., 1.)],
            true,
        ),
        (
            "L",
            &[(0., 0.), (2., 0.), (2., 1.), (1., 1.), (1., 2.), (0., 2.)],
            true,
        ),
        (
            "deep concavity",
            &[(0., 0.), (4., 0.), (4., 4.), (2., 1.), (0., 4.)],
            true,
        ),
    ];
    for (name, ring, shapely) in cases {
        assert_eq!(poly(ring).is_valid(), shapely, "{name}");
    }
}
