//! A degenerate convex hull must be reported, not treated as "everything is
//! visible".
//!
//! `crates/3d/src/hidden_point_removal.rs:302`
//! ```ignore
//! let tet = match find_initial_tetrahedron(points) {
//!     Ok(t) => t,
//!     Err(_) => { for v in on_hull.iter_mut() { *v = true; } return on_hull; }
//! };
//! ```
//!
//! When the point set has no initial tetrahedron — collinear, coplanar or
//! coincident — the incremental hull builder gave up and marked **every** point
//! as on the hull, then returned `Ok`. The caller cannot tell that from a real
//! answer.
//!
//! Measured before the fix: 11 collinear points on the z-axis with the eye at
//! `(0, 0, 20)` returned **11/11 visible**. The correct answer is 1 — the
//! nearest point — with the other 10 exactly occluded by it.
//!
//! A hull algorithm cannot answer at all when the input is degenerate (an
//! incremental 3D hull has no faces to start from), so inventing "everything is
//! visible" is not a conservative choice, it is a confident wrong answer. The
//! degeneracy now propagates out as an `Err`, which is already the shape of
//! the public API (`Result<HprResult, String>`) and already handled by the
//! Python binding, which maps it to `PyRuntimeError`.
//!
//! (The planar case is arguably defensible in principle — a plane is its own
//! silhouette — but the same code path served collinear and coincident input
//! too, where no such argument exists, so all degenerate configurations are now
//! reported uniformly rather than silently answered.)

use cv_3d::hidden_point_removal::{hidden_point_removal, select_visible_points};
use nalgebra::Point3;
use std::f64::consts::PI;

fn sphere_points(n: usize) -> Vec<Point3<f64>> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * (i as f64) / golden;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            Point3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
        })
        .collect()
}

fn eye() -> Point3<f64> {
    Point3::new(0.0, 0.0, 20.0)
}

// ── Defect 4a: collinear input ───────────────────────────────────────────────

/// 11 collinear points on the z-axis. Only the nearest one is visible; the other
/// 10 are exactly occluded by it.
#[test]
fn collinear_points_are_not_all_reported_visible() {
    let points: Vec<Point3<f64>> = (0..11).map(|i| Point3::new(0.0, 0.0, i as f64)).collect();

    let result = hidden_point_removal(&points, &eye(), 0.0);

    match result {
        Err(msg) => {
            assert!(
                !msg.is_empty(),
                "a degeneracy must be explained, not signalled with an empty message"
            );
        }
        Ok(res) => panic!(
            "collinear input returned Ok with {}/{} points visible. Every point of a \
             collinear run behind the nearest one is exactly occluded by it, so at most \
             one point can be visible; this input previously returned 11/11. Degeneracy \
             must be reported, not answered with 'everything is visible'.",
            res.visible_indices.len(),
            points.len()
        ),
    }
}

/// The `select_visible_points` convenience wrapper must propagate the same
/// degeneracy rather than returning every point as "visible".
#[test]
fn select_visible_points_also_reports_collinear_degeneracy() {
    let points: Vec<Point3<f64>> = (0..11).map(|i| Point3::new(0.0, 0.0, i as f64)).collect();

    match select_visible_points(&points, &eye(), 0.0) {
        Err(_) => {}
        Ok(v) => panic!(
            "select_visible_points returned {}/{} points as visible for collinear \
             input; at most the single nearest point is visible",
            v.len(),
            points.len()
        ),
    }
}

// ── Defect 4a: coincident input ─────────────────────────────────────────────

/// Every point at the same location. There is no hull, no extent and no
/// meaningful answer.
#[test]
fn coincident_points_are_reported_as_degenerate() {
    let points = vec![Point3::new(0.0, 0.0, 0.0); 11];

    let result = hidden_point_removal(&points, &eye(), 0.0);

    assert!(
        result.is_err(),
        "11 coincident points reported Ok with {:?} visible: no convex hull exists, so \
         there is no visibility answer to give",
        result.map(|r| r.visible_indices.len())
    );
}

/// The same cloud translated along z — still coincident, so still degenerate.
/// Guards against a degeneracy test that only happens to pass for one position.
#[test]
fn coincident_points_at_another_location_are_also_degenerate() {
    let points = vec![Point3::new(1.0, 2.0, 3.0); 8];

    assert!(
        hidden_point_removal(&points, &eye(), 0.0).is_err(),
        "coincident points must be reported as degenerate wherever they sit"
    );
}

// ── Controls: non-degenerate inputs must keep working ───────────────────────

/// CONTROL: a full 3D sphere cloud still answers, still finds most of the near
/// hemisphere, and still rejects nearly all of the far hemisphere. A fix that
/// returned `Err` for everything would pass the tests above and fail here.
#[test]
fn a_sphere_cloud_still_answers_normally() {
    let points = sphere_points(200);
    let eye = Point3::new(0.0, 0.0, 5.0);

    let result = hidden_point_removal(&points, &eye, 0.0)
        .expect("CONTROL: a full 3D sphere cloud is not degenerate and must answer");

    assert_eq!(
        result.visibility_mask.len(),
        points.len(),
        "CONTROL: the mask must stay aligned with the input"
    );

    let front_total = points.iter().filter(|p| p.z > 0.2).count();
    let front_visible = result
        .visible_indices
        .iter()
        .filter(|&&i| points[i].z > 0.2)
        .count();
    assert!(
        front_visible as f64 / front_total as f64 > 0.7,
        "CONTROL: most near-hemisphere points of a sphere must be visible, got \
         {front_visible}/{front_total}"
    );

    let back_total = points.iter().filter(|p| p.z < -0.2).count();
    let back_visible = result
        .visible_indices
        .iter()
        .filter(|&&i| points[i].z < -0.2)
        .count();
    assert!(
        (back_visible as f64) < back_total as f64 * 0.3,
        "CONTROL: most far-hemisphere points must be hidden, got \
         {back_visible}/{back_total}"
    );
}

/// CONTROL: a dense cube is not degenerate either, and the three faces nearest
/// the eye must come back visible while the three opposite faces stay hidden.
#[test]
fn a_cube_cloud_still_answers_normally() {
    let mut points = Vec::new();
    let steps = 5;
    for i in 0..steps {
        for j in 0..steps {
            let u = -0.5 + (i as f64 + 0.5) / steps as f64;
            let v = -0.5 + (j as f64 + 0.5) / steps as f64;
            points.push(Point3::new(0.5, u, v));
            points.push(Point3::new(-0.5, u, v));
            points.push(Point3::new(u, 0.5, v));
            points.push(Point3::new(u, -0.5, v));
            points.push(Point3::new(u, v, 0.5));
            points.push(Point3::new(u, v, -0.5));
        }
    }

    let eye = Point3::new(5.0, 5.0, 5.0);
    let result = hidden_point_removal(&points, &eye, 0.0)
        .expect("CONTROL: a full 3D cube cloud is not degenerate and must answer");

    let visible: std::collections::HashSet<usize> =
        result.visible_indices.iter().copied().collect();

    let per_face = steps * steps;
    for (label, want_visible) in [
        ("+X", true),
        ("+Y", true),
        ("+Z", true),
        ("-X", false),
        ("-Y", false),
        ("-Z", false),
    ] {
        // Face points are interleaved in blocks of six per (i, j) cell, in the
        // order +X, -X, +Y, -Y, +Z, -Z.
        let mut count = 0usize;
        for cell in 0..per_face {
            let offset = match label {
                "+X" => 0,
                "-X" => 1,
                "+Y" => 2,
                "-Y" => 3,
                "+Z" => 4,
                _ => 5,
            };
            if visible.contains(&(cell * 6 + offset)) {
                count += 1;
            }
        }
        if want_visible {
            assert!(
                count as f64 / per_face as f64 > 0.5,
                "CONTROL: face {label} faces the eye and must be mostly visible, got \
                 {count}/{per_face}"
            );
        } else {
            assert!(
                (count as f64) < per_face as f64 * 0.5,
                "CONTROL: face {label} points away and must be mostly hidden, got \
                 {count}/{per_face}"
            );
        }
    }
}

/// CONTROL: too few points has always been an error and must stay one.
#[test]
fn too_few_points_is_still_an_error() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    assert!(hidden_point_removal(&points, &eye(), 0.0).is_err());
}

/// CONTROL: a cloud that is nearly, but not exactly, collinear (one point nudged
/// off the line by 1e-3) is a valid — if thin — 3D hull and must still answer.
/// This is what keeps the degeneracy check from being a blanket rejection of
/// thin clouds.
#[test]
fn a_thin_but_genuine_3d_cloud_still_answers() {
    let mut points: Vec<Point3<f64>> = (0..12).map(|i| Point3::new(0.0, 0.0, i as f64)).collect();
    points[3] = Point3::new(1e-3, 0.0, 3.0);
    points[7] = Point3::new(0.0, -1e-3, 7.0);

    let result = hidden_point_removal(&points, &eye(), 0.0)
        .expect("CONTROL: a cloud with genuine 3D extent is not degenerate");

    assert!(
        !result.visible_indices.is_empty(),
        "CONTROL: a non-degenerate cloud must report at least one visible point"
    );
    assert!(
        result.visibility_mask.len() == points.len(),
        "CONTROL: a non-degenerate cloud must still produce a mask aligned with the \
         input, got {} entries for {} points",
        result.visibility_mask.len(),
        points.len()
    );
    // The hull of a sliver this thin contains every one of its points - the flat
    // sides ARE its silhouette - so the meaningful control is only that the
    // answer is produced at all, not that points are culled.
}
