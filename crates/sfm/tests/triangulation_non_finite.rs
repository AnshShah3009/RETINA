//! `cv_sfm` triangulation must reject non-finite input instead of hanging.
//!
//! `.svd(true, true)` does not return on non-finite input: nalgebra's
//! bidiagonalisation decides convergence by *comparison*, and every comparison
//! against NaN is false. `triangulate_point_dlt` assembles a 4x4 system and
//! decomposes it that way, so one NaN pixel hung it indefinitely.
//!
//! Measured on the unfixed code: finite input returned
//! `[1.032, 0.780, 2.000]` (correct — the test geometry places the point at
//! depth 2), and one NaN pixel had to be killed at 120 s.
//!
//! There is no timeout guard here. If a regression reintroduces the hang, the
//! test *times out* rather than fails, which is the honest signal for this class.

use nalgebra::{Matrix3x4, Point2};

/// Camera 1 at the origin, camera 2 at (0.5, 0, 0), both identity rotation, so
/// `P = K [R | t]` with the translation in the 4th column scaled by `fx`.
fn projections() -> (Matrix3x4<f64>, Matrix3x4<f64>) {
    let (fx, fy) = (620.0, 615.0);
    let p1 = Matrix3x4::new(
        fx, 0.0, 0.0, 0.0, //
        0.0, fy, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );
    let p2 = Matrix3x4::new(
        fx,
        0.0,
        0.0,
        fx * 0.5, //
        0.0,
        fy,
        0.0,
        0.0, //
        0.0,
        0.0,
        1.0,
        0.0,
    );
    (p1, p2)
}

#[test]
fn triangulate_point_dlt_rejects_a_non_finite_pixel() {
    let (p1, p2) = projections();
    let a1 = Point2::new(320.0, 240.0);
    let a2 = Point2::new(475.0, 240.0);

    // Control: finite input triangulates, at the depth the geometry implies.
    let pt = cv_sfm::triangulation::triangulate_point_dlt(&a1, &a2, &p1, &p2)
        .expect("finite input must still triangulate");
    assert!(
        (pt.z - 2.0).abs() < 1e-6,
        "control: expected depth 2.0, got {pt:?}"
    );

    let bad = Point2::new(f64::NAN, 240.0);
    let err = cv_sfm::triangulation::triangulate_point_dlt(&bad, &a2, &p1, &p2)
        .expect_err("a NaN pixel must be reported, not decomposed");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn triangulate_point_dlt_rejects_an_infinite_pixel() {
    let (p1, p2) = projections();
    let a1 = Point2::new(f64::INFINITY, 240.0);
    let a2 = Point2::new(475.0, 240.0);
    let err = cv_sfm::triangulation::triangulate_point_dlt(&a1, &a2, &p1, &p2)
        .expect_err("an infinite pixel must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn triangulate_point_dlt_rejects_a_non_finite_projection_matrix() {
    let (p1, mut p2) = projections();
    p2[(0, 3)] = f64::NAN;
    let a = Point2::new(320.0, 240.0);
    let err = cv_sfm::triangulation::triangulate_point_dlt(&a, &a, &p1, &p2)
        .expect_err("a NaN projection matrix must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn triangulate_points_rejects_a_non_finite_correspondence() {
    let (p1, p2) = projections();
    let good = Point2::new(320.0, 240.0);
    let other = Point2::new(475.0, 240.0);
    let bad = Point2::new(320.0, f64::NAN);

    assert!(
        cv_sfm::triangulation::triangulate_points(&[good], &[other], &p1, &p2).is_ok(),
        "control: finite input must still triangulate"
    );

    let err = cv_sfm::triangulation::triangulate_points(&[good, bad], &[other, other], &p1, &p2)
        .expect_err("a NaN correspondence must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}
