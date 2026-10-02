//! Public entry points must reject non-finite input instead of hanging.
//!
//! `.svd(true, true)` does not return on non-finite input: nalgebra's
//! bidiagonalisation decides convergence by *comparison*, and every comparison
//! against NaN is false, so the iteration can never terminate. Measured on the
//! current nalgebra 0.33: a 3x3 with one NaN returns (with NaN singular values),
//! but a 4x4 with one NaN had to be killed at 200 s.
//!
//! Every solve below assembles a design matrix and decomposes it that way, so
//! one NaN pixel or correspondence among many was enough to hang the call with
//! nothing reported. Two were confirmed reachable from the public API:
//! `solve_pnp_dlt` and `triangulate_points`.
//!
//! These tests assert the error. They are cheap, and the alternative - the
//! behaviour going unnoticed again - is a process that never returns.
//!
//! Note there is no timeout guard: if a regression reintroduces the hang, the
//! test *times out* rather than fails, which is the honest signal for this
//! class of defect.

use cv_calib3d::undistort_points;
use cv_core::{CameraIntrinsics, Distortion};
use nalgebra::{Matrix3x4, Point2, Point3};

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(620.0, 615.0, 320.0, 240.0, 640, 480)
}

/// `P = K [R | t]`: camera 1 at the origin, camera 2 at (0.5, 0, 0) with
/// identity rotation. The translation goes in the 4th column scaled by `fx`.
fn stereo_projections() -> (Matrix3x4<f64>, Matrix3x4<f64>) {
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
fn solve_pnp_dlt_rejects_a_non_finite_object_point() {
    let k = intrinsics();
    let mut obj: Vec<Point3<f64>> = (0..12)
        .map(|i| Point3::new(i as f64 * 0.1, (i % 4) as f64 * 0.1, 1.0))
        .collect();
    let img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| Point2::new(p.x * 620.0 + 320.0, p.y * 615.0 + 240.0))
        .collect();

    // Control: the same call with finite input succeeds.
    assert!(
        cv_calib3d::solve_pnp_dlt(&obj, &img, &k).is_ok(),
        "control: finite input must still solve, or this test proves nothing"
    );

    obj[2].z = f64::NAN;
    let err = cv_calib3d::solve_pnp_dlt(&obj, &img, &k)
        .expect_err("a NaN object point must be reported, not decomposed");
    assert!(
        format!("{err}").contains("not finite"),
        "the error should name the problem, got: {err}"
    );
}

#[test]
fn solve_pnp_dlt_rejects_a_non_finite_image_point() {
    let k = intrinsics();
    let obj: Vec<Point3<f64>> = (0..12)
        .map(|i| Point3::new(i as f64 * 0.1, (i % 4) as f64 * 0.1, 1.0))
        .collect();
    let mut img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| Point2::new(p.x * 620.0 + 320.0, p.y * 615.0 + 240.0))
        .collect();

    assert!(cv_calib3d::solve_pnp_dlt(&obj, &img, &k).is_ok());

    img[3] = Point2::new(f64::NAN, 0.0);
    let err = cv_calib3d::solve_pnp_dlt(&obj, &img, &k)
        .expect_err("a NaN image point must be reported, not decomposed");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn solve_pnp_ransac_rejects_non_finite_input() {
    let k = intrinsics();
    let obj: Vec<Point3<f64>> = (0..12)
        .map(|i| Point3::new(i as f64 * 0.1, (i % 4) as f64 * 0.1, 1.0))
        .collect();
    let mut img: Vec<Point2<f64>> = obj
        .iter()
        .map(|p| Point2::new(p.x * 620.0 + 320.0, p.y * 615.0 + 240.0))
        .collect();
    img[5] = Point2::new(1.0, f64::INFINITY);

    let err = cv_calib3d::solve_pnp_ransac(&obj, &img, &k, None, 2.0, 64)
        .expect_err("non-finite input must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn triangulate_points_rejects_a_non_finite_correspondence() {
    let (p1, p2) = stereo_projections();
    let a1 = Point2::new(320.0, 240.0);
    let a2 = Point2::new(475.0, 240.0);

    assert!(
        cv_calib3d::triangulate_points(&p1, &p2, &[a1], &[a2]).is_ok(),
        "control: finite input must still triangulate"
    );

    let bad = Point2::new(f64::NAN, 240.0);
    let err = cv_calib3d::triangulate_points(&p1, &p2, &[bad], &[a2])
        .expect_err("a NaN correspondence must be reported, not decomposed");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn triangulate_points_rejects_a_non_finite_projection_matrix() {
    let (p1, mut p2) = stereo_projections();
    p2[(0, 3)] = f64::NAN;
    let a = Point2::new(320.0, 240.0);
    let err = cv_calib3d::triangulate_points(&p1, &p2, &[a], &[a])
        .expect_err("a NaN projection matrix must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}

#[test]
fn undistort_points_rejects_a_non_finite_pixel() {
    let k = intrinsics();
    let d = Distortion::none();

    let finite = vec![Point2::new(120.0, 100.0)];
    assert!(
        undistort_points(&finite, &k, &d).is_ok(),
        "control: finite input must still undistort"
    );

    // Before the guard this returned `Ok([NaN, 10.0])` - a NaN point handed back
    // as a successful result, which then flows into every downstream metric.
    let bad = vec![Point2::new(f64::NAN, 10.0)];
    let err = undistort_points(&bad, &k, &d).expect_err("a NaN pixel must be reported");
    assert!(format!("{err}").contains("not finite"), "got: {err}");
}
