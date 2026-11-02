//! `solve_pnp_dlt` on planar targets must recover the true pose.
//!
//! The planar branch fits a homography from in-plane PCA coordinates
//! `(a, b) = ((X - c)·u, (X - c)·v)` to the image, so `H ~ [R·u, R·v, R·c + t]`.
//! It used to read `[h1 h2 h1×h2]` as the rotation itself and `h3` as the
//! translation, which is only true when the PCA basis `[u v n]` is the identity
//! and the centroid `c` is the origin - i.e. never for a chessboard whose
//! origin sits at a corner. Measured on exact projections of a 5×5 board at
//! `x ∈ [0.5, 0.9]`, `y ∈ [0.3, 0.7]`, `z = 0`, identity rotation, `t = (-0.1,
//! 0.05, 2)`: max reprojection error 344 px and a translation error of 0.86
//! (out of a 2 m depth); a tilted plane was off by 92 px. `cv2.solvePnP` on the
//! same data reprojects to 1e-12 px.

use cv_calib3d::solve_pnp_dlt;
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Point2, Point3, Rotation3, Vector3};

fn project(k: &CameraIntrinsics, pose: &Pose, x: &Point3<f64>) -> Point2<f64> {
    let c = pose.rotation_matrix() * x.coords + pose.translation;
    Point2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy)
}

fn boards() -> Vec<(&'static str, Vec<Point3<f64>>)> {
    vec![
        (
            "square board, origin at a corner",
            (0..25)
                .map(|i| Point3::new(0.5 + 0.1 * (i % 5) as f64, 0.3 + 0.1 * (i / 5) as f64, 0.0))
                .collect(),
        ),
        (
            "rectangular board, origin at a corner",
            (0..24)
                .map(|i| Point3::new(0.5 + 0.1 * (i % 6) as f64, 0.3 + 0.1 * (i / 6) as f64, 0.0))
                .collect(),
        ),
        (
            "tilted plane",
            (0..24)
                .map(|i| {
                    let a = 0.1 * (i % 6) as f64;
                    let b = 0.1 * (i / 6) as f64;
                    Point3::new(a, b, 0.3 * a + 0.2 * b + 0.1)
                })
                .collect(),
        ),
    ]
}

#[test]
fn planar_dlt_recovers_ground_truth_pose() {
    let k = CameraIntrinsics::new(800.0, 760.0, 320.0, 240.0, 640, 480);
    let rotations = [
        Rotation3::from_euler_angles(0.2, -0.3, 0.4),
        Rotation3::from_euler_angles(-0.1, 0.25, 1.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.0),
        Rotation3::from_euler_angles(0.4, 0.1, -2.5),
    ];
    let t = Vector3::new(-0.1, 0.05, 2.0);
    for (ri, rot) in rotations.iter().enumerate() {
        let gt = Pose::new(rot.matrix().clone_owned(), t);
        for (name, obj) in boards() {
            let img: Vec<_> = obj.iter().map(|p| project(&k, &gt, p)).collect();
            let est = solve_pnp_dlt(&obj, &img, &k)
                .unwrap_or_else(|e| panic!("rotation {ri}, {name}: {e}"));
            let reproj = obj
                .iter()
                .zip(&img)
                .map(|(o, i)| (project(&k, &est, o) - i).norm())
                .fold(0.0, f64::max);
            let rot_err = (est.rotation_matrix() - gt.rotation_matrix()).norm();
            let t_err = (est.translation - t).norm();
            assert!(
                reproj < 1e-6 && rot_err < 1e-8 && t_err < 1e-8,
                "rotation {ri}, {name}: max reprojection {reproj:.3e} px, \
                 |R - R_gt| {rot_err:.3e}, |t - t_gt| {t_err:.3e}"
            );
        }
    }
}

/// With noise the planar solve stays close to the truth (it is a linear
/// initialiser, not a refinement, so the bar is loose: measured 0.024 here,
/// against 0.010 for `cv2.solvePnP(SOLVEPNP_EPNP)` on the same data; the old
/// decomposition was 62 px out).
#[test]
fn planar_dlt_is_close_under_pixel_noise() {
    let k = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480);
    let gt = Pose::new(
        Rotation3::from_euler_angles(0.3, -0.2, 0.5)
            .matrix()
            .clone_owned(),
        Vector3::new(-0.2, 0.1, 1.5),
    );
    let obj: Vec<Point3<f64>> = (0..48)
        .map(|i| Point3::new(0.03 * (i % 8) as f64, 0.03 * (i / 8) as f64, 0.0))
        .collect();
    // Deterministic +-0.5 px perturbation.
    let img: Vec<Point2<f64>> = obj
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q = project(&k, &gt, p);
            let s = (i as f64 * 1.7).sin() * 0.5;
            let c = (i as f64 * 2.3).cos() * 0.5;
            Point2::new(q.x + s, q.y + c)
        })
        .collect();
    let est = solve_pnp_dlt(&obj, &img, &k).unwrap();
    let reproj = obj
        .iter()
        .zip(&img)
        .map(|(o, i)| (project(&k, &est, o) - i).norm())
        .fold(0.0, f64::max);
    assert!(reproj < 2.0, "max reprojection {reproj:.3} px");
    assert!(
        (est.translation - gt.translation).norm() < 0.05,
        "translation {:?} vs {:?}",
        est.translation,
        gt.translation
    );
}
