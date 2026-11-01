//! Pose refinement must survive an initial rotation that is the identity only
//! up to roundoff.
//!
//! `solve_pnp_refine` linearises the rotation with `extrinsics_to_params`,
//! which used to read the axis-angle straight off `Rotation3::scaled_axis()`:
//! `acos((trace - 1) / 2)` with no clamp. `solve_pnp_dlt` on exactly projected
//! points can hand it a rotation whose trace rounds a few ulps *above* 3 - a
//! camera at the origin looking down +z with 20 points on a 0.75-scaled target
//! measured 3.0000000000000013 - and `acos` of that is NaN. The seed then had a
//! NaN axis, every Levenberg-Marquardt step was poisoned, and the returned pose
//! was all NaN: a finite, exactly consistent input with no finite answer.

use cv_calib3d::{solve_pnp_dlt, solve_pnp_refine};
use cv_core::{CameraIntrinsics, Pose};
use nalgebra::{Point2, Point3};

/// 5 x 2 x 2 lattice, 20 points, lying at z = 2 .. 2 + 0.75 in front of the
/// camera. This exact layout is what makes the DLT rotation round above the
/// identity's trace of 3; the point count, the spacing and the depth all
/// matter, so it is spelled out rather than generated with a perturbation.
fn object_points(scale: f64) -> Vec<Point3<f64>> {
    let mut pts = Vec::with_capacity(20);
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..5 {
                pts.push(Point3::new(
                    ix as f64 * scale,
                    iy as f64 * scale,
                    iz as f64 * scale + 2.0,
                ));
            }
        }
    }
    pts
}

fn project(intr: &CameraIntrinsics, pose: &Pose, p: &Point3<f64>) -> Point2<f64> {
    let c = pose.rotation * p.coords + pose.translation;
    Point2::new(intr.fx * c.x / c.z + intr.cx, intr.fy * c.y / c.z + intr.cy)
}

#[test]
fn refined_pose_is_finite_when_initial_rotation_is_identity_up_to_roundoff() {
    // Identity camera: the rotation under test is the identity, and the exact
    // projections come from it.
    for scale in [0.75, 1.5, 3.0] {
        let intr = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let truth = Pose::identity();
        let obj = object_points(scale);
        let img: Vec<Point2<f64>> = obj.iter().map(|p| project(&intr, &truth, p)).collect();

        // The DLT pose is the seed the refiner is handed in practice (RANSAC
        // and the localization front end both feed it one); at scale 0.75 its
        // rotation matrix is the identity plus roundoff.
        let seed = solve_pnp_dlt(&obj, &img, &intr).expect("DLT should succeed on exact data");

        let refined = solve_pnp_refine(&seed, &obj, &img, &intr, None, 20)
            .expect("refining exact data should succeed");

        assert!(
            refined.rotation.coords.iter().all(|v| v.is_finite()),
            "scale {scale}: refined rotation is not finite: {:?}",
            refined.rotation.coords
        );
        assert!(
            refined.translation.iter().all(|v| v.is_finite()),
            "scale {scale}: refined translation is not finite: {:?}",
            refined.translation
        );

        // The refined pose is the true pose, not merely finite.
        let rot_err = (refined.rotation_matrix() - truth.rotation_matrix()).norm();
        assert!(
            rot_err < 1e-9,
            "scale {scale}: refined rotation drifted by {rot_err:e}"
        );
        let t_err = (refined.translation - truth.translation).norm();
        assert!(
            t_err < 1e-9,
            "scale {scale}: refined translation drifted by {t_err:e}"
        );
    }
}
