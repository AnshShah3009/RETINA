//! The rotation Jacobian of `project_points_with_jacobian` must match an
//! independent central difference in the rotation vector - including near
//! theta = pi, where the old matrix -> rvec conversion divided by sin(theta)
//! and produced noise (|J - FD| = 5.3e5 at pi - 1e-7).

use cv_calib3d::{project_points_with_jacobian, ProjectPointsOptions};
use cv_core::{CameraIntrinsics, Distortion, Pose};
use nalgebra::{Point3, Rotation3, Unit, Vector3};

fn project(
    rvec: &Vector3<f64>,
    t: &Vector3<f64>,
    p: &Point3<f64>,
    k: &CameraIntrinsics,
    d: &Distortion,
) -> (f64, f64) {
    let r = Rotation3::from_scaled_axis(*rvec);
    let c = r * p.coords + t;
    let (x, y) = d.apply(c.x / c.z, c.y / c.z);
    (k.fx * x + k.cx, k.fy * y + k.cy)
}

#[test]
fn rotation_jacobian_matches_finite_differences_up_to_pi() {
    let k = CameraIntrinsics::new(600.0, 610.0, 320.0, 240.0, 640, 480);
    let d = Distortion::new(-0.1, 0.02, 1e-3, -5e-4, 0.0);
    let t = Vector3::new(0.1, -0.2, 5.0);
    let pts = [Point3::new(0.3, -0.2, 0.4), Point3::new(-0.5, 0.1, -0.3)];
    let axis = Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0));
    for theta in [
        1.0,
        std::f64::consts::PI - 1e-3,
        std::f64::consts::PI - 1e-7,
    ] {
        let rvec = axis.into_inner() * theta;
        let pose = Pose::new(Rotation3::from_scaled_axis(rvec).into_inner(), t);
        let res = project_points_with_jacobian(
            &pts,
            &k,
            &pose,
            &d,
            ProjectPointsOptions {
                compute_jacobians: true,
            },
        )
        .unwrap();
        let j = res.jacobian_rotation.unwrap();
        let h = 1e-6;
        for (i, p) in pts.iter().enumerate() {
            for c in 0..3 {
                let mut a = rvec;
                let mut b = rvec;
                a[c] += h;
                b[c] -= h;
                let (ua, va) = project(&a, &t, p, &k, &d);
                let (ub, vb) = project(&b, &t, p, &k, &d);
                let (fu, fv) = ((ua - ub) / (2.0 * h), (va - vb) / (2.0 * h));
                assert!(
                    (j[(2 * i, c)] - fu).abs() < 1e-3 && (j[(2 * i + 1, c)] - fv).abs() < 1e-3,
                    "theta {theta}: point {i}, column {c}: J = ({}, {}), FD = ({fu}, {fv})",
                    j[(2 * i, c)],
                    j[(2 * i + 1, c)]
                );
            }
        }
    }
}
