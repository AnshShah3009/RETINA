//! `Triangulator::refine_pnp` converges like Gauss-Newton when the camera is
//! far from the world origin.
//!
//! The pose is updated as `R' = exp(w) R`, `t' = t + dt`, so the camera-frame
//! point moves by `w x (R X)` and the rotation block of the Jacobian is
//! `-[R X]x`. It used `-[R X + t]x` (the camera-frame point), which is only
//! right when `t = 0`; with the camera 5 units away the rotation and
//! translation blocks were coupled wrongly and the "Gauss-Newton" step was
//! not one. The unit test `pnp_jacobian_matches_finite_differences` checks
//! the Jacobian directly; this checks what it costs.

use cv_calib3d::Triangulator;
use cv_core::{CameraIntrinsics, CameraModel, Distortion, PinholeModel, Pose};
use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};

fn rms(pose: &Pose, model: &PinholeModel, obj: &[Vector3<f64>], img: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for (p, uv) in obj.iter().zip(img) {
        let pc = pose.rotation * p + pose.translation;
        let q = model.project(&Point3::from(pc));
        sum += (q.x - uv[0]).powi(2) + (q.y - uv[1]).powi(2);
    }
    (sum / obj.len() as f64).sqrt()
}

#[test]
fn refine_pnp_converges_with_a_distant_camera() {
    let model = PinholeModel::new(
        CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480),
        Distortion::none(),
    );
    let truth = Pose::new(
        Rotation3::from_euler_angles(0.1, -0.2, 0.05).into_inner(),
        Vector3::new(0.4, -0.3, 6.0),
    );
    let obj: Vec<Vector3<f64>> = (0..30)
        .map(|i| {
            let t = i as f64;
            Vector3::new(
                (t * 0.7).sin() * 1.5,
                (t * 1.3).cos() * 1.2,
                (t * 0.4).sin() * 0.8,
            )
        })
        .collect();
    let img: Vec<[f64; 2]> = obj
        .iter()
        .map(|p| {
            let q = model.project(&Point3::from(truth.rotation * p + truth.translation));
            [q.x, q.y]
        })
        .collect();

    // Start 0.08 rad and 0.3 units away.
    let start = Pose {
        rotation: UnitQuaternion::from_euler_angles(0.05, -0.03, 0.04) * truth.rotation,
        translation: truth.translation + Vector3::new(0.2, -0.2, 0.1),
    };
    let e0 = rms(&start, &model, &obj, &img);
    let refined = Triangulator::refine_pnp(&start, &obj, &img, &model, 4);
    let e4 = rms(&refined, &model, &obj, &img);
    // Noise-free data: Gauss-Newton is quadratically convergent from here.
    assert!(
        e4 < 1e-6,
        "reprojection RMS {e0:.3} px -> {e4:.3e} px after 4 iterations"
    );
}
