//! P3P must recover the true pose on randomly drawn exact problems.
//!
//! `test_p3p_recovers_ground_truth` in `src/pnp.rs` checks a single fixed
//! scene, and that scene happens to take the branch where Kabsch's `V U^T`
//! is already a proper rotation. On three points the cross-covariance is
//! rank 2, so the reflection branch is taken about half the time - and that
//! branch used to negate *column* 2 of `V^T` where the correction is *row* 2
//! (`R = V diag(1, 1, -1) U^T`). The result was not a rotation that maps the
//! points, so the side-length check discarded it: the true pose was found in
//! ~500 of 1000 exact problems, which cripples `solve_pnp_ransac` (half of all
//! all-inlier samples produced no correct hypothesis).

use cv_calib3d::PnpSolver;
use cv_core::{CameraIntrinsics, Distortion, PinholeModel};
use nalgebra::{Point3, Rotation3, Unit, Vector3};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn model() -> PinholeModel {
    PinholeModel {
        intrinsics: CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 640, 480),
        distortion: Distortion::none(),
    }
}

#[test]
fn p3p_recovers_the_true_pose_on_random_exact_problems() {
    let m = model();
    let mut rng = Lcg(7);
    let trials = 1000;
    let mut found = 0;
    let mut behind = 0;
    for _ in 0..trials {
        let axis = Unit::new_normalize(Vector3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ));
        let r_true =
            Rotation3::from_axis_angle(&axis, rng.range(0.0, std::f64::consts::PI)).into_inner();
        let t_true = Vector3::new(
            rng.range(-0.5, 0.5),
            rng.range(-0.5, 0.5),
            rng.range(4.0, 8.0),
        );

        // Object points in camera-centred terms in a box in front of the
        // camera, then expressed in the world frame.
        let mut obj = [Vector3::zeros(); 3];
        let mut img = [[0.0f64; 2]; 3];
        for i in 0..3 {
            let pc = Vector3::new(
                rng.range(-1.5, 1.5),
                rng.range(-1.0, 1.0),
                t_true.z + rng.range(-1.0, 1.0),
            );
            obj[i] = r_true.transpose() * (pc - t_true);
            let pr = m.intrinsics.project(&Point3::from(pc));
            img[i] = [pr.x, pr.y];
        }

        let poses = PnpSolver::estimate_p3p(&obj, &img, &m).expect("P3P should not error");
        for pose in &poses {
            let r = pose.rotation_matrix();
            if obj.iter().any(|p| (r * p + pose.translation).z <= 0.0) {
                behind += 1;
            }
        }
        if poses.iter().any(|pose| {
            (pose.rotation_matrix() - r_true).norm() < 1e-6
                && (pose.translation - t_true).norm() < 1e-6
        }) {
            found += 1;
        }
    }
    assert!(
        found >= trials * 99 / 100,
        "true pose recovered in {found}/{trials} exact problems"
    );
    assert_eq!(
        behind, 0,
        "{behind} candidate poses put a point behind the camera"
    );
}
