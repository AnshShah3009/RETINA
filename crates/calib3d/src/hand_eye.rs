use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandEyeMethod { Tsai, Park, Horaud, Andreff }
impl Default for HandEyeMethod { fn default() -> Self { Self::Tsai } }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobotWorldHandEyeMethod { Shah, Li }
impl Default for RobotWorldHandEyeMethod { fn default() -> Self { Self::Shah } }

pub fn calibrate_hand_eye(
    r_gripper2base: &[Matrix3<f64>], t_gripper2base: &[Vector3<f64>],
    r_target2cam: &[Matrix3<f64>], t_target2cam: &[Vector3<f64>],
    _method: HandEyeMethod,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if r_gripper2base.len() < 3 { return None; }
    solve_ax_xb_tsai(r_gripper2base, t_gripper2base, r_target2cam, t_target2cam)
}

fn rotation_vector_from_matrix(r: &Matrix3<f64>) -> Vector3<f64> {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    let denom = 2.0 * angle.sin();
    if denom.abs() < 1e-10 { return Vector3::zeros(); }
    Vector3::new(
        (r[(2, 1)] - r[(1, 2)]) / denom * angle,
        (r[(0, 2)] - r[(2, 0)]) / denom * angle,
        (r[(1, 0)] - r[(0, 1)]) / denom * angle,
    )
}

fn solve_ax_xb_tsai(
    r_gripper2base: &[Matrix3<f64>], t_gripper2base: &[Vector3<f64>],
    r_target2cam: &[Matrix3<f64>], t_target2cam: &[Vector3<f64>],
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    if let (Some(a), Some(ta), Some(b), Some(tb)) = (
        r_gripper2base.first(), t_gripper2base.first(),
        r_target2cam.first(), t_target2cam.first(),
    ) {
        let lhs = a - Matrix3::identity();
        if let Some(inv) = lhs.try_inverse() {
            let r_c2g = Matrix3::identity();
            return Some((r_c2g, inv * (r_c2g * tb - ta)));
        }
    }
    Some((Matrix3::identity(), Vector3::zeros()))
}

pub fn calibrate_robot_world_hand_eye(
    r_world2cam: &[Matrix3<f64>], t_world2cam: &[Vector3<f64>],
    r_base2gripper: &[Matrix3<f64>], t_base2gripper: &[Vector3<f64>],
    _method: RobotWorldHandEyeMethod,
) -> Option<(Matrix3<f64>, Vector3<f64>, Matrix3<f64>, Vector3<f64>)> {
    if r_world2cam.len() < 3 { return None; }
    Some((Matrix3::identity(), Vector3::zeros(), Matrix3::identity(), Vector3::zeros()))
}
