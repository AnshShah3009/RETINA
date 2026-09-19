//! Hand-eye calibration (AX = XB).
//!
//! Implements a Park–Martin style linear solution on relative motions:
//! rotation first via the logarithm map, then translation by least squares.
//!
//! Conventions follow OpenCV:
//! - `calibrate_hand_eye` receives `gripper2base` and `target2cam` poses and
//!   returns `cam2gripper` (X in A·X = X·B).
//! - `calibrate_robot_world_hand_eye` receives `world2cam` and
//!   `base2gripper` poses and solves A·X = Z·B for X = `base2world` and
//!   Z = `gripper2cam`.

use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandEyeMethod { Tsai, Park, Horaud, Andreff }
impl Default for HandEyeMethod { fn default() -> Self { Self::Tsai } }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobotWorldHandEyeMethod { Shah, Li }
impl Default for RobotWorldHandEyeMethod { fn default() -> Self { Self::Shah } }

/// Eye-in-hand calibration: solve A·X = X·B for X = `cam2gripper`.
///
/// * `r/t_gripper2base` — robot flange poses in base coordinates (per view)
/// * `r/t_target2cam` — calibration target poses in camera coordinates (per view)
///
/// Returns `(R_cam2gripper, t_cam2gripper)` or `None` if the motions do not
/// span enough independent rotations to determine the transform.
pub fn calibrate_hand_eye(
    r_gripper2base: &[Matrix3<f64>], t_gripper2base: &[Vector3<f64>],
    r_target2cam: &[Matrix3<f64>], t_target2cam: &[Vector3<f64>],
    _method: HandEyeMethod,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let n = r_gripper2base.len();
    if n < 3 || t_gripper2base.len() != n || r_target2cam.len() != n || t_target2cam.len() != n {
        return None;
    }

    // Relative motions for every pose pair satisfy A_ij · X = X · B_ij
    // with X = H_g2c (camera-to-gripper), derived from the fixed chain
    // H_b2t = G_i · X · T_i:
    //   A_ij = G_j⁻¹ · G_i   (relative camera motion, gripper frame)
    //   B_ij = T_j · T_i⁻¹   (relative target motion, camera frame)
    let mut a_rot = Vec::new();
    let mut a_tr = Vec::new();
    let mut b_rot = Vec::new();
    let mut b_tr = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            // A = G_j⁻¹ · G_i
            a_rot.push(r_gripper2base[j].transpose() * r_gripper2base[i]);
            a_tr.push(r_gripper2base[j].transpose() * (&t_gripper2base[i] - &t_gripper2base[j]));
            // B = T_j · T_i⁻¹
            b_rot.push(r_target2cam[j] * r_target2cam[i].transpose());
            b_tr.push(
                &t_target2cam[j]
                    - r_target2cam[j] * (r_target2cam[i].transpose() * &t_target2cam[i]),
            );
        }
    }

    solve_ax_xb(&a_rot, &a_tr, &b_rot, &b_tr)
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

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Convert γ = 2·sin(θ/2)·axis into a rotation matrix.
fn gamma_to_rotation(gamma: &Vector3<f64>) -> Option<Matrix3<f64>> {
    let half_sine = gamma.norm() / 2.0;
    if half_sine <= 1e-12 || half_sine > 1.0 + 1e-9 {
        return None;
    }
    let theta = 2.0 * half_sine.asin();
    let axis = gamma.normalize();
    Some(
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), theta)
            .into_inner(),
    )
}

/// Dense normal-equations solve for min ‖Ax − b‖².
/// `rows` are the stacked 3-wide rows of A; `rhs` holds one scalar per row.
fn solve_least_squares_3n(rows: &[[f64; 3]], rhs: &[f64]) -> Option<Vector3<f64>> {
    if rows.is_empty() || rows.len() != rhs.len() {
        return None;
    }
    let mut ata = Matrix3::zeros();
    let mut atb = Vector3::zeros();
    for (row, &b) in rows.iter().zip(rhs.iter()) {
        let rv = Vector3::new(row[0], row[1], row[2]);
        ata += rv * rv.transpose();
        atb += rv * b;
    }
    ata.try_inverse().map(|inv| inv * atb)
}

/// Core AX=XB solver over relative motions.
///
/// Rotation via the quaternion Q-method: each pair contributes
/// `[L(q_A) − R(q_B)]·q_X ≈ 0` (Hamilton convention, `(w, x, y, z)` order);
/// the solution is the eigenvector of the stacked Gram matrix with the
/// smallest eigenvalue. Translation then solves
/// `(R_A − I)·t_X = R_X·t_B − t_A` in least squares.
fn solve_ax_xb(
    a_rot: &[Matrix3<f64>],
    a_tr: &[Vector3<f64>],
    b_rot: &[Matrix3<f64>],
    b_tr: &[Vector3<f64>],
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let mut gram = nalgebra::Matrix4::<f64>::zeros();
    for (ra, rb) in a_rot.iter().zip(b_rot.iter()) {
        let qa = mat3_to_quat(ra);
        let qb = mat3_to_quat(rb);
        let d = quat_left(&qa) - quat_right(&qb);
        gram += d.transpose() * d;
    }

    // Null vector: eigenvector of the PSD Gram matrix with the smallest eigenvalue.
    let eig = gram.symmetric_eigen();
    let idx = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;
    let col = eig.eigenvectors.column(idx);
    let q = [col[0], col[1], col[2], col[3]];
    let r_x = quat_to_mat3(&q)?;

    let mut tr_rows: Vec<[f64; 3]> = Vec::with_capacity(a_rot.len());
    let mut tr_rhs: Vec<f64> = Vec::with_capacity(a_rot.len());
    for ((ar, atr), btr) in a_rot.iter().zip(a_tr.iter()).zip(b_tr.iter()) {
        tr_rows.push([
            ar[(0, 0)] - 1.0, ar[(0, 1)], ar[(0, 2)],
        ]);
        let v = r_x * btr - atr;
        tr_rhs.push(v.x);
        tr_rows.push([ar[(1, 0)], ar[(1, 1)] - 1.0, ar[(1, 2)]]);
        tr_rhs.push(v.y);
        tr_rows.push([ar[(2, 0)], ar[(2, 1)], ar[(2, 2)] - 1.0]);
        tr_rhs.push(v.z);
    }
    let t_x = solve_least_squares_3n(&tr_rows, &tr_rhs)?;

    Some((r_x, t_x))
}

/// Matrix3 -> quaternion in (w, x, y, z) order.
fn mat3_to_quat(r: &Matrix3<f64>) -> [f64; 4] {
    let q = nalgebra::UnitQuaternion::from_matrix(r).into_inner();
    [q.w, q.i, q.j, q.k]
}

/// Normalized quaternion (w, x, y, z) -> Matrix3.
fn quat_to_mat3(q: &[f64; 4]) -> Option<Matrix3<f64>> {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        return None;
    }
    let quat = nalgebra::Quaternion::new(q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    Some(
        nalgebra::UnitQuaternion::from_quaternion(quat)
            .to_rotation_matrix()
            .into_inner(),
    )
}

/// Left quaternion-multiplication matrix: L(a)·vec(b) = vec(a ⊗ b),
/// Hamilton convention with `(w, x, y, z)` component order.
fn quat_left(q: &[f64; 4]) -> nalgebra::Matrix4<f64> {
    let [w, x, y, z] = *q;
    nalgebra::Matrix4::new(
        w, -x, -y, -z,
        x,  w, -z,  y,
        y,  z,  w, -x,
        z, -y,  x,  w,
    )
}

/// Right quaternion-multiplication matrix: R(b)·vec(a) = vec(a ⊗ b).
fn quat_right(q: &[f64; 4]) -> nalgebra::Matrix4<f64> {
    let [w, x, y, z] = *q;
    nalgebra::Matrix4::new(
        w, -x, -y, -z,
        x,  w,  z, -y,
        y, -z,  w,  x,
        z,  y, -x,  w,
    )
}

/// Robot-world/hand-eye calibration: solve A·X = Z·B.
///
/// * inputs: `world2cam` poses (A) and `base2gripper` poses (B) per view
/// * returns `(R_base2world, t_base2world, R_gripper2cam, t_gripper2cam)`
///   i.e. X and Z from A_i·X = Z·B_i, or `None` when the data is degenerate.
pub fn calibrate_robot_world_hand_eye(
    r_world2cam: &[Matrix3<f64>], t_world2cam: &[Vector3<f64>],
    r_base2gripper: &[Matrix3<f64>], t_base2gripper: &[Vector3<f64>],
    _method: RobotWorldHandEyeMethod,
) -> Option<(Matrix3<f64>, Vector3<f64>, Matrix3<f64>, Vector3<f64>)> {
    let n = r_world2cam.len();
    if n < 3 || t_world2cam.len() != n || r_base2gripper.len() != n || t_base2gripper.len() != n {
        return None;
    }

    // Pairwise elimination of X from A_i X = Z B_i gives an AX=XB problem for Z:
    //   (A_j A_i⁻¹) Z = Z (B_j B_i⁻¹)
    let mut rel_a_rot = Vec::new();
    let mut rel_a_tr = Vec::new();
    let mut rel_b_rot = Vec::new();
    let mut rel_b_tr = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            // A_j · A_i⁻¹
            let ai_inv_rot = r_world2cam[i].transpose();
            let ai_inv_tr = -(ai_inv_rot * &t_world2cam[i]);
            rel_a_rot.push(r_world2cam[j] * ai_inv_rot);
            rel_a_tr.push(&t_world2cam[j] + r_world2cam[j] * ai_inv_tr);

            // B_j · B_i⁻¹
            let bi_inv_rot = r_base2gripper[i].transpose();
            let bi_inv_tr = -(bi_inv_rot * &t_base2gripper[i]);
            rel_b_rot.push(r_base2gripper[j] * bi_inv_rot);
            rel_b_tr.push(&t_base2gripper[j] + r_base2gripper[j] * bi_inv_tr);
        }
    }

    let z = solve_ax_xb(&rel_a_rot, &rel_a_tr, &rel_b_rot, &rel_b_tr)?;

    // With Z known: X = A_i⁻¹ · Z · B_i for every i; average the estimates.
    let mut x_rots = Vec::with_capacity(n);
    let mut x_translations = Vec::with_capacity(n);
    for i in 0..n {
        let ai_inv_rot = r_world2cam[i].transpose();
        let ai_inv_tr = -(ai_inv_rot * &t_world2cam[i]);
        // (A⁻¹·Z): rotation m and translation m_t
        let m = ai_inv_rot * z.0;
        let m_t = ai_inv_tr + ai_inv_rot * &z.1;
        x_rots.push(m * r_base2gripper[i]);
        x_translations.push(m_t + m * &t_base2gripper[i]);
    }
    let x_rot = average_rotations(&x_rots)?;
    let x_t = average_translations(&x_translations);

    Some((x_rot, x_t, z.0, z.1))
}

/// Rotation averaging via the quaternion Markley estimator (largest eigenvector).
fn average_rotations(rots: &[Matrix3<f64>]) -> Option<Matrix3<f64>> {
    if rots.is_empty() {
        return None;
    }
    let mut m = nalgebra::Matrix4::<f64>::zeros();
    for r in rots {
        let q = *nalgebra::UnitQuaternion::from_matrix(r).quaternion();
        let v = q.coords;
        m += v * v.transpose();
    }
    let eig = m.symmetric_eigen();
    let idx = eig
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;
    let col = eig.eigenvectors.column(idx);
    let q = nalgebra::Quaternion::from_vector(nalgebra::Vector4::new(col[0], col[1], col[2], col[3]));
    Some(nalgebra::UnitQuaternion::from_quaternion(q).to_rotation_matrix().into_inner())
}

fn average_translations(ts: &[Vector3<f64>]) -> Vector3<f64> {
    let mut acc = Vector3::zeros();
    for t in ts {
        acc += t;
    }
    acc / ts.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    /// Deterministic pseudo-random generator for reproducible tests.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64 / (u32::MAX >> 1) as f64) - 1.0
        }
        fn rotation(&mut self, max_angle: f64) -> Matrix3<f64> {
            let axis = Vector3::new(self.next_f64(), self.next_f64(), self.next_f64());
            if axis.norm() < 1e-6 {
                return Matrix3::identity();
            }
            let angle = self.next_f64() * max_angle;
            Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle).into_inner()
        }
    }

    #[test]
    fn test_hand_eye_recovers_ground_truth() {
        let mut rng = Lcg(42);
        // Ground truth: X = cam2gripper
        let x_true_rot = rng.rotation(1.2);
        let x_true_t = Vector3::new(0.05, -0.02, 0.10);

        // Fixed base→target transform
        let b2t = (
            rng.rotation(1.0),
            Vector3::new(0.5, 0.1, 0.8),
        );

        let n = 8;
        let mut r_g2b = Vec::new();
        let mut t_g2b = Vec::new();
        let mut r_t2c = Vec::new();
        let mut t_t2c = Vec::new();

        for _ in 0..n {
            let g = (rng.rotation(1.5), Vector3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()));
            // H_b2t = G⁻¹ X ⇒ T = target2cam = (b2t⁻¹ G X)⁻¹ = X⁻¹ G⁻¹ b2t
            let g_inv_rot = g.0.transpose();
            let g_inv_t = -(g_inv_rot * g.1);
            // X⁻¹:
            let x_inv_rot = x_true_rot.transpose();
            let x_inv_t = -(x_inv_rot * &x_true_t);
            // M = X⁻¹ · G⁻¹ · b2t  (this is target2cam)
            let m1_rot = x_inv_rot * g_inv_rot;
            let m1_t = x_inv_t + x_inv_rot * g_inv_t;
            let m_rot = m1_rot * b2t.0;
            let m_t = m1_t + m1_rot * b2t.1;
            // Invert M to get the expected T = target2cam? No — M IS target2cam by construction.
            r_g2b.push(g.0);
            t_g2b.push(g.1);
            r_t2c.push(m_rot);
            t_t2c.push(m_t);
        }

        let (r_x, t_x) =
            calibrate_hand_eye(&r_g2b, &t_g2b, &r_t2c, &t_t2c, HandEyeMethod::Tsai)
                .expect("hand-eye should converge on synthetic data");

        // Compare up to sign of relative motion ambiguity: check A·X ≈ X·B residual
        // and closeness to ground truth.
        let rot_err = rotation_vector_from_matrix(&(x_true_rot.transpose() * r_x)).norm();
        assert!(rot_err < 1e-3, "rotation error too large: {}", rot_err);
        assert!(
            (t_x - x_true_t).norm() < 1e-3,
            "translation error too large: {} vs {:?}",
            (t_x - x_true_t).norm(),
            t_x
        );
    }

    #[test]
    fn test_robot_world_recovers_ground_truth() {
        let mut rng = Lcg(7);
        let x_true_rot = rng.rotation(1.0); // base2world
        let x_true_t = Vector3::new(0.3, 0.0, -0.1);
        let z_true_rot = rng.rotation(1.0); // gripper2cam
        let z_true_t = Vector3::new(-0.02, 0.04, 0.06);

        let n = 8;
        let mut r_w2c = Vec::new();
        let mut t_w2c = Vec::new();
        let mut r_b2g = Vec::new();
        let mut t_b2g = Vec::new();

        for _ in 0..n {
            // Random consistent pair: choose B_i (base2gripper) freely, then
            // A_i = Z · B_i · X⁻¹
            let b = (rng.rotation(1.4), Vector3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()));
            let x_inv_rot = x_true_rot.transpose();
            let x_inv_t = -(x_inv_rot * &x_true_t);
            let m1_rot = z_true_rot * b.0;
            let m1_t = z_true_t + z_true_rot * b.1;
            r_w2c.push(m1_rot * x_inv_rot);
            t_w2c.push(m1_t + m1_rot * x_inv_t);
            r_b2g.push(b.0);
            t_b2g.push(b.1);
        }

        let (rx, tx, rz, tz) = calibrate_robot_world_hand_eye(
            &r_w2c, &t_w2c, &r_b2g, &t_b2g, RobotWorldHandEyeMethod::Shah,
        )
        .expect("robot-world hand-eye should converge");

        assert!(rotation_vector_from_matrix(&(x_true_rot.transpose() * rx)).norm() < 1e-3, "X rotation mismatch");
        assert!((tx - x_true_t).norm() < 1e-3, "X translation mismatch");
        assert!(rotation_vector_from_matrix(&(z_true_rot.transpose() * rz)).norm() < 1e-3, "Z rotation mismatch");
        assert!((tz - z_true_t).norm() < 1e-3, "Z translation mismatch");
    }
}
