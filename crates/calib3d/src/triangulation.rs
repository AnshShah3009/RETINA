use crate::Result;
use cv_core::{CameraIntrinsics, CameraModel, Pose};
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3};

/// Linear triangulation from two views.
///
/// Reconstructs 3D points from corresponding 2D points in two camera views
/// using the DLT (Direct Linear Transform) method with SVD.
///
/// # Arguments
/// * `p1` - First camera projection matrix (3x4)
/// * `p2` - Second camera projection matrix (3x4)
/// * `pts1` - Corresponding 2D points in first view
/// * `pts2` - Corresponding 2D points in second view
///
/// # Returns
/// Vector of reconstructed 3D points, or error if SVD fails
pub fn triangulate_points(
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
) -> Result<Vec<Point3<f64>>> {
    if pts1.len() != pts2.len() {
        return Err(cv_core::Error::AlgorithmError(
            "triangulate_points requires equal point counts".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(pts1.len());
    for (a, b) in pts1.iter().zip(pts2.iter()) {
        let mut m = Matrix4::<f64>::zeros();
        for c in 0..4 {
            m[(0, c)] = a.x * p1[(2, c)] - p1[(0, c)];
            m[(1, c)] = a.y * p1[(2, c)] - p1[(1, c)];
            m[(2, c)] = b.x * p2[(2, c)] - p2[(0, c)];
            m[(3, c)] = b.y * p2[(2, c)] - p2[(1, c)];
        }
        let svd = m.svd(true, true);
        let vt = svd.v_t.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in triangulate_points".to_string())
        })?;
        let xh = vt.row(3);
        let w = xh[(0, 3)];
        if w.abs() < 1e-12 {
            out.push(Point3::new(0.0, 0.0, 0.0));
            continue;
        }
        out.push(Point3::new(xh[(0, 0)] / w, xh[(0, 1)] / w, xh[(0, 2)] / w));
    }

    Ok(out)
}

/// Extract pose from essential matrix and points.
///
/// Recovers camera extrinsics from an essential matrix by testing four possible
/// decompositions and selecting the one that produces the most points with
/// positive depth in both camera frames.
///
/// # Arguments
/// * `essential` - Essential matrix (3x3)
/// * `pts1` - Corresponding 2D points in first view
/// * `pts2` - Corresponding 2D points in second view
/// * `intrinsics` - Camera intrinsics for normalization
///
/// # Returns
/// Camera extrinsics (rotation and translation) of the second camera relative to the first,
/// or error if fewer than 5 points are provided or all candidates fail
pub fn recover_pose_from_essential(
    essential: &Matrix3<f64>,
    pts1: &[Point2<f64>],
    pts2: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Pose> {
    if pts1.len() != pts2.len() || pts1.len() < 5 {
        return Err(cv_core::Error::AlgorithmError(
            "recover_pose_from_essential needs >=5 paired points".to_string(),
        ));
    }

    let svd = essential.svd(true, true);
    let mut u = svd.u.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD U missing in recover_pose_from_essential".to_string())
    })?;
    let mut vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD V^T missing in recover_pose_from_essential".to_string())
    })?;

    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * vt;
    let r2 = u * w.transpose() * vt;
    let t = u.column(2).into_owned();

    let candidates = [
        Pose::new(r1, t),
        Pose::new(r1, -t),
        Pose::new(r2, t),
        Pose::new(r2, -t),
    ];

    let k_inv = intrinsics.inverse_matrix();
    let norm1: Vec<Point2<f64>> = pts1
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();
    let norm2: Vec<Point2<f64>> = pts2
        .iter()
        .map(|p| {
            let v = k_inv * Vector3::new(p.x, p.y, 1.0);
            Point2::new(v[0] / v[2], v[1] / v[2])
        })
        .collect();

    let p1 = Matrix3x4::new(
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );

    let mut best = None;
    let mut best_score = i32::MIN;
    for cand in candidates {
        let rot_mat = cand.rotation_matrix();
        let p2 = Matrix3x4::new(
            rot_mat[(0, 0)],
            rot_mat[(0, 1)],
            rot_mat[(0, 2)],
            cand.translation[0],
            rot_mat[(1, 0)],
            rot_mat[(1, 1)],
            rot_mat[(1, 2)],
            cand.translation[1],
            rot_mat[(2, 0)],
            rot_mat[(2, 1)],
            rot_mat[(2, 2)],
            cand.translation[2],
        );

        let tri = triangulate_points(&p1, &p2, &norm1, &norm2)?;
        let mut score = 0i32;
        for x in &tri {
            let z1 = x.z;
            let x2 = cand.rotation * x.coords + cand.translation;
            let z2 = x2[2];
            if z1 > 0.0 && z2 > 0.0 {
                score += 1;
            }
        }
        if score > best_score {
            best_score = score;
            best = Some(cand);
        }
    }

    best.ok_or_else(|| cv_core::Error::AlgorithmError("No valid pose candidate found".to_string()))
}

/// Linear Triangulation using the Direct Linear Transform (DLT) method.
pub struct Triangulator;

impl Triangulator {
    /// Triangulate a 3D point from two 2D observations and camera projection matrices.
    /// Observations should be in normalized camera coordinates (or pixel coordinates if P includes K).
    pub fn triangulate_linear(
        p1: &Matrix3x4<f64>,
        p2: &Matrix3x4<f64>,
        pt1: &[f64; 2],
        pt2: &[f64; 2],
    ) -> crate::Result<Vector3<f64>> {
        let mut a = Matrix4::zeros();

        // Observation 1: u1 = (P1_1 * X) / (P1_3 * X), v1 = (P1_2 * X) / (P1_3 * X)
        // -> u1 * (P1_3 * X) - P1_1 * X = 0
        // -> v1 * (P1_3 * X) - P1_2 * X = 0
        for j in 0..4 {
            a[(0, j)] = pt1[0] * p1[(2, j)] - p1[(0, j)];
            a[(1, j)] = pt1[1] * p1[(2, j)] - p1[(1, j)];
            a[(2, j)] = pt2[0] * p2[(2, j)] - p2[(0, j)];
            a[(3, j)] = pt2[1] * p2[(2, j)] - p2[(1, j)];
        }

        let svd = nalgebra::SVD::new(a, false, true);
        let v_t = svd.v_t.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed to compute V_t".into())
        })?;

        // Check for degeneracy: the smallest singular value should be significantly smaller than the second smallest
        if svd.singular_values[3] > 0.1 * svd.singular_values[2] {
            return Err(cv_core::Error::AlgorithmError(
                "Degenerate triangulation configuration".into(),
            ));
        }

        let x_h = v_t.row(3); // Last row of V^T

        if x_h[3].abs() < 1e-9 {
            return Err(cv_core::Error::AlgorithmError(
                "Point at infinity or degenerate".into(),
            ));
        }

        Ok(Vector3::new(
            x_h[0] / x_h[3],
            x_h[1] / x_h[3],
            x_h[2] / x_h[3],
        ))
    }

    /// Triangulate multiple points.
    pub fn triangulate_points(
        pose1: &Pose,
        pose2: &Pose,
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
    ) -> Vec<crate::Result<Vector3<f64>>> {
        // Construct projection matrices P = [R | t]
        // Assuming normalized camera coordinates (K = I)
        let p1 = pose1.matrix().fixed_view::<3, 4>(0, 0).into_owned();
        let p2 = pose2.matrix().fixed_view::<3, 4>(0, 0).into_owned();

        pts1.iter()
            .zip(pts2.iter())
            .map(|(pt1, pt2)| Self::triangulate_linear(&p1, &p2, pt1, pt2))
            .collect()
    }

    /// Estimate camera pose using iterative Levenberg-Marquardt refinement.
    /// Returns the refined Pose given an initial guess, 3D points, 2D projections, and camera intrinsics.
    pub fn refine_pnp(
        initial_pose: &Pose,
        object_points: &[Vector3<f64>],
        image_points: &[[f64; 2]],
        model: &cv_core::PinholeModel,
        max_iters: usize,
    ) -> Pose {
        // Implementation using numerical differentiation for projection to support distortion
        let mut current_pose = *initial_pose;
        let mut lambda = 0.001;

        let n = object_points.len();
        let eps = 1e-6;

        for _ in 0..max_iters {
            let mut jtj = nalgebra::Matrix6::<f64>::zeros();
            let mut jtr = nalgebra::Vector6::<f64>::zeros();
            let mut current_err = 0.0;

            let rot = current_pose.rotation;
            let t = current_pose.translation;

            for i in 0..n {
                let p_w = object_points[i];
                let p_c = rot * p_w + t; // Point in camera frame

                // If point is behind camera, ignore
                if p_c.z <= 1e-6 {
                    continue;
                }

                let uv = model.project(&Point3::from(p_c));
                let du = uv.x - image_points[i][0];
                let dv = uv.y - image_points[i][1];
                current_err += du * du + dv * dv;

                // Numerical Jacobian of projection d(u,v)/d(p_c)
                let mut j_proj = nalgebra::Matrix2x3::zeros();

                let p_c_x = Point3::new(p_c.x + eps, p_c.y, p_c.z);
                let p_c_x_neg = Point3::new(p_c.x - eps, p_c.y, p_c.z);
                let uv_x = model.project(&p_c_x);
                let uv_x_neg = model.project(&p_c_x_neg);
                j_proj.set_column(
                    0,
                    &nalgebra::Vector2::new(
                        (uv_x.x - uv_x_neg.x) / (2.0 * eps),
                        (uv_x.y - uv_x_neg.y) / (2.0 * eps),
                    ),
                );

                let p_c_y = Point3::new(p_c.x, p_c.y + eps, p_c.z);
                let p_c_y_neg = Point3::new(p_c.x, p_c.y - eps, p_c.z);
                let uv_y = model.project(&p_c_y);
                let uv_y_neg = model.project(&p_c_y_neg);
                j_proj.set_column(
                    1,
                    &nalgebra::Vector2::new(
                        (uv_y.x - uv_y_neg.x) / (2.0 * eps),
                        (uv_y.y - uv_y_neg.y) / (2.0 * eps),
                    ),
                );

                let p_c_z = Point3::new(p_c.x, p_c.y, p_c.z + eps);
                let p_c_z_neg = Point3::new(p_c.x, p_c.y, p_c.z - eps);
                let uv_z = model.project(&p_c_z);
                let uv_z_neg = model.project(&p_c_z_neg);
                j_proj.set_column(
                    2,
                    &nalgebra::Vector2::new(
                        (uv_z.x - uv_z_neg.x) / (2.0 * eps),
                        (uv_z.y - uv_z_neg.y) / (2.0 * eps),
                    ),
                );

                // Jacobian d(p_c)/d(pose)
                // d(p_c)/dt = I
                // d(p_c)/domega = -[p_c]x

                let dpc_domega = nalgebra::Matrix3::new(
                    0.0, p_c.z, -p_c.y, -p_c.z, 0.0, p_c.x, p_c.y, -p_c.x, 0.0,
                ); // Note: this is actually [p_c]x, so d/domega is -[p_c]x?
                   // p_new = R * p + t.  R approx (I + [w]x). p_new = p + [w]x * p + t = p - [p]x * w + t.
                   // So d(p)/d(w) = -[p]x.

                let j_rot = j_proj * (-dpc_domega);
                let j_trans = j_proj; // * I

                let mut j = nalgebra::Matrix2x6::zeros();
                j.fixed_view_mut::<2, 3>(0, 0).copy_from(&j_rot);
                j.fixed_view_mut::<2, 3>(0, 3).copy_from(&j_trans);

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            let mut lhs = jtj;
            for k in 0..6 {
                lhs[(k, k)] *= 1.0 + lambda;
            }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                // Update pose
                let omega = Vector3::new(delta[0], delta[1], delta[2]);
                let dt = Vector3::new(delta[3], delta[4], delta[5]);

                let d_rot = nalgebra::Rotation3::new(omega);
                let next_rot = d_rot * current_pose.rotation.to_rotation_matrix();
                let next_t = current_pose.translation - dt; // We solved J*delta = -r, so new = old + delta?
                                                            // Wait, typically J*delta = -r -> delta is step towards solution.
                                                            // My J was d(error)/d(param).  Actually J should be d(residual)/d(param).
                                                            // residual = proj - obs.
                                                            // r_new = r_old + J * delta. Want r_new = 0. J * delta = -r_old.
                                                            // So delta = - (J^T J)^-1 J^T r.
                                                            // But here I solved (J^T J) * delta = J^T r.  So delta is (J^T J)^-1 J^T r.
                                                            // So this delta is -step? No, J^T r is gradient.
                                                            // Gauss-Newton: step = -(J^T J)^-1 J^T r.
                                                            // Here delta = (J^T J)^-1 (J^T r).
                                                            // So step = -delta.

                // Let's check my previous code:
                // next_t = current_pose.translation - dt;
                // This implies dt was "positive" step size but subtracted.

                let next_pose = Pose::new(next_rot.into_inner(), next_t);

                // Simple check for improvement
                let mut next_err = 0.0;
                for i in 0..n {
                    let p_c = next_pose.rotation * object_points[i] + next_pose.translation;
                    if p_c.z > 0.0 {
                        let uv = model.project(&Point3::from(p_c));
                        next_err += (uv.x - image_points[i][0]).powi(2)
                            + (uv.y - image_points[i][1]).powi(2);
                    }
                }

                if next_err < current_err {
                    current_pose = next_pose;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 {
                        break;
                    }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        current_pose
    }

    /// Refine a 3D point estimate using non-linear least squares (Gauss-Newton).
    pub fn refine_triangulation(
        projection_matrices: &[Matrix3x4<f64>],
        observations: &[[f64; 2]],
        initial_point: Vector3<f64>,
        max_iters: usize,
    ) -> Vector3<f64> {
        let mut p = initial_point;
        let mut lambda = 0.001; // Levenberg-Marquardt

        for _ in 0..max_iters {
            let mut jtj = Matrix3::<f64>::zeros();
            let mut jtr = Vector3::<f64>::zeros();
            let mut current_err = 0.0;

            for (i, p_mat) in projection_matrices.iter().enumerate() {
                let obs = observations[i];

                // Project point: x = PX
                let x_h = p_mat * p.insert_row(3, 1.0);
                let z_inv = 1.0 / x_h.z;
                let u = x_h.x * z_inv;
                let v = x_h.y * z_inv;

                let du = u - obs[0];
                let dv = v - obs[1];
                current_err += du * du + dv * dv;

                // Jacobian d(u,v) / d(X,Y,Z)
                // u = (p00*X + p01*Y + p02*Z + p03) / (p20*X + p21*Y + p22*Z + p23)
                // du/dX = (p00 * x_h.z - x_h.x * p20) / (x_h.z^2)
                let mut j = nalgebra::Matrix2x3::zeros();
                for k in 0..3 {
                    j[(0, k)] = (p_mat[(0, k)] * x_h.z - x_h.x * p_mat[(2, k)]) * (z_inv * z_inv);
                    j[(1, k)] = (p_mat[(1, k)] * x_h.z - x_h.y * p_mat[(2, k)]) * (z_inv * z_inv);
                }

                jtj += j.transpose() * j;
                jtr += j.transpose() * nalgebra::Vector2::new(du, dv);
            }

            // Solve (J^T J + lambda*I) * delta = J^T r
            let mut lhs = jtj;
            for i in 0..3 {
                lhs[(i, i)] *= 1.0 + lambda;
            }

            if let Some(delta) = lhs.lu().solve(&jtr) {
                let next_p = p - delta;

                // Check if error improved
                let mut next_err = 0.0;
                for (i, p_mat) in projection_matrices.iter().enumerate() {
                    let obs = observations[i];
                    let x_h = p_mat * next_p.insert_row(3, 1.0);
                    let z_inv = 1.0 / x_h.z;
                    let du = x_h.x * z_inv - obs[0];
                    let dv = x_h.y * z_inv - obs[1];
                    next_err += du * du + dv * dv;
                }

                if next_err < current_err {
                    p = next_p;
                    lambda /= 10.0;
                    if delta.norm() < 1e-8 {
                        break;
                    }
                } else {
                    lambda *= 10.0;
                }
            } else {
                break;
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3x4;

    #[test]
    fn test_triangulation() {
        // Camera 1 at origin
        let p1 = Matrix3x4::identity();
        // Camera 2 translated by 1.0 in x
        let mut p2 = Matrix3x4::identity();
        p2[(0, 3)] = -1.0;

        // Point at (0, 0, 5)
        let x_true = Vector3::new(0.0, 0.0, 5.0);

        // Project to cameras
        // x1 = (0, 0, 5) -> [0, 0, 5] -> (0/5, 0/5) = (0, 0)
        // x2 = (-1, 0, 5) -> [-1, 0, 5] -> (-1/5, 0/5) = (-0.2, 0)
        let pt1 = [0.0, 0.0];
        let pt2 = [-0.2, 0.0];

        let x_tri = Triangulator::triangulate_linear(&p1, &p2, &pt1, &pt2).unwrap();

        assert!((x_tri - x_true).norm() < 1e-6);
    }
}
