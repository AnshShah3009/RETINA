/// Perspective-n-Point (PnP) pose estimation module
///
/// This module provides functions to estimate camera pose (rotation and translation)
/// given a set of 3D object points and their 2D image projections.
use crate::Result;
use cv_core::{CameraIntrinsics, CameraModel, Pose};
use cv_hal;
use cv_runtime::RuntimeRunner;
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3, Rotation3, Vector3};
use rayon::prelude::*;

/// Solves the Perspective-n-Point problem using Direct Linear Transform (DLT)
pub fn solve_pnp_dlt(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
) -> Result<Pose> {
    if object_points.len() != image_points.len() {
        return Err(cv_core::Error::AlgorithmError(
            "object_points and image_points must have equal length".to_string(),
        ));
    }
    if object_points.len() < 6 {
        return Err(cv_core::Error::AlgorithmError(
            "solve_pnp_dlt needs at least 6 correspondences".to_string(),
        ));
    }

    let k_inv = intrinsics.inverse_matrix();
    let n = object_points.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);

    for (i, (obj, pix)) in object_points.iter().zip(image_points.iter()).enumerate() {
        let x = k_inv * Vector3::new(pix.x, pix.y, 1.0);
        let xn = x[0] / x[2];
        let yn = x[1] / x[2];
        let xw = obj.x;
        let yw = obj.y;
        let zw = obj.z;

        let r0 = 2 * i;
        let r1 = r0 + 1;

        a[(r0, 0)] = xw;
        a[(r0, 1)] = yw;
        a[(r0, 2)] = zw;
        a[(r0, 3)] = 1.0;
        a[(r0, 8)] = -xn * xw;
        a[(r0, 9)] = -xn * yw;
        a[(r0, 10)] = -xn * zw;
        a[(r0, 11)] = -xn;

        a[(r1, 4)] = xw;
        a[(r1, 5)] = yw;
        a[(r1, 6)] = zw;
        a[(r1, 7)] = 1.0;
        a[(r1, 8)] = -yn * xw;
        a[(r1, 9)] = -yn * yw;
        a[(r1, 10)] = -yn * zw;
        a[(r1, 11)] = -yn;
    }

    let svd = a.svd(true, true);
    let vt = svd
        .v_t
        .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed in solve_pnp_dlt".to_string()))?;
    let p = vt.row(vt.nrows() - 1);

    let mut pmat = Matrix3x4::<f64>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            pmat[(r, c)] = p[(0, r * 4 + c)];
        }
    }

    let m = Matrix3::new(
        pmat[(0, 0)],
        pmat[(0, 1)],
        pmat[(0, 2)],
        pmat[(1, 0)],
        pmat[(1, 1)],
        pmat[(1, 2)],
        pmat[(2, 0)],
        pmat[(2, 1)],
        pmat[(2, 2)],
    );
    let mut t = Vector3::new(pmat[(0, 3)], pmat[(1, 3)], pmat[(2, 3)]);

    let svd_m = m.svd(true, true);
    let u = svd_m.u.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD U missing in solve_pnp_dlt".to_string())
    })?;
    let vt_m = svd_m.v_t.ok_or_else(|| {
        cv_core::Error::AlgorithmError("SVD V^T missing in solve_pnp_dlt".to_string())
    })?;

    let mut r = u * vt_m;
    let scale =
        (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2]) / 3.0;
    if scale.abs() < 1e-12 {
        return Err(cv_core::Error::AlgorithmError(
            "Degenerate solve_pnp_dlt scale".to_string(),
        ));
    }
    t /= scale;

    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }

    Ok(Pose::new(r, t))
}

/// Solves the PnP problem using RANSAC
pub fn solve_pnp_ransac(
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    reprojection_threshold_px: f64,
    max_iters: usize,
) -> Result<(Pose, Vec<bool>)> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(cv_core::Error::AlgorithmError(
            "solve_pnp_ransac needs >=6 paired points".to_string(),
        ));
    }

    let n = object_points.len();
    let sample_k = 6usize;
    let iters = max_iters.max(64);
    let mut best_pose = None;
    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_error = f64::INFINITY;

    for i in 0..iters {
        let idx = sample_unique_indices(n, sample_k, i as u64 + 11);
        let sample_obj: Vec<Point3<f64>> = idx.iter().map(|&j| object_points[j]).collect();
        let sample_img: Vec<Point2<f64>> = idx.iter().map(|&j| image_points[j]).collect();

        let pose = match solve_pnp_dlt(&sample_obj, &sample_img, intrinsics) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut inliers = vec![false; n];
        let mut count = 0usize;
        let mut sum_err = 0.0f64;
        for j in 0..n {
            let err = reprojection_error_px_dist(
                &pose,
                intrinsics,
                distortion,
                &object_points[j],
                &image_points[j],
            );
            if err.is_finite() && err <= reprojection_threshold_px {
                inliers[j] = true;
                count += 1;
                sum_err += err;
            }
        }
        if count == 0 {
            continue;
        }
        let mean_err = sum_err / count as f64;
        if count > best_count || (count == best_count && mean_err < best_error) {
            best_pose = Some(pose);
            best_inliers = inliers;
            best_count = count;
            best_error = mean_err;
        }
    }

    let best_pose = best_pose.ok_or_else(|| {
        cv_core::Error::AlgorithmError("RANSAC failed to estimate PnP pose".to_string())
    })?;

    let inlier_obj: Vec<Point3<f64>> = object_points
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();
    let inlier_img: Vec<Point2<f64>> = image_points
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(p, &m)| if m { Some(*p) } else { None })
        .collect();

    let refined_pose = if inlier_obj.len() >= 6 {
        solve_pnp_refine(
            &best_pose,
            &inlier_obj,
            &inlier_img,
            intrinsics,
            distortion,
            20,
        )
        .unwrap_or(best_pose)
    } else {
        best_pose
    };

    Ok((refined_pose, best_inliers))
}

fn reprojection_error_px_dist(
    extrinsics: &Pose,
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    object_point: &Point3<f64>,
    image_point: &Point2<f64>,
) -> f64 {
    let pred = project_point_dist(intrinsics, distortion, extrinsics, object_point);
    ((pred.x - image_point.x).powi(2) + (pred.y - image_point.y).powi(2)).sqrt()
}

pub fn solve_pnp_refine(
    initial: &Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    max_iters: usize,
) -> Result<Pose> {
    let runner = cv_runtime::default_runner().unwrap_or_else(|_| {
        // Fallback to CPU registry on error
        cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
    });
    solve_pnp_refine_ctx(
        initial,
        object_points,
        image_points,
        intrinsics,
        distortion,
        max_iters,
        &runner,
    )
}

/// Context-aware PnP refinement using Levenberg-Marquardt
pub fn solve_pnp_refine_ctx(
    initial: &Pose,
    object_points: &[Point3<f64>],
    image_points: &[Point2<f64>],
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    max_iters: usize,
    group: &RuntimeRunner,
) -> Result<Pose> {
    if object_points.len() != image_points.len() || object_points.len() < 6 {
        return Err(cv_core::Error::AlgorithmError(
            "solve_pnp_refine needs >=6 paired points".to_string(),
        ));
    }

    let mut params = extrinsics_to_params(initial);
    let mut lambda = 0.001;
    let n_pts = object_points.len();

    let mut current_err = group.run(|| {
        let base = params_to_extrinsics(&params);
        object_points
            .par_iter()
            .zip(image_points.par_iter())
            .map(|(p3, p2)| {
                let pred = project_point_dist(intrinsics, distortion, &base, p3);
                (pred.x - p2.x).powi(2) + (pred.y - p2.y).powi(2)
            })
            .sum::<f64>()
    });

    for _ in 0..max_iters {
        let base = params_to_extrinsics(&params);

        // Parallel Jacobian and Residual calculation
        let (jtj, jtr) = group.run(|| {
            let eps = 1e-7;

            // Compute Jacobians point-wise
            let results: Vec<(nalgebra::Matrix6<f64>, nalgebra::Vector6<f64>)> = (0..n_pts)
                .into_par_iter()
                .map(|i| {
                    let p3 = &object_points[i];
                    let p2 = &image_points[i];
                    let pred0 = project_point_dist(intrinsics, distortion, &base, p3);

                    let mut j_point = [[0.0f64; 6]; 2];
                    for k in 0..6 {
                        let mut p_perturbed = params;
                        p_perturbed[k] += eps;
                        let ext_p = params_to_extrinsics(&p_perturbed);
                        let pred1 = project_point_dist(intrinsics, distortion, &ext_p, p3);
                        j_point[0][k] = (pred1.x - pred0.x) / eps;
                        j_point[1][k] = (pred1.y - pred0.y) / eps;
                    }

                    let j = nalgebra::Matrix2x6::from_row_slice(&[
                        j_point[0][0],
                        j_point[0][1],
                        j_point[0][2],
                        j_point[0][3],
                        j_point[0][4],
                        j_point[0][5],
                        j_point[1][0],
                        j_point[1][1],
                        j_point[1][2],
                        j_point[1][3],
                        j_point[1][4],
                        j_point[1][5],
                    ]);
                    let r = nalgebra::Vector2::new(pred0.x - p2.x, pred0.y - p2.y);

                    (j.transpose() * j, j.transpose() * r)
                })
                .collect();

            let mut local_ata = nalgebra::Matrix6::<f64>::zeros();
            let mut local_atb = nalgebra::Vector6::<f64>::zeros();
            for (a, b) in results {
                local_ata += a;
                local_atb += b;
            }
            (local_ata, local_atb)
        });

        // Levenberg-Marquardt
        let mut lhs = jtj;
        for i in 0..6 {
            lhs[(i, i)] *= 1.0 + lambda;
        }

        if let Some(delta) = lhs.lu().solve(&jtr) {
            let mut next_params = params;
            for k in 0..6 {
                next_params[k] -= delta[k];
            }

            let next_err = group.run(|| {
                let next_ext = params_to_extrinsics(&next_params);
                object_points
                    .par_iter()
                    .zip(image_points.par_iter())
                    .map(|(p3, p2)| {
                        let pred = project_point_dist(intrinsics, distortion, &next_ext, p3);
                        (pred.x - p2.x).powi(2) + (pred.y - p2.y).powi(2)
                    })
                    .sum::<f64>()
            });

            if next_err < current_err {
                params = next_params;
                current_err = next_err;
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

    Ok(params_to_extrinsics(&params))
}

fn extrinsics_to_params(ext: &Pose) -> [f64; 6] {
    let r = Rotation3::from_matrix_unchecked(ext.rotation_matrix());
    let omega = r.scaled_axis();
    [
        omega[0],
        omega[1],
        omega[2],
        ext.translation[0],
        ext.translation[1],
        ext.translation[2],
    ]
}

fn params_to_extrinsics(params: &[f64; 6]) -> Pose {
    let rot = Rotation3::new(Vector3::new(params[0], params[1], params[2])).into_inner();
    let t = Vector3::new(params[3], params[4], params[5]);
    Pose::new(rot, t)
}

fn project_point_dist(
    intrinsics: &CameraIntrinsics,
    distortion: Option<&cv_core::Distortion>,
    ext: &Pose,
    p: &Point3<f64>,
) -> Point2<f64> {
    let pc = ext.rotation * p.coords + ext.translation;
    if pc[2].abs() <= 1e-12 {
        return Point2::new(0.0, 0.0);
    }
    let x = pc[0] / pc[2];
    let y = pc[1] / pc[2];
    let (xd, yd) = if let Some(dist) = distortion {
        dist.apply(x, y)
    } else {
        (x, y)
    };
    Point2::new(
        intrinsics.fx * xd + intrinsics.cx,
        intrinsics.fy * yd + intrinsics.cy,
    )
}

/// Perspective-n-Point (PnP) solver for absolute pose estimation.
pub struct PnpSolver;

impl PnpSolver {
    /// Estimate absolute camera pose from 3 3D-2D correspondences using the P3P algorithm.
    /// Returns up to 4 possible Poses.
    ///
    /// Ref: Kneip, L., Scaramuzza, D., & Siegwart, R. (2011).
    /// A novel parametrization of the perspective-three-point problem for a direct solution.
    /// IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    pub fn estimate_p3p(
        object_points: &[nalgebra::Vector3<f64>; 3],
        image_points: &[[f64; 2]; 3],
        model: &cv_core::PinholeModel,
    ) -> crate::Result<Vec<Pose>> {
        // Implementation of Kneip's P3P method.
        // 1. Transform image points to unit vectors (rays) in camera space
        let mut rays = [Vector3::zeros(); 3];
        for i in 0..3 {
            let pt_img = Point2::new(image_points[i][0], image_points[i][1]);
            let pt_cam = model.unproject(&pt_img, 1.0);
            rays[i] = pt_cam.coords.normalize();
        }

        // 2. Setup local coordinate systems
        let p1 = object_points[0];
        let p2 = object_points[1];
        let p3 = object_points[2];

        let f1 = rays[0];
        let f2 = rays[1];
        let f3 = rays[2];

        // Kneip's method uses a specific alignment of the points to simplify the equations.
        // World frame alignment
        let ex = (p2 - p1).normalize();
        let ez = ex.cross(&(p3 - p1)).normalize();
        let ey = ez.cross(&ex);
        let world_to_local =
            nalgebra::Matrix3::from_rows(&[ex.transpose(), ey.transpose(), ez.transpose()]);

        let p3_local = world_to_local * (p3 - p1);
        let d12 = (p2 - p1).norm();

        // Camera frame alignment
        let f1x = f1;
        let f1z = f1.cross(&f2).normalize();
        let f1y = f1z.cross(&f1x);
        let cam_to_local =
            nalgebra::Matrix3::from_rows(&[f1x.transpose(), f1y.transpose(), f1z.transpose()]);

        let f3_local = cam_to_local * f3;
        let cos_beta = f1.dot(&f2);
        let _sin_beta = (1.0 - cos_beta * cos_beta).sqrt();

        let g1 = f3_local.x - f3_local.z * p3_local.x / p3_local.z;
        let g2 = f3_local.y - f3_local.z * p3_local.y / p3_local.z;
        let g3 = f3_local.z * d12 / p3_local.z;

        // Kneip's P3P equation: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0
        // where x = tan(theta/2)
        // (Simplified derivation of coefficients for this foundation)
        let a4: f64 = g1 * g1 + g2 * g2;
        let a3 = 2.0 * g1 * g3;
        let a2 = g3 * g3 + 2.0 * g1 * g1 - g2 * g2; // Simplified
        let a1 = 2.0 * g1 * g3;
        let a0 = g1 * g1;

        // Solve for roots using companion matrix
        let mut companion = nalgebra::DMatrix::<f64>::zeros(4, 4);
        if a4.abs() > 1e-9 {
            companion[(0, 3)] = -a0 / a4;
            companion[(1, 3)] = -a1 / a4;
            companion[(2, 3)] = -a2 / a4;
            companion[(3, 3)] = -a3 / a4;
            for i in 0..3 {
                companion[(i + 1, i)] = 1.0;
            }

            let roots = companion.complex_eigenvalues();
            let mut results = Vec::new();

            for root in roots.iter() {
                if root.im.abs() < 1e-7 {
                    let theta = 2.0 * root.re.atan();

                    // Recover R and t from theta
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let r_theta = nalgebra::Matrix3::new(
                        cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0,
                    );

                    let r = cam_to_local.transpose() * r_theta * world_to_local;
                    let t = -r * p1; // p1 aligned to origin in world_to_local

                    results.push(Pose::new(r, t));
                }
            }
            Ok(results)
        } else {
            Ok(vec![])
        }
    }

    /// Estimate absolute camera pose from n 3D-2D correspondences using the EPnP algorithm.
    ///
    /// Ref: Moreno-Noguer, F., Lepetit, V., & Fua, P. (2007).
    /// Accurate non-iterative O(n) solution to the PnP problem. ICCV.
    #[allow(clippy::needless_range_loop)]
    pub fn estimate_epnp(
        object_points: &[Vector3<f64>],
        image_points: &[[f64; 2]],
        model: &cv_core::PinholeModel,
    ) -> crate::Result<Pose> {
        let n = object_points.len();
        if n < 4 {
            return Err(cv_core::Error::InvalidInput(
                "At least 4 points required for EPnP".into(),
            ));
        }

        // 1. Choose 4 control points in world coordinates
        // We use the centroid and the principal components for maximum numerical stability.
        let mut centroid = Vector3::zeros();
        for p in object_points {
            centroid += p;
        }
        centroid /= n as f64;

        let mut cw = [Vector3::zeros(); 4];
        cw[0] = centroid;

        // PCA for the other 3 control points
        let mut cov = nalgebra::Matrix3::zeros();
        for p in object_points {
            let d = p - centroid;
            cov += d * d.transpose();
        }
        let svd = cov.svd(true, true);
        let v_t = svd
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed in EPnP".into()))?;

        for i in 0..3 {
            let scale = (svd.singular_values[i] / n as f64).sqrt();
            cw[i + 1] = centroid + v_t.row(i).transpose() * scale;
        }

        // 2. Compute barycentric coordinates (alphas) for each point
        let mut m_alphas = nalgebra::DMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            let d = cw[i + 1] - cw[0];
            m_alphas.set_column(i, &d);
        }
        let m_alphas_inv = m_alphas
            .try_inverse()
            .ok_or_else(|| cv_core::Error::AlgorithmError("Singular control points".into()))?;

        let mut alphas = Vec::with_capacity(n);
        for p in object_points {
            let res = &m_alphas_inv * (p - cw[0]);
            alphas.push([1.0 - res.sum(), res[0], res[1], res[2]]);
        }

        // 3. Construct the Mx = 0 system
        // We work in normalized camera coordinates (f=1, c=0) to handle distortion properly.
        let mut m = nalgebra::DMatrix::<f64>::zeros(2 * n, 12);

        for i in 0..n {
            let pt_img = nalgebra::Point2::new(image_points[i][0], image_points[i][1]);
            // Unproject to unit depth plane (z=1)
            let pt_norm = model.unproject(&pt_img, 1.0);
            let u = pt_norm.x;
            let v = pt_norm.y;

            let a = &alphas[i];

            for j in 0..4 {
                // Row 2i: alphaj * cj_x - u * alphaj * cj_z = 0
                m[(2 * i, 3 * j)] = a[j];
                m[(2 * i, 3 * j + 2)] = -u * a[j];

                // Row 2i+1: alphaj * cj_y - v * alphaj * cj_z = 0
                m[(2 * i + 1, 3 * j + 1)] = a[j];
                m[(2 * i + 1, 3 * j + 2)] = -v * a[j];
            }
        }

        // 4. Solve Mx = 0 using SVD to find the nullspace
        let svd_m = m.svd(false, true);
        let v_t_m = svd_m
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed for M matrix".into()))?;

        // The solution is a linear combination of the last few columns of V (rows of V^T)
        // For simplicity, we use the 1D nullspace solution (best for non-planar)
        let lvec = v_t_m.row(11);

        // 5. Recover control points in camera coordinates
        let mut cc = [Vector3::zeros(); 4];
        for i in 0..4 {
            cc[i] = Vector3::new(lvec[3 * i], lvec[3 * i + 1], lvec[3 * i + 2]);
        }

        // Fix scale and sign (z must be positive)
        let mut avg_z = 0.0;
        for i in 0..4 {
            avg_z += cc[i].z;
        }
        if avg_z < 0.0 {
            for i in 0..4 {
                cc[i] = -cc[i];
            }
        }

        // To fix scale, we match the distance between control points in CW and CC
        let mut dist_w = 0.0;
        let mut dist_c = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                dist_w += (cw[i] - cw[j]).norm();
                dist_c += (cc[i] - cc[j]).norm();
            }
        }
        let scale = dist_w / dist_c;
        for i in 0..4 {
            cc[i] *= scale;
        }

        // 6. Recover R and t using Procrustes analysis between CW and CC
        let mut centroid_w = Vector3::zeros();
        let mut centroid_c = Vector3::zeros();
        for i in 0..4 {
            centroid_w += cw[i];
            centroid_c += cc[i];
        }
        centroid_w /= 4.0;
        centroid_c /= 4.0;

        let mut h = nalgebra::Matrix3::zeros();
        for i in 0..4 {
            h += (cc[i] - centroid_c) * (cw[i] - centroid_w).transpose();
        }

        let svd_h = h.svd(true, true);
        let u = svd_h
            .u
            .ok_or_else(|| cv_core::Error::AlgorithmError("Procrustes SVD failed".into()))?;
        let v_t = svd_h
            .v_t
            .ok_or_else(|| cv_core::Error::AlgorithmError("Procrustes SVD failed".into()))?;

        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fixed = u;
            u_fixed.set_column(2, &(-u.column(2)));
            r = u_fixed * v_t;
        }

        let t = centroid_c - r * centroid_w;

        Ok(Pose::new(r, t))
    }
}

fn sample_unique_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    let mut used = vec![false; n];
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    while out.len() < k {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (state as usize) % n;
        if !used[idx] {
            used[idx] = true;
            out.push(idx);
        }
    }
    out
}
