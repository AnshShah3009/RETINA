/// Perspective-n-Point (PnP) pose estimation module
///
/// This module provides functions to estimate camera pose (rotation and translation)
/// given a set of 3D object points and their 2D image projections.
use crate::Result;
use cv_core::{CameraIntrinsics, Pose};
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
        return Err(cv_core::Error::CalibrationError(
            "object_points and image_points must have equal length".to_string(),
        ));
    }
    if object_points.len() < 6 {
        return Err(cv_core::Error::CalibrationError(
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
    let vt = svd.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD failed in solve_pnp_dlt".to_string())
    })?;
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
        cv_core::Error::CalibrationError("SVD U missing in solve_pnp_dlt".to_string())
    })?;
    let vt_m = svd_m.v_t.ok_or_else(|| {
        cv_core::Error::CalibrationError("SVD V^T missing in solve_pnp_dlt".to_string())
    })?;

    let mut r = u * vt_m;
    let scale =
        (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2]) / 3.0;
    if scale.abs() < 1e-12 {
        return Err(cv_core::Error::CalibrationError(
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
        return Err(cv_core::Error::CalibrationError(
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
        cv_core::Error::CalibrationError("RANSAC failed to estimate PnP pose".to_string())
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
        return Err(cv_core::Error::CalibrationError(
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
