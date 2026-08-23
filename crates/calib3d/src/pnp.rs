/// Perspective-n-Point (PnP) pose estimation module
///
/// This module provides functions to estimate camera pose (rotation and translation)
/// given a set of 3D object points and their 2D image projections.
use crate::Result;
use cv_core::{CameraIntrinsics, CameraModel, Pose};
use cv_hal;
use cv_runtime::RuntimeRunner;
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Rotation3, Vector3};
use rayon::prelude::*;

/// Solves the Perspective-n-Point problem using Direct Linear Transform (DLT)
///
/// Object points are Hartley-normalized before assembly. Planar inputs (the
/// typical chessboard/calibration-target case) are detected automatically and
/// solved through a homography decomposition instead — plain DLT is rank
/// deficient there and returns an arbitrary mixture of solutions.
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

    // Normalized image coordinates (K^-1 applied once up front).
    let norm_image: Vec<(f64, f64)> = image_points
        .iter()
        .map(|pix| {
            let x = k_inv * Vector3::new(pix.x, pix.y, 1.0);
            (x[0] / x[2], x[1] / x[2])
        })
        .collect();

    // ---- Planarity detection ----
    let n_pts = object_points.len();
    let centroid = object_points.iter().fold(Vector3::<f64>::zeros(), |acc, p| {
        acc + p.coords
    }) / n_pts as f64;
    let mut cov = Matrix3::<f64>::zeros();
    for p in object_points {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    let cov_svd = cov.svd(true, true);
    let svs = cov_svd.singular_values;
    let planar = svs[0] > 1e-12 && svs[2] <= 1e-6 * svs[0];

    let pose = if planar {
        // ---- Homography-based pose for planar targets ----
        // Plane basis from the covariance eigenvectors.
        let u_mat = cov_svd.u.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in planarity analysis".to_string())
        })?;
        let u_axis = u_mat.column(0);
        let n_axis = u_mat.column(2);
        let v_axis = n_axis.cross(&u_axis);

        let src: Vec<(f64, f64)> = object_points
            .iter()
            .map(|p| {
                let d = p.coords - centroid;
                (d.dot(&u_axis), d.dot(&v_axis))
            })
            .collect();
        let (t_src, src_n) = hartley_2d(&src);

        // Homography DLT on normalized 2D -> normalized image.
        let mut h_sys = DMatrix::<f64>::zeros(2 * n_pts, 9);
        for (i, &(xw, yw)) in src_n.iter().enumerate() {
            let (xn, yn) = norm_image[i];
            let r0 = 2 * i;
            let r1 = r0 + 1;
            h_sys[(r0, 0)] = xw;
            h_sys[(r0, 1)] = yw;
            h_sys[(r0, 2)] = 1.0;
            h_sys[(r0, 6)] = -xn * xw;
            h_sys[(r0, 7)] = -xn * yw;
            h_sys[(r0, 8)] = -xn;
            h_sys[(r1, 3)] = xw;
            h_sys[(r1, 4)] = yw;
            h_sys[(r1, 5)] = 1.0;
            h_sys[(r1, 6)] = -yn * xw;
            h_sys[(r1, 7)] = -yn * yw;
            h_sys[(r1, 8)] = -yn;
        }
        let h_svd = h_sys.svd(true, true);
        let h_vt = h_svd.v_t.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in homography DLT".to_string())
        })?;
        let hv = h_vt.row(h_vt.nrows() - 1);
        let mut h_norm = Matrix3::<f64>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                h_norm[(r, c)] = hv[r * 3 + c];
            }
        }
        if h_norm.norm().abs() < 1e-15 {
            return Err(cv_core::Error::AlgorithmError(
                "Degenerate homography in solve_pnp_dlt".to_string(),
            ));
        }

        // Undo the source-side Hartley transform: H = H_norm · T_src.
        let t_src_m = Matrix3::new(
            t_src[0], t_src[1], t_src[2], t_src[3], t_src[4], t_src[5], t_src[6], t_src[7],
            t_src[8],
        );
        let h = h_norm * t_src_m;

        // Malis/Vargas decomposition: H ≈ [r1 r2 t].
        let h1 = h.column(0);
        let h2 = h.column(1);
        let h3 = h.column(2);
        let norm1 = h1.norm();
        if norm1 < 1e-12 {
            return Err(cv_core::Error::AlgorithmError(
                "Degenerate homography columns in solve_pnp_dlt".to_string(),
            ));
        }
        let mut lambda = 1.0 / norm1;
        let mut r1 = lambda * h1;
        let mut r2 = lambda * h2;
        let mut r3 = r1.cross(&r2);
        let mut rr = Matrix3::from_columns(&[r1, r2, r3]);
        if rr.determinant() < 0.0 {
            lambda = -lambda;
            r1 = lambda * h1;
            r2 = lambda * h2;
            r3 = r1.cross(&r2);
            rr = Matrix3::from_columns(&[r1, r2, r3]);
        }
        // Orthonormalize (removes noise-induced drift from pure rotation).
        let rr_svd = rr.svd(true, true);
        let ru = rr_svd.u.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in pose extraction".to_string())
        })?;
        let rv = rr_svd.v_t.ok_or_else(|| {
            cv_core::Error::AlgorithmError("SVD failed in pose extraction".to_string())
        })?;
        let mut rot = ru * rv;
        if rot.determinant() < 0.0 {
            // Flip the least-significant direction and redo.
            let u_fixed = {
                let mut u2 = ru.clone();
                for c in 0..3 {
                    u2[(c, 2)] = -u2[(c, 2)];
                }
                u2
            };
            rot = u_fixed * rv;
        }
        let trans = lambda * h3;

        // The in-plane PCA basis sign is arbitrary; each sign combination
        // yields a different impostor pose (a 180° rotation about some axis)
        // that leaves depths unchanged and defeats cheirality checks.
        // Disambiguate by direct normalized-coordinate reprojection error
        // over all four det(+1) sign-flip variants.
        let mut best_rot = rot;
        let mut best_err = f64::INFINITY;
        let signs: [[f64; 3]; 4] = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        for sg in signs {
            let d = Matrix3::new(sg[0], 0.0, 0.0, 0.0, sg[1], 0.0, 0.0, 0.0, sg[2]);
            let cand = rot * d;
            let mut acc = 0.0;
            for (obj, &(xn, yn)) in object_points.iter().zip(norm_image.iter()) {
                let pc = cand * obj.coords + trans;
                if !pc[2].is_finite() || pc[2].abs() < 1e-12 {
                    acc = f64::INFINITY;
                    break;
                }
                acc += (pc[0] / pc[2] - xn).powi(2) + (pc[1] / pc[2] - yn).powi(2);
            }
            if acc < best_err {
                best_err = acc;
                best_rot = cand;
            }
        }

        Pose::new(best_rot, trans)
    } else {
        // ---- General DLT with Hartley normalization of object points ----
        let mean = centroid;
        let rms = (object_points
            .iter()
            .map(|p| (p.coords - mean).norm())
            .sum::<f64>()
            / n_pts as f64)
            .max(1e-12);
        let scale = (3.0f64).sqrt() / rms;

        let mut a = DMatrix::<f64>::zeros(2 * n_pts, 12);
        for (i, obj) in object_points.iter().enumerate() {
            let d = obj.coords - mean;
            let xw = d[0] * scale;
            let yw = d[1] * scale;
            let zw = d[2] * scale;
            let (xn, yn) = norm_image[i];

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
            cv_core::Error::AlgorithmError("SVD failed in solve_pnp_dlt".to_string())
        })?;
        let p = vt.row(vt.nrows() - 1);

        let mut pmat = Matrix3x4::<f64>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                pmat[(r, c)] = p[(0, r * 4 + c)];
            }
        }

        // Undo normalization: P = P′ · T  where  x_norm = T · x_world
        // (translate by centroid, scale by s).
        let t_fwd = Matrix4::new(
            scale,
            0.0,
            0.0,
            -scale * mean[0],
            0.0,
            scale,
            0.0,
            -scale * mean[1],
            0.0,
            0.0,
            scale,
            -scale * mean[2],
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let mut full = Matrix4::<f64>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                full[(r, c)] = pmat[(r, c)];
            }
        }
        full[(3, 3)] = 1.0;
        let p_denorm = full * t_fwd;
        for r in 0..3 {
            for c in 0..4 {
                pmat[(r, c)] = p_denorm[(r, c)];
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
        let scale_m =
            (svd_m.singular_values[0] + svd_m.singular_values[1] + svd_m.singular_values[2]) / 3.0;
        if scale_m.abs() < 1e-12 {
            return Err(cv_core::Error::AlgorithmError(
                "Degenerate solve_pnp_dlt scale".to_string(),
            ));
        }
        t /= scale_m;

        if r.determinant() < 0.0 {
            r = -r;
            t = -t;
        }
        Pose::new(r, t)
    };

    Ok(pose)
}

/// Hartley 2D normalization: translate to centroid, scale so mean distance is sqrt(2).
fn hartley_2d(pts: &[(f64, f64)]) -> ([f64; 9], Vec<(f64, f64)>) {
    let n = pts.len().max(1) as f64;
    let mean_x = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let rms = (pts
        .iter()
        .map(|p| {
            let dx = p.0 - mean_x;
            let dy = p.1 - mean_y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n)
        .max(1e-12);
    let s = (2.0f64).sqrt() / rms;
    (
        [s, 0.0, -s * mean_x, 0.0, s, -s * mean_y, 0.0, 0.0, 1.0],
        pts.iter()
            .map(|p| ((p.0 - mean_x) * s, (p.1 - mean_y) * s))
            .collect(),
    )
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
    /// Estimate absolute camera pose from 3 3D-2D correspondences using the
    /// P3P algorithm (Grunert's distance parametrization).
    ///
    /// Eliminating two of the three unknown camera-to-point distances yields
    /// a quartic in the ratio q = v/w; each real positive root reconstructs
    /// candidate distances, and the rigid transform mapping object points to
    /// their ray-scaled camera points is recovered by Kabsch/Umeyama.
    /// Returns up to 4 candidate poses; callers must disambiguate by
    /// cheirality or reprojection.
    pub fn estimate_p3p(
        object_points: &[nalgebra::Vector3<f64>; 3],
        image_points: &[[f64; 2]; 3],
        model: &cv_core::PinholeModel,
    ) -> crate::Result<Vec<Pose>> {
        // Unit rays through each image point.
        let mut rays = [Vector3::zeros(); 3];
        for i in 0..3 {
            let pt_img = Point2::new(image_points[i][0], image_points[i][1]);
            let pt_cam = model.unproject(&pt_img, 1.0);
            let norm = pt_cam.coords.norm();
            if norm < 1e-12 {
                return Err(cv_core::Error::InvalidInput(
                    "degenerate image point at principal point".into(),
                ));
            }
            rays[i] = pt_cam.coords / norm;
        }

        let pa = object_points[0];
        let pb = object_points[1];
        let pc = object_points[2];

        // Side lengths, each opposite the apex angle between its rays.
        let sa = (pb - pc).norm(); // opposite angle(rB, rC)
        let sb = (pa - pc).norm(); // opposite angle(rA, rC)
        let sc = (pa - pb).norm(); // opposite angle(rA, rB)
        if sa < 1e-12 || sb < 1e-12 || sc < 1e-12 {
            return Err(cv_core::Error::InvalidInput(
                "P3P requires non-degenerate (non-collinear, distinct) points".into(),
            ));
        }
        let ca = rays[1].dot(&rays[2]);
        let cb = rays[0].dot(&rays[2]);
        let cg = rays[0].dot(&rays[1]);

        // Quartic coefficients in q = v/w, low-degree first.
        //   p(q) = [m·d(q) + a²·(q² − 1)] / [2a²(cg·q − cb)]
        //   w²  = a² / d(q),  d(q) = q² − 2ca·q + 1,  m = b² − c²
        // obtained by eliminating p from the law-of-cosines system:
        //   b²·d = a²(p² + 1 − 2p·cb),  c²·d = a²(p² + q² − 2pq·cg),
        //   w²  = a²/d.
        let m = sb * sb - sc * sc;
        let dpoly = [1.0, -2.0 * ca, 1.0];
        let npoly = [
            m * dpoly[0] - sa * sa,
            m * dpoly[1],
            m * dpoly[2] + sa * sa,
        ];
        let dpoly_d = [-2.0 * sa * sa * cb, 2.0 * sa * sa * cg];

        fn pmul(a: &[f64], b: &[f64]) -> Vec<f64> {
            let mut out = vec![0.0; a.len() + b.len() - 1];
            for (i, &x) in a.iter().enumerate() {
                for (j, &y) in b.iter().enumerate() {
                    out[i + j] += x * y;
                }
            }
            out
        }
        fn padd(mut a: Vec<f64>, b: &[f64]) -> Vec<f64> {
            if a.len() < b.len() {
                a.resize(b.len(), 0.0);
            }
            for (i, &v) in b.iter().enumerate() {
                a[i] += v;
            }
            a
        }

        let t1 = pmul(&npoly, &npoly); // scaled by a² below
        let t2 = pmul(&npoly, &dpoly_d); // scaled by −2a²cb below
        // T3 = (a² − b²·d) · D²  where  a² appears ONLY in the constant term.
        let t3 = {
            let t = [sa * sa - sb * sb, 2.0 * sb * sb * ca, -sb * sb];
            pmul(&t, &pmul(&dpoly_d, &dpoly_d))
        };

        let mut coeff = vec![0.0f64; 5];
        for (deg, &v) in t1.iter().enumerate() {
            coeff[deg] += sa * sa * v;
        }
        for (deg, &v) in t2.iter().enumerate() {
            coeff[deg] += -2.0 * sa * sa * cb * v;
        }
        for (deg, &v) in t3.iter().enumerate() {
            coeff[deg] += v;
        }

        let c4 = coeff[4];
        if !c4.is_finite() || c4.abs() < 1e-14 {
            return Ok(Vec::new());
        }
 

        let mut companion = nalgebra::DMatrix::<f64>::zeros(4, 4);
        companion[(0, 3)] = -coeff[0] / c4;
        companion[(1, 3)] = -coeff[1] / c4;
        companion[(2, 3)] = -coeff[2] / c4;
        companion[(3, 3)] = -coeff[3] / c4;
        for i in 0..3 {
            companion[(i + 1, i)] = 1.0;
        }

        let roots = companion.complex_eigenvalues();

        let mut results = Vec::new();
        let mut seen_q: Vec<f64> = Vec::new();

        for root in roots.iter() {
            // Companion-matrix eigenvalues of a real quartic come in
            // conjugate pairs with tiny spurious imaginary parts; accept
            // roots whose imaginary component is negligible RELATIVE to the
            // real one, then polish with a Newton step on the quartic.
            let scale = root.re.abs().max(1.0);
            if root.im.abs() > 1e-6 * scale {
                continue;
            }
            let mut q = root.re;

            let poly_at = |x: f64| -> f64 {
                coeff[4] * x.powi(4)
                    + coeff[3] * x.powi(3)
                    + coeff[2] * x * x
                    + coeff[1] * x
                    + coeff[0]
            };
            let poly_prime = |x: f64| -> f64 {
                4.0 * coeff[4] * x.powi(3)
                    + 3.0 * coeff[3] * x * x
                    + 2.0 * coeff[2] * x
                    + coeff[1]
            };
            for _ in 0..5 {
                let f = poly_at(q);
                let fp = poly_prime(q);
                if !fp.is_finite() || fp.abs() < 1e-14 {
                    break;
                }
                q -= f / fp;
            }

            if q <= 1e-8 || !q.is_finite() {
                continue;
            }
            if seen_q.iter().any(|&prev| (prev - q).abs() < 1e-9) {
                continue;
            }
            seen_q.push(q);

            let d = q * q - 2.0 * ca * q + 1.0;
            if d <= 1e-12 {
                continue;
            }
            let w_sq = sa * sa / d;
            if w_sq <= 0.0 || !w_sq.is_finite() {
                continue;
            }
            let w = w_sq.sqrt();

            let denom = 2.0 * sa * sa * (cg * q - cb);
            if denom.abs() < 1e-12 {
                continue;
            }
            let p = (m * d + sa * sa * (q * q - 1.0)) / denom;

            let cam_a = rays[0] * (p * w);
            let cam_b = rays[1] * (q * w);
            let cam_c = rays[2] * w;

            // Sanity: reconstructed side lengths must match.
            let tol = 1e-6 * sa.max(sb).max(sc).max(1.0);
            if ((cam_b - cam_c).norm() - sa).abs() > tol
                || ((cam_a - cam_c).norm() - sb).abs() > tol
                || ((cam_a - cam_b).norm() - sc).abs() > tol
            {
                continue;
            }

            // Kabsch (rigid, no scale): find R,t minimizing Σ|R·P + t − Q|²,
            // i.e. world→camera under this crate's convention.
            let src = [pa, pb, pc];
            let dst = [cam_a, cam_b, cam_c];
            let mut src_c = Vector3::zeros();
            let mut dst_c = Vector3::zeros();
            for i in 0..3 {
                src_c += src[i];
                dst_c += dst[i];
            }
            src_c /= 3.0;
            dst_c /= 3.0;

            let mut h_mat = Matrix3::<f64>::zeros();
            for i in 0..3 {
                let dp = src[i] - src_c;
                let dq = dst[i] - dst_c;
                h_mat += dp * dq.transpose();
            }
            let svd_h = h_mat.svd(true, true);
            let u = svd_h
                .u
                .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed in P3P".to_string()))?;
            let vt = svd_h
                .v_t
                .ok_or_else(|| cv_core::Error::AlgorithmError("SVD failed in P3P".to_string()))?;
            let mut r_rot = vt.transpose() * u.transpose();
            if r_rot.determinant() < 0.0 {
                let mut vt_fix = vt.clone();
                for rr in 0..3 {
                    vt_fix[(rr, 2)] = -vt_fix[(rr, 2)];
                }
                r_rot = vt_fix.transpose() * u.transpose();
            }
            let t_vec = dst_c - r_rot * src_c;

            // Reject mirrored solutions that fail side reconstruction again.
            let ok = src.iter().zip(dst.iter()).all(|(sp, dq)| {
                let pred = r_rot * sp + t_vec;
                (pred - dq).norm() < tol.max(1e-9)
            });
            if !ok {
                continue;
            }

            results.push(Pose::new(r_rot, t_vec));
        }

        Ok(results)
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

#[cfg(test)]
mod dlt_planar_tests {
    use super::*;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(800.0, 810.0, 320.0, 240.0, 640, 480)
    }

    #[test]
    fn test_dlt_planar_target_low_reprojection_error() {
        // Regression: for planar targets (z=0 chessboard-style), plain DLT is
        // rank-deficient and previously returned poses with ~1000 px error.
        let intr = intrinsics();
        let r_true = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0)), 0.09).into_inner();
        let t_true = Vector3::new(0.1, -0.05, 2.0);

        // 7x7 grid of planar object points at z = 0.
        let object_points: Vec<Point3<f64>> = (0..49)
            .map(|i| {
                Point3::new(
                    (i % 7) as f64 * 0.03 - 0.09,
                    (i / 7) as f64 * 0.03 - 0.09,
                    0.0,
                )
            })
            .collect();

        let image_points: Vec<Point2<f64>> = object_points
            .iter()
            .map(|p| {
                let pc = r_true * p.coords + t_true;
                let pr = intr.project(&Point3::from(pc));
                Point2::new(pr.x, pr.y)
            })
            .collect();

        let pose = solve_pnp_dlt(&object_points, &image_points, &intr).unwrap();
        drop(t_true);

        // Reprojection error with the recovered pose must be tiny.
        let r_rec = pose.rotation_matrix();
        let mut err_sq = 0.0;
        for (obj, img) in object_points.iter().zip(image_points.iter()) {
            let pc = r_rec * obj.coords + pose.translation;
            if pc[2] <= 0.0 {
                panic!("recovered pose places point behind camera");
            }
            let pr = intr.project(&Point3::from(pc));
            err_sq += (pr.x - img.x).powi(2) + (pr.y - img.y).powi(2);
        }
        let rms = (err_sq / object_points.len() as f64).sqrt();
        assert!(rms < 1e-3, "planar DLT reprojection RMS too large: {}", rms);
    }

    #[test]
    fn test_dlt_non_planar_low_reprojection_error() {
        let intr = intrinsics();
        let r_true = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::new(-2.0, 1.0, 0.5)), 0.19).into_inner();
        let t_true = Vector3::new(0.2, 0.1, 2.5);

        // Non-planar cloud: two offset grids.
        let mut object_points: Vec<Point3<f64>> = Vec::new();
        for i in 0..36 {
            let x = (i % 6) as f64 * 0.04 - 0.1;
            let y = (i / 6) as f64 * 0.04 - 0.1;
            let z = if i % 2 == 0 { 0.0 } else { 0.08 };
            object_points.push(Point3::new(x, y, z));
        }

        let image_points: Vec<Point2<f64>> = object_points
            .iter()
            .map(|p| {
                let pc = r_true * p.coords + t_true;
                let pr = intr.project(&Point3::from(pc));
                Point2::new(pr.x, pr.y)
            })
            .collect();

        let pose = solve_pnp_dlt(&object_points, &image_points, &intr).unwrap();
        let r_rec = pose.rotation_matrix();
        let mut err_sq = 0.0;
        for (obj, img) in object_points.iter().zip(image_points.iter()) {
            let pc = r_rec * obj.coords + pose.translation;
            let pr = intr.project(&Point3::from(pc));
            err_sq += (pr.x - img.x).powi(2) + (pr.y - img.y).powi(2);
        }
        let rms = (err_sq / object_points.len() as f64).sqrt();
        assert!(rms < 1e-3, "non-planar DLT reprojection RMS too large: {}", rms);
    }
}


#[cfg(test)]
mod p3p_tests {
    use super::*;

    fn model() -> cv_core::PinholeModel {
        cv_core::PinholeModel {
            intrinsics: CameraIntrinsics::new(800.0, 810.0, 320.0, 240.0, 640, 480),
            distortion: cv_core::Distortion::none(),
        }
    }

    #[test]
    fn test_p3p_recovers_ground_truth() {
        let r_true = Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0)),
            0.3,
        )
        .into_inner();
        let t_true = Vector3::new(0.2, -0.1, 3.0);

        let obj = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5, 0.0, 0.1),
            Vector3::new(0.1, 0.6, -0.05),
        ];

        let m = model();
        let mut img = [[0.0f64; 2]; 3];
        for i in 0..3 {
            let pc = r_true * obj[i] + t_true;
            let pr = m.intrinsics.project(&Point3::from(pc));
            img[i] = [pr.x, pr.y];
        }

        let poses =
            PnpSolver::estimate_p3p(&obj, &img, &m).expect("P3P should not error");

        assert!(!poses.is_empty(), "P3P must return at least one solution");
        // The ground-truth pose must be among the (up to 4) candidates.
        let matched = poses.iter().any(|pose| {
            let rel = r_true.transpose() * pose.rotation_matrix();
            let ang = ((rel[(0, 0)] + rel[(1, 1)] + rel[(2, 2)]) / 2.0)
                .clamp(-1.0, 1.0)
                .acos();
            ang < 1e-6 && (pose.translation - t_true).norm() < 1e-9
        });
        assert!(matched, "ground-truth pose missing from P3P candidates");
    }

    #[test]
    fn test_p3p_rejects_degenerate_input() {
        let m = model();
        // Duplicate object point -> zero-length side must be rejected.
        let obj = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];
        let img = [[100.0, 100.0], [100.0, 100.0], [300.0, 250.0]];
        assert!(PnpSolver::estimate_p3p(&obj, &img, &m).is_err());
    }
}

