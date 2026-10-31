//! RGBD Odometry
//!
//! Estimates camera motion from consecutive RGBD frames.
//! Implements multi-scale odometry with different loss functions.

use nalgebra::{Matrix4, Matrix6, Point3, Vector3, Vector6};
use rayon::prelude::*;

use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::RuntimeRunner;

/// RGBD Odometry result
#[derive(Debug, Clone)]
pub struct OdometryResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
}

/// Odometry method
#[derive(Debug, Clone, Copy)]
pub enum OdometryMethod {
    PointToPlane,
    Intensity,
    Hybrid,
}

/// Compute RGBD odometry between two frames using best available runner
#[allow(clippy::too_many_arguments)]
pub fn compute_rgbd_odometry(
    source_depth: &[f32],
    target_depth: &[f32],
    source_color: Option<&[Vector3<u8>]>,
    target_color: Option<&[Vector3<u8>]>,
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    method: OdometryMethod,
) -> Option<OdometryResult> {
    let runner = cv_runtime::best_runner().ok().map(|(r, _)| r)?;
    compute_rgbd_odometry_ctx(
        source_depth,
        target_depth,
        source_color,
        target_color,
        intrinsics,
        width,
        height,
        method,
        &runner,
    )
}

/// Compute RGBD odometry between two frames with explicit context
#[allow(clippy::too_many_arguments)]
pub fn compute_rgbd_odometry_ctx(
    source_depth: &[f32],
    target_depth: &[f32],
    source_color: Option<&[Vector3<u8>]>,
    target_color: Option<&[Vector3<u8>]>,
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    method: OdometryMethod,
    group: &RuntimeRunner,
) -> Option<OdometryResult> {
    // Multi-scale pyramid
    let transformation = Matrix4::identity();

    // GPU Path
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        let intrinsics_array = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy];

        // Convert color to u32 for HAL if provided
        let src_color_u32 = source_color.map(|c| {
            c.iter()
                .map(|v| (v.x as u32) | ((v.y as u32) << 8) | ((v.z as u32) << 16))
                .collect::<Vec<_>>()
        });
        let dst_color_u32 = target_color.map(|c| {
            c.iter()
                .map(|v| (v.x as u32) | ((v.y as u32) << 8) | ((v.z as u32) << 16))
                .collect::<Vec<_>>()
        });

        if let Ok((transform, fitness, rmse)) = cv_hal::gpu_kernels::odometry_gpu::compute_odometry(
            gpu,
            source_depth,
            target_depth,
            src_color_u32.as_deref(),
            dst_color_u32.as_deref(),
            &intrinsics_array,
            width as u32,
            height as u32,
            &transformation,
        ) {
            return Some(OdometryResult {
                transformation: transform,
                fitness,
                inlier_rmse: rmse,
            });
        }
    }

    // CPU Fallback (Rayon)
    // Multi-scale pyramid
    let scales = vec![0.125, 0.25, 0.5, 1.0];
    let mut transformation = Matrix4::identity();
    let mut solved = false;

    for scale in scales {
        let scaled_width = (width as f32 * scale) as usize;
        let scaled_height = (height as f32 * scale) as usize;

        // Create scaled intrinsics
        let scaled_intrinsics = crate::tsdf::CameraIntrinsics::new(
            intrinsics.fx * scale,
            intrinsics.fy * scale,
            intrinsics.cx * scale,
            intrinsics.cy * scale,
            scaled_width as u32,
            scaled_height as u32,
        );

        // Downsample depth images
        let source_scaled = downsample_depth(source_depth, width, height, scale);
        let target_scaled = downsample_depth(target_depth, width, height, scale);
        // Colours shrink with the frame. They used to be passed at full
        // resolution against scaled dimensions - the photometric paths then
        // indexed a full-res buffer with scaled strides, reading the wrong
        // pixels at every level but the finest.
        let source_color_scaled = source_color.map(|c| downsample_color(c, width, height, scale));
        let target_color_scaled = target_color.map(|c| downsample_color(c, width, height, scale));

        // Compute odometry at this scale
        let scale_result = match method {
            OdometryMethod::PointToPlane => compute_point_to_plane_ctx(
                &source_scaled,
                &target_scaled,
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
                group,
            ),
            OdometryMethod::Intensity => compute_intensity(
                &source_scaled,
                &target_scaled,
                source_color_scaled.as_deref(),
                target_color_scaled.as_deref(),
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
            ),
            OdometryMethod::Hybrid => compute_hybrid(
                &source_scaled,
                &target_scaled,
                source_color_scaled.as_deref(),
                target_color_scaled.as_deref(),
                &scaled_intrinsics,
                scaled_width,
                scaled_height,
                &transformation,
            ),
        };

        if let Some(result) = scale_result {
            transformation = result.transformation;
            solved = true;
        }
    }

    // No scale produced an answer. Returning `Some(identity)` here would hand
    // the caller a successful-looking result containing no information - the
    // exact failure `compute_hybrid` used to have, reintroduced at the
    // dispatch level. Nothing solved, nothing to report.
    if !solved {
        return None;
    }

    // Compute final fitness
    let (fitness, rmse) = evaluate_odometry_ctx(
        source_depth,
        target_depth,
        intrinsics,
        width,
        height,
        &transformation,
        group,
    );

    // The pyramid loop above is best-effort: each scale returns `None` on
    // failure and the previous transform is kept. If *every* scale failed, the
    // transformation handed on is the identity - not because the camera did not
    // move, but because nothing was ever estimated. That is exactly the shape
    // of failure this crate has been removing: `compute_hybrid` used to return
    // `Some` with the identity and a fabricated `fitness: 0.0`, and `fitness`
    // is also `0.0` here when the whole pyramid failed, so a caller reading the
    // score alone could not tell the two apart. Returning the identity *as a
    // success* is the wrong half of the answer either way.
    //
    // Measured: a source frame with valid depth against a target frame with none
    // returned `Some(OdometryResult { transformation: identity,
    // fitness: 0.0, inlier_rmse: 0.0 })` - a "result" in which the camera did
    // not move, nothing was matched, and the RMSE of an empty match set is
    // reported as a perfect 0.0. The honest signal is the one the signature
    // already has: `None`, matching `compute_intensity` and `compute_hybrid`.
    if fitness <= 0.0 {
        return None;
    }

    Some(OdometryResult {
        transformation,
        fitness,
        inlier_rmse: rmse,
    })
}

/// Point-to-plane odometry (6-DOF: translation + rotation)
fn compute_point_to_plane_ctx(
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    init_transform: &Matrix4<f32>,
    group: &RuntimeRunner,
) -> Option<OdometryResult> {
    let mut transformation = *init_transform;
    let max_iterations = 10;
    let mut prev_rmse = f32::MAX;

    // Precompute target vertex and normal maps
    let (target_vertices, target_normals) =
        compute_vertex_normal_map_ctx(target_depth, intrinsics, width, height, group);

    for _ in 0..max_iterations {
        // Build 6x6 linear system in parallel
        let (ata, atb, total_residual, valid_points) = group.run(|| {
            (0..height)
                .into_par_iter()
                .map(|v| {
                    let mut local_ata = Matrix6::<f32>::zeros();
                    let mut local_atb = Vector6::<f32>::zeros();
                    let mut local_residual = 0.0f32;
                    let mut local_valid = 0u32;

                    for u in 0..width {
                        let idx = v * width + u;
                        // The slices may be shorter than the declared frame.
                        let Some(&depth) = source_depth.get(idx) else {
                            continue;
                        };

                        if depth <= 0.0 {
                            continue;
                        }

                        // Backproject source point
                        let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                        let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                        let z = depth;
                        let source_point = Point3::new(x, y, z);

                        // Transform to target frame
                        let target_point = transformation.transform_point(&source_point);

                        // Project to target image
                        let tu = (target_point.x * intrinsics.fx / target_point.z + intrinsics.cx)
                            as i32;
                        let tv = (target_point.y * intrinsics.fy / target_point.z + intrinsics.cy)
                            as i32;

                        if tu < 0 || tu >= width as i32 || tv < 0 || tv >= height as i32 {
                            continue;
                        }

                        let tidx = (tv as usize) * width + (tu as usize);
                        let target_vertex = target_vertices[tidx];
                        let target_normal = target_normals[tidx];

                        if target_vertex.z <= 0.0 || target_normal.norm_squared() < 1e-6 {
                            continue;
                        }

                        // Point-to-plane error
                        let diff = target_point - target_vertex;
                        let residual = diff.dot(&target_normal);

                        // 6-DOF Jacobian: [n.x, n.y, n.z, (p x n).x, (p x n).y, (p x n).z]
                        // where p is the transformed source point and n is the target normal
                        let p = target_point.coords;
                        let n = target_normal;
                        let cross = p.cross(&n);

                        let jacobian = Vector6::new(n.x, n.y, n.z, cross.x, cross.y, cross.z);

                        local_ata += jacobian * jacobian.transpose();
                        local_atb += jacobian * residual;
                        local_residual += residual * residual;
                        local_valid += 1;
                    }
                    (local_ata, local_atb, local_residual, local_valid)
                })
                .reduce(
                    || (Matrix6::zeros(), Vector6::zeros(), 0.0f32, 0u32),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
                )
        });

        if valid_points < 100 {
            return None;
        }

        // Convergence check
        let rmse = (total_residual / valid_points as f32).sqrt();
        if (prev_rmse - rmse).abs() < 1e-6 {
            break;
        }
        prev_rmse = rmse;

        // Solve 6x6 system for [tx, ty, tz, rx, ry, rz]
        if let Some(ata_inv) = ata.try_inverse() {
            let delta = -(ata_inv * atb);

            // Apply update using SE(3) exponential map
            let update = exponential_map_se3(&delta);
            transformation = update * transformation;
        }
    }

    // Compute fitness and RMSE for the final transformation
    let (fitness, rmse) = compute_fitness_rmse(
        source_depth,
        &target_vertices,
        &target_normals,
        intrinsics,
        width,
        height,
        &transformation,
        group,
    );

    Some(OdometryResult {
        transformation,
        fitness,
        inlier_rmse: rmse,
    })
}

/// Compute fitness (fraction of valid correspondences) and point-to-plane RMSE
#[allow(clippy::too_many_arguments)]
fn compute_fitness_rmse(
    source_depth: &[f32],
    target_vertices: &[Point3<f32>],
    target_normals: &[Vector3<f32>],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    transformation: &Matrix4<f32>,
    group: &RuntimeRunner,
) -> (f32, f32) {
    let (total_error, valid, total_source) = group.run(|| {
        (0..height)
            .into_par_iter()
            .map(|v| {
                let mut err = 0.0f32;
                let mut valid_count = 0u32;
                let mut source_count = 0u32;

                for u in 0..width {
                    let idx = v * width + u;
                    // The slice may be shorter than the declared frame; the
                    // zero-dimension guard above does not cover that.
                    let Some(&depth) = source_depth.get(idx) else {
                        continue;
                    };
                    if depth <= 0.0 {
                        continue;
                    }
                    source_count += 1;

                    let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                    let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                    let tp = transformation.transform_point(&Point3::new(x, y, depth));

                    let tu = (tp.x * intrinsics.fx / tp.z + intrinsics.cx) as i32;
                    let tv = (tp.y * intrinsics.fy / tp.z + intrinsics.cy) as i32;

                    if tu < 0 || tu >= width as i32 || tv < 0 || tv >= height as i32 {
                        continue;
                    }
                    let tidx = tv as usize * width + tu as usize;
                    if target_vertices[tidx].z <= 0.0 || target_normals[tidx].norm_squared() < 1e-6
                    {
                        continue;
                    }

                    let diff = tp - target_vertices[tidx];
                    let r = diff.dot(&target_normals[tidx]);
                    err += r * r;
                    valid_count += 1;
                }
                (err, valid_count, source_count)
            })
            .reduce(
                || (0.0f32, 0u32, 0u32),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            )
    });

    if valid == 0 || total_source == 0 {
        return (0.0, 0.0);
    }
    let fitness = valid as f32 / total_source as f32;
    let rmse = (total_error / valid as f32).sqrt();
    (fitness, rmse)
}

/// Exponential map from se(3) to SE(3)
///
/// Maps a 6-vector [tx, ty, tz, rx, ry, rz] to a 4x4 homogeneous transformation
/// matrix using Rodrigues' formula for rotation and the left Jacobian for translation.
pub(crate) fn exponential_map_se3(delta: &Vector6<f32>) -> Matrix4<f32> {
    let omega = Vector3::new(delta[3], delta[4], delta[5]);
    let v = Vector3::new(delta[0], delta[1], delta[2]);

    let theta = omega.norm();

    let rotation = if theta < 1e-6 {
        nalgebra::Matrix3::identity()
    } else {
        let k = omega / theta;
        let k_cross = nalgebra::Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        nalgebra::Matrix3::identity()
            + k_cross * theta.sin()
            + k_cross * k_cross * (1.0 - theta.cos())
    };

    // Proper SE(3) exponential map using left Jacobian
    let translation = if theta < 1e-6 {
        v
    } else {
        let k = omega / theta;
        let k_cross = nalgebra::Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        let k_cross_sq = k_cross * k_cross;
        let left_jacobian = nalgebra::Matrix3::identity()
            + k_cross * ((1.0 - theta.cos()) / theta)
            + k_cross_sq * ((theta - theta.sin()) / theta);
        left_jacobian * v
    };

    let mut transform = Matrix4::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

/// Luminance of an RGB triple, the photometric signal direct alignment uses.
fn luminance(c: &Vector3<u8>) -> f32 {
    0.299 * c.x as f32 + 0.589 * c.y as f32 + 0.114 * c.z as f32
}

/// Bilinear sample of a single-channel image at floating coordinates;
/// `None` when the sample would touch outside the image.
fn sample_bilinear(img: &[f32], width: usize, height: usize, x: f32, y: f32) -> Option<f32> {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let v00 = img[y0 * width + x0];
    let v10 = img[y0 * width + x1];
    let v01 = img[y1 * width + x0];
    let v11 = img[y1 * width + x1];
    Some(
        v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy,
    )
}

/// Central-difference gradients of a single-channel image.
fn image_gradients(img: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; width * height];
    let mut gy = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(width - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(height - 1);
            gx[y * width + x] = 0.5 * (img[y * width + xp] - img[y * width + xm]);
            gy[y * width + x] = 0.5 * (img[yp * width + x] - img[ym * width + x]);
        }
    }
    (gx, gy)
}

/// Intensity (photometric) odometry: direct image alignment.
///
/// Residual per source pixel: `I_target(pi(T * backproject(u, d))) - I_source`,
/// minimized by the same Gauss-Newton / SE(3) loop shape as the point-to-plane
/// method. The pixel Jacobian chains the projection derivative into the target
/// intensity gradient. Requires colour on both sides - a colourless call is
/// not a photometric problem, so it is `None`, not a depth-only guess.
///
/// This used to be `None` unconditionally, so `OdometryMethod::Intensity` was
/// a public enum variant that could not succeed.
#[allow(clippy::too_many_arguments)]
fn compute_intensity(
    source_depth: &[f32],
    _target_depth: &[f32],
    source_color: Option<&[Vector3<u8>]>,
    target_color: Option<&[Vector3<u8>]>,
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    init_transform: &Matrix4<f32>,
) -> Option<OdometryResult> {
    let src = source_color?;
    let tgt = target_color?;
    if src.len() < width * height || tgt.len() < width * height {
        return None;
    }

    let target_i: Vec<f32> = tgt.iter().map(luminance).collect();
    let (gx, gy) = image_gradients(&target_i, width, height);
    let source_i: Vec<f32> = src.iter().map(luminance).collect();

    let mut transformation = *init_transform;
    let max_iterations = 50;
    let mut last_valid = 0usize;
    let mut last_rmse = 0f32;
    // Levenberg-Marquardt damping. Undamped Gauss-Newton overshoots on the
    // weakly-constrained directions a planar scene leaves (translation along
    // the plane and the rotation about the in-plane normal both shift the
    // image the same way to first order - the normal equations see a condition
    // number of ~4e4) and then *settles into a neighbouring local minimum*,
    // returning a wrong pose with a straight face. Damping with accept/reject
    // keeps every accepted step a decrease.
    let mut lambda = 1e-3_f32;

    for _ in 0..max_iterations {
        let mut ata = Matrix6::<f32>::zeros();
        let mut atb = Vector6::<f32>::zeros();
        let mut total_residual = 0.0f32;
        let mut valid = 0usize;

        for v in 0..height {
            for u in 0..width {
                let idx = v * width + u;
                let Some(&depth) = source_depth.get(idx) else {
                    continue;
                };
                if depth <= 0.0 {
                    continue;
                }

                let p = Point3::new(
                    (u as f32 - intrinsics.cx) * depth / intrinsics.fx,
                    (v as f32 - intrinsics.cy) * depth / intrinsics.fy,
                    depth,
                );
                let tp = transformation.transform_point(&p);
                if tp.z <= 1e-6 {
                    continue;
                }

                let su = tp.x * intrinsics.fx / tp.z + intrinsics.cx;
                let sv = tp.y * intrinsics.fy / tp.z + intrinsics.cy;
                let (Some(i_t), Some(didx), Some(didy)) = (
                    sample_bilinear(&target_i, width, height, su, sv),
                    sample_bilinear(&gx, width, height, su, sv),
                    sample_bilinear(&gy, width, height, su, sv),
                ) else {
                    continue;
                };

                let residual = i_t - source_i[idx];

                // d(pixel)/d(point): the projection's 2x3 derivative, chained
                // with the intensity gradient into grad_p = d(residual)/dP'.
                let (fx, fy, z) = (intrinsics.fx, intrinsics.fy, tp.z);
                let gux = didx * fx / z;
                let guy = didy * fy / z;
                let guz = -didx * fx * tp.x / (z * z) - didy * fy * tp.y / (z * z);
                let grad_p = Vector3::new(gux, guy, guz);
                // dP'/d(t) = I and dP'/d(omega) = -[P']x for a left-multiplied
                // increment delta = (t, omega), and with
                // a^T [b]x = (a x b)^T that leaves the omega block as
                // +(P' x grad_p)^T. (An earlier version carried a spurious
                // minus here and the solver walked away from the minimum.)
                let omega_block = tp.coords.cross(&grad_p);
                let jac = Vector6::new(
                    grad_p.x,
                    grad_p.y,
                    grad_p.z,
                    omega_block.x,
                    omega_block.y,
                    omega_block.z,
                );

                ata += jac * jac.transpose();
                atb += jac * residual;
                total_residual += residual * residual;
                valid += 1;
            }
        }

        if valid < 100 {
            return None;
        }
        last_valid = valid;
        last_rmse = (total_residual / valid as f32).sqrt();

        // Marquardt: damp along the diagonal, and accept the step only if it
        // actually decreases the cost.
        let mut damped = ata;
        for i in 0..6 {
            damped[(i, i)] += lambda * ata[(i, i)].max(1e-12);
        }
        let Some(damped_inv) = damped.try_inverse() else {
            break;
        };
        let delta = -(damped_inv * atb);
        let trial = exponential_map_se3(&delta) * transformation;

        let mut trial_residual = 0.0f32;
        let mut trial_valid = 0usize;
        for v in 0..height {
            for u in 0..width {
                let idx = v * width + u;
                let Some(&depth) = source_depth.get(idx) else {
                    continue;
                };
                if depth <= 0.0 {
                    continue;
                }
                let p = Point3::new(
                    (u as f32 - intrinsics.cx) * depth / intrinsics.fx,
                    (v as f32 - intrinsics.cy) * depth / intrinsics.fy,
                    depth,
                );
                let tp = trial.transform_point(&p);
                if tp.z <= 1e-6 {
                    continue;
                }
                let su = tp.x * intrinsics.fx / tp.z + intrinsics.cx;
                let sv = tp.y * intrinsics.fy / tp.z + intrinsics.cy;
                if let Some(i_t) = sample_bilinear(&target_i, width, height, su, sv) {
                    let r = i_t - source_i[idx];
                    trial_residual += r * r;
                    trial_valid += 1;
                }
            }
        }
        if trial_valid > 0 && trial_residual < total_residual {
            // Converged when an accepted step barely helps. A rejected step is
            // NOT convergence - the old check broke out whenever nothing
            // moved, which is what a run of rejections looks like.
            let improved = total_residual - trial_residual;
            transformation = trial;
            lambda = (lambda / 3.0).max(1e-9);
            if improved <= 1e-9 * total_residual.max(1.0) {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1e9);
            if lambda >= 1e9 {
                break;
            }
        }
    }

    Some(OdometryResult {
        transformation,
        fitness: last_valid as f32 / (width * height) as f32,
        inlier_rmse: last_rmse,
    })
}

/// Hybrid odometry (combines depth and intensity).
///
/// One Gauss-Newton system carries **both** residual kinds per source pixel:
/// the point-to-plane residual against the target's vertex/normal maps (in
/// metres) and the photometric residual (normalised by 255 into ~[0,1], a
/// relative weighting kept visible here rather than hidden in a magic
/// constant - tuning it is a parameter for later, not a correctness question).
/// Geometric rows need `target_depth`; photometric rows need both colours, so
/// a call missing either is not a hybrid problem and returns `None` rather
/// than quietly solving a weaker one.
///
/// This used to be `None` unconditionally (after first returning a fabricated
/// identity transform), so `OdometryMethod::Hybrid` was a public enum variant
/// that could not succeed.
#[allow(clippy::too_many_arguments)]
fn compute_hybrid(
    source_depth: &[f32],
    target_depth: &[f32],
    source_color: Option<&[Vector3<u8>]>,
    target_color: Option<&[Vector3<u8>]>,
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    init_transform: &Matrix4<f32>,
) -> Option<OdometryResult> {
    let src = source_color?;
    let tgt = target_color?;
    if src.len() < width * height || tgt.len() < width * height {
        return None;
    }

    let (target_vertices, target_normals) = compute_vertex_normal_map_ctx(
        target_depth,
        intrinsics,
        width,
        height,
        &RuntimeRunner::Sync(cv_runtime::registry().ok()?.default_cpu().id()),
    );
    let target_i: Vec<f32> = tgt.iter().map(luminance).collect();
    let (gx, gy) = image_gradients(&target_i, width, height);
    let source_i: Vec<f32> = src.iter().map(luminance).collect();
    const PHOTOMETRIC_SCALE: f32 = 1.0 / 255.0;

    let mut transformation = *init_transform;
    let max_iterations = 50;
    let mut last_valid = 0usize;
    let mut last_rmse = 0f32;
    // Marquardt damping for the same reason as `compute_intensity`: undamped
    // Gauss-Newton overshoots on weakly-constrained directions and settles
    // into a neighbouring minimum with a straight face.
    let mut lambda = 1e-3_f32;

    // Accumulate the combined normal equations at a given pose. Used for the
    // trial step too, so accept/reject measures the real cost.
    let accumulate = |t: &Matrix4<f32>| -> (Matrix6<f32>, Vector6<f32>, f32, usize) {
        let mut ata = Matrix6::<f32>::zeros();
        let mut atb = Vector6::<f32>::zeros();
        let mut total_residual = 0.0f32;
        let mut valid = 0usize;

        for v in 0..height {
            for u in 0..width {
                let idx = v * width + u;
                let Some(&depth) = source_depth.get(idx) else {
                    continue;
                };
                if depth <= 0.0 {
                    continue;
                }

                let p = Point3::new(
                    (u as f32 - intrinsics.cx) * depth / intrinsics.fx,
                    (v as f32 - intrinsics.cy) * depth / intrinsics.fy,
                    depth,
                );
                let tp = t.transform_point(&p);
                if tp.z <= 1e-6 {
                    continue;
                }

                let mut contributed = false;

                // Geometric row: point-to-plane against the target maps.
                let tu = (tp.x * intrinsics.fx / tp.z + intrinsics.cx) as i32;
                let tv = (tp.y * intrinsics.fy / tp.z + intrinsics.cy) as i32;
                if tu >= 0 && tu < width as i32 && tv >= 0 && tv < height as i32 {
                    let tidx = (tv as usize) * width + (tu as usize);
                    let (tvert, tnorm) = (target_vertices[tidx], target_normals[tidx]);
                    if tvert.z > 0.0 && tnorm.norm_squared() > 1e-6 {
                        let residual = (tp - tvert).dot(&tnorm);
                        let cross = tp.coords.cross(&tnorm);
                        let jac =
                            Vector6::new(tnorm.x, tnorm.y, tnorm.z, cross.x, cross.y, cross.z);
                        ata += jac * jac.transpose();
                        atb += jac * residual;
                        total_residual += residual * residual;
                        contributed = true;
                    }
                }

                // Photometric row: intensity at the projected sample.
                let su = tp.x * intrinsics.fx / tp.z + intrinsics.cx;
                let sv = tp.y * intrinsics.fy / tp.z + intrinsics.cy;
                if let (Some(i_t), Some(didx), Some(didy)) = (
                    sample_bilinear(&target_i, width, height, su, sv),
                    sample_bilinear(&gx, width, height, su, sv),
                    sample_bilinear(&gy, width, height, su, sv),
                ) {
                    let residual = (i_t - source_i[idx]) * PHOTOMETRIC_SCALE;
                    let (fx, fy, z) = (intrinsics.fx, intrinsics.fy, tp.z);
                    let gux = didx * fx / z;
                    let guy = didy * fy / z;
                    let guz = -didx * fx * tp.x / (z * z) - didy * fy * tp.y / (z * z);
                    let grad_p = Vector3::new(gux, guy, guz) * PHOTOMETRIC_SCALE;
                    let omega_block = tp.coords.cross(&grad_p);
                    let jac = Vector6::new(
                        grad_p.x,
                        grad_p.y,
                        grad_p.z,
                        omega_block.x,
                        omega_block.y,
                        omega_block.z,
                    );
                    ata += jac * jac.transpose();
                    atb += jac * residual;
                    contributed = true;
                }

                if contributed {
                    valid += 1;
                }
            }
        }
        (ata, atb, total_residual, valid)
    };

    for _ in 0..max_iterations {
        let (ata, atb, total_residual, valid) = accumulate(&transformation);
        if valid < 100 {
            return None;
        }
        last_valid = valid;
        last_rmse = (total_residual / valid as f32).sqrt();

        let mut damped = ata;
        for i in 0..6 {
            damped[(i, i)] += lambda * ata[(i, i)].max(1e-12);
        }
        let Some(damped_inv) = damped.try_inverse() else {
            break;
        };
        let delta = -(damped_inv * atb);
        let trial = exponential_map_se3(&delta) * transformation;

        let (_, _, trial_cost, trial_valid) = accumulate(&trial);
        if trial_valid > 0 && trial_cost < total_residual {
            // Converged when an accepted step barely helps. A rejected step is
            // NOT convergence - the old check broke out of the loop whenever
            // nothing moved, which is exactly what a run of rejections looks
            // like.
            let improved = total_residual - trial_cost;
            transformation = trial;
            lambda = (lambda / 3.0).max(1e-9);
            if improved <= 1e-9 * total_residual.max(1.0) {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1e9);
            if lambda >= 1e9 {
                break;
            }
        }
    }

    Some(OdometryResult {
        transformation,
        fitness: last_valid as f32 / (width * height) as f32,
        inlier_rmse: last_rmse,
    })
}

/// Downsample a colour image the way [`downsample_depth`] shrinks depth: box
/// average over each output pixel's source footprint.
fn downsample_color(
    input: &[Vector3<u8>],
    width: usize,
    height: usize,
    scale: f32,
) -> Vec<Vector3<u8>> {
    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    if new_width == 0 || new_height == 0 || input.len() < width * height {
        return Vec::new();
    }
    let mut output = vec![Vector3::new(0u8, 0, 0); new_width * new_height];
    let sx = width as f32 / new_width as f32;
    let sy = height as f32 / new_height as f32;
    for y in 0..new_height {
        for x in 0..new_width {
            let x0 = (x as f32 * sx) as usize;
            let x1 = (((x + 1) as f32 * sx) as usize).max(x0 + 1).min(width);
            let y0 = (y as f32 * sy) as usize;
            let y1 = (((y + 1) as f32 * sy) as usize).max(y0 + 1).min(height);
            let (mut r, mut g, mut b, mut n) = (0f32, 0f32, 0f32, 0f32);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let c = input[yy * width + xx];
                    r += c.x as f32;
                    g += c.y as f32;
                    b += c.z as f32;
                    n += 1.0;
                }
            }
            output[y * new_width + x] = Vector3::new(
                (r / n).round() as u8,
                (g / n).round() as u8,
                (b / n).round() as u8,
            );
        }
    }
    output
}

/// Downsample depth image
#[allow(clippy::needless_range_loop)]
fn downsample_depth(input: &[f32], width: usize, height: usize, scale: f32) -> Vec<f32> {
    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    let mut output = vec![0.0; new_width * new_height];

    // At coarse pyramid scales the downsampled frame can be empty (and the
    // source frame can be zero-sized); return early so `par_chunks_mut(0)`
    // cannot panic and the `- 1` clamps below cannot underflow.
    if new_width == 0 || new_height == 0 {
        return output;
    }

    output
        .par_chunks_mut(new_width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..new_width {
                let src_x = (x as f32 / scale) as usize;
                let src_y = (y as f32 / scale) as usize;
                let src_idx = (src_y.min(height.saturating_sub(1))) * width
                    + (src_x.min(width.saturating_sub(1)));
                // `src_idx` is clamped to `width * height`, but the slice may be
                // shorter than that: `compute_rgbd_odometry(&[1.0; 10], &[1.0;
                // 10], .., 64, 64, ..)` indexed element 2048 of a 10-element
                // slice and panicked. The zero-width guard further up does not
                // cover a non-zero frame with a short slice.
                if src_idx >= input.len() {
                    row[x] = 0.0;
                } else {
                    row[x] = input[src_idx];
                }
            }
        });

    output
}

/// Compute vertex and normal maps from depth with explicit context
fn compute_vertex_normal_map_ctx(
    depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    group: &RuntimeRunner,
) -> (Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    let mut vertices = vec![Point3::origin(); width * height];
    let mut normals = vec![Vector3::zeros(); width * height];

    // A zero-dimension frame has no vertices to differentiate, and
    // `par_chunks_mut(0)` panics outright with "chunk_size must not be zero".
    //
    // This is reachable from the public entry point for any frame narrower than
    // 8 pixels: `compute_rgbd_odometry_ctx` runs a four-level pyramid whose
    // coarsest level is `scale = 0.125`, so a 7x7 frame downsamples to
    // `(7*0.125) as usize = 0` columns and every call panicked in a rayon
    // worker. A 1x1 frame and a declared 0x0 frame panic identically. The
    // `length_checks` regression test covers a *short slice*, not a *zero
    // dimension* — the two are different guards.
    if width == 0 || height == 0 {
        return (vertices, normals);
    }

    group.run(|| {
        // Compute vertices in parallel
        vertices.par_iter_mut().enumerate().for_each(|(idx, v)| {
            let x = (idx % width) as f32;
            let y = (idx / width) as f32;
            // A depth slice shorter than width*height used to index past the
            // end here; in a closure that means leaving the vertex at its
            // default rather than aborting the caller.
            let z = depth.get(idx).copied().unwrap_or(0.0);

            if z > 0.0 {
                let vx = (x - intrinsics.cx) * z / intrinsics.fx;
                let vy = (y - intrinsics.cy) * z / intrinsics.fy;
                *v = Point3::new(vx, vy, z);
            }
        });

        // Compute normals using central differences in parallel
        let vertices_ref = &vertices;
        normals
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y == height - 1 {
                    return;
                }
                for x in 1..width - 1 {
                    let idx = y * width + x;

                    let left = vertices_ref[idx - 1];
                    let right = vertices_ref[idx + 1];
                    let up = vertices_ref[(y - 1) * width + x];
                    let down = vertices_ref[(y + 1) * width + x];

                    if left.z > 0.0 && right.z > 0.0 && up.z > 0.0 && down.z > 0.0 {
                        let dx = right.coords - left.coords;
                        let dy = down.coords - up.coords;
                        row[x] = dx.cross(&dy).normalize();
                    }
                }
            });
    });

    (vertices, normals)
}

/// Evaluate odometry quality with explicit context
fn evaluate_odometry_ctx(
    source_depth: &[f32],
    target_depth: &[f32],
    intrinsics: &crate::tsdf::CameraIntrinsics,
    width: usize,
    height: usize,
    transformation: &Matrix4<f32>,
    group: &RuntimeRunner,
) -> (f32, f32) {
    let (total_error, valid_points) = group.run(|| {
        (0..height)
            .into_par_iter()
            .map(|v| {
                let mut local_error = 0.0;
                let mut local_valid = 0;
                for u in 0..width {
                    let idx = v * width + u;
                    // The slices may be shorter than the declared frame.
                    let Some(&depth) = source_depth.get(idx) else {
                        continue;
                    };

                    if depth <= 0.0 {
                        continue;
                    }

                    let x = (u as f32 - intrinsics.cx) * depth / intrinsics.fx;
                    let y = (v as f32 - intrinsics.cy) * depth / intrinsics.fy;
                    let z = depth;
                    let point = Point3::new(x, y, z);

                    let transformed = transformation.transform_point(&point);
                    let tu = (transformed.x * intrinsics.fx / transformed.z + intrinsics.cx) as i32;
                    let tv = (transformed.y * intrinsics.fy / transformed.z + intrinsics.cy) as i32;

                    if tu >= 0 && tu < width as i32 && tv >= 0 && tv < height as i32 {
                        let tidx = (tv as usize) * width + (tu as usize);
                        let Some(&target_z) = target_depth.get(tidx) else {
                            continue;
                        };

                        if target_z > 0.0 {
                            let error = (transformed.z - target_z).abs();
                            local_error += error * error;
                            local_valid += 1;
                        }
                    }
                }
                (local_error, local_valid)
            })
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    });

    if valid_points == 0 {
        return (0.0, 0.0);
    }

    let rmse = (total_error / valid_points as f32).sqrt();
    // Denominator matches compute_fitness_rmse: pixels with valid SOURCE
    // depth, not all pixels — otherwise the same scene reports materially
    // different fitness depending on which path ran.
    let total_source = source_depth.iter().filter(|&&d| d > 0.0).count() as f32;
    let fitness = if total_source > 0.0 {
        valid_points as f32 / total_source
    } else {
        0.0
    };

    (fitness, rmse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_depth_zero_sized_frame() {
        // Empty and zero-sized frames must not panic (previously underflowed
        // on `height - 1` / `par_chunks_mut(0)`).
        assert!(downsample_depth(&[], 0, 0, 0.125).is_empty());
        assert!(downsample_depth(&[1.0, 2.0, 3.0, 4.0], 2, 2, 0.125).is_empty());
    }

    #[test]
    fn test_downsample_depth_basic() {
        // 4x4 frame at scale 0.5 -> 2x2, nearest-neighbour sampling.
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = downsample_depth(&input, 4, 4, 0.5);
        assert_eq!(out, vec![0.0, 2.0, 8.0, 10.0]);
    }
}
