//! Local bundle adjustment over the newest keyframes.
//!
//! Not `cv_sfm::bundle_adjust`: that holds only the first camera fixed (its
//! rigid gauge), leaves the scale free, and optimises every other camera it is
//! given, so it cannot hold a whole set of out-of-window keyframes fixed as a
//! sliding window needs. Here the newest keyframes and
//! the points they observe are optimised, and every other keyframe that sees
//! those points constrains them with its pose held fixed - the usual local BA.
//!
//! The solver is Levenberg-Marquardt on the reduced camera system (Schur
//! complement of the block-diagonal point part), with a Huber kernel, so it is
//! cheap for a handful of keyframes and a few thousand observations.
//!
//! A point back-projected from a depth image also contributes its creating
//! keyframe's depth (see [`crate::types::MapPoint::depth_m`]). Reprojection
//! alone cannot see scale, and on an RGB-D map it slid keyframes off the
//! points that depth had just placed.

use crate::types::WorldMap;
use cv_core::{CameraIntrinsics, FrameConvention, Pose, TypedPose};
use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x6, Matrix6, Matrix6x3, Point2, Point3, Rotation3,
    Vector2, Vector3, Vector6,
};
use std::collections::HashMap;

/// Huber threshold (pixels): the 95% chi-square bound for 2 DOF at 1 px noise.
const HUBER_PX: f64 = 2.447;
/// An observation further than this from its projection after the adjustment
/// is unlinked from its keyframe.
const OUTLIER_PX: f64 = 2.447;

/// What one adjustment did.
#[derive(Debug, Clone)]
pub struct LocalBaReport {
    pub optimised_keyframes: usize,
    pub fixed_keyframes: usize,
    pub points: usize,
    pub observations: usize,
    /// RMS reprojection error (pixels) before and after.
    pub rms_before: f64,
    pub rms_after: f64,
    /// Observations unlinked as outliers afterwards.
    pub outliers_removed: usize,
}

struct Obs {
    /// Slot in `poses`.
    cam: usize,
    /// Slot in `points`.
    point: usize,
    uv: Point2<f64>,
    kf_id: u64,
    kp_idx: usize,
}

/// A depth measurement in the keyframe that created the point.
struct DepthRef {
    cam: usize,
    point: usize,
    z_meas: f64,
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn world_to_camera(p: &TypedPose) -> Pose {
    let r_wc = p.rotation.cast::<f64>();
    let r_cw = r_wc.transpose();
    Pose::new(r_cw, -(r_cw * p.position.cast::<f64>().coords))
}

fn typed_pose(w2c: &Pose) -> TypedPose {
    let c2w = w2c.inverse();
    TypedPose::new(
        Point3::from(c2w.translation.cast::<f32>()),
        c2w.rotation_matrix().cast::<f32>(),
        FrameConvention::opencv(),
    )
}

/// Residual (projection - observation) and camera-frame point, or `None`
/// behind the camera.
fn residual(
    k: &CameraIntrinsics,
    pose: &Pose,
    x: &Vector3<f64>,
    uv: &Point2<f64>,
) -> Option<(Vector2<f64>, Vector3<f64>)> {
    let c = pose.rotation * x + pose.translation;
    if c.z <= 1e-9 {
        return None;
    }
    let u = k.fx * c.x / c.z + k.cx;
    let v = k.fy * c.y / c.z + k.cy;
    Some((Vector2::new(u - uv.x, v - uv.y), c))
}

fn huber_weight(r: f64) -> f64 {
    if r <= HUBER_PX { 1.0 } else { HUBER_PX / r }
}

fn scalar_robust(r: f64) -> f64 {
    let n = r.abs();
    if n <= HUBER_PX {
        r * r
    } else {
        2.0 * HUBER_PX * n - HUBER_PX * HUBER_PX
    }
}

/// Depth of `x` in `pose` (world -> camera), as a residual in pixels.
///
/// `(z - z_meas) * fx / z_meas` is the pixel-sized form of a metre error, so
/// it shares the reprojection Huber threshold. The gradients match the camera
/// update used for reprojection (`R <- exp(w) R`, `t <- t + v`):
/// `dz/dw = (rx.y, -rx.x, 0)` and `dz/dv = (0, 0, 1)`, with `rx = R x`.
/// Returns `(residual, d(residual)/d(point), d(residual)/d(camera))`.
fn depth_jac(
    k: &CameraIntrinsics,
    pose: &Pose,
    x: &Vector3<f64>,
    z_meas: f64,
) -> Option<(f64, Vector3<f64>, Vector6<f64>)> {
    if !(z_meas > 1e-6) {
        return None;
    }
    let r_mat = pose.rotation_matrix();
    let rx = r_mat * x;
    let z = rx.z + pose.translation.z;
    if z <= 1e-9 {
        return None;
    }
    let scale = k.fx / z_meas;
    let g_point = scale * r_mat.row(2).transpose();
    let g_cam = scale * Vector6::new(rx.y, -rx.x, 0.0, 0.0, 0.0, 1.0);
    Some(((z - z_meas) * scale, g_point, g_cam))
}

fn depth_penalty(k: &CameraIntrinsics, pose: &Pose, x: &Vector3<f64>, z_meas: f64) -> f64 {
    match depth_jac(k, pose, x, z_meas) {
        Some((r, _, _)) => scalar_robust(r),
        None => 1e6,
    }
}

fn robust_cost(
    k: &CameraIntrinsics,
    poses: &[Pose],
    pts: &[Vector3<f64>],
    obs: &[Obs],
    depths: &[DepthRef],
) -> f64 {
    let reproj: f64 = obs
        .iter()
        .map(|o| match residual(k, &poses[o.cam], &pts[o.point], &o.uv) {
            Some((r, _)) => {
                let n = r.norm();
                if n <= HUBER_PX {
                    n * n
                } else {
                    2.0 * HUBER_PX * n - HUBER_PX * HUBER_PX
                }
            }
            None => 1e6,
        })
        .sum();
    let depth: f64 = depths
        .iter()
        .map(|d| depth_penalty(k, &poses[d.cam], &pts[d.point], d.z_meas))
        .sum();
    reproj + depth
}

/// True when an optimised keyframe will back-project a once-seen depth.
///
/// That point is not in the normal equations. Rescaling the keyframe and
/// then placing the point at the old depth puts the measurement on the
/// rescaled ray, so the depth counts as a scale pin even though it is not
/// a variable.
fn once_seen_depth_on_optimised_camera(map: &WorldMap, kf_ids: &[u64], fixed: &[bool]) -> bool {
    for (c, &kf_id) in kf_ids.iter().enumerate() {
        if fixed[c] {
            continue;
        }
        let Some(kf) = map.keyframes.get(kf_id as usize) else {
            continue;
        };
        let kf = kf.read().unwrap_or_else(|e| e.into_inner());
        for p in &map.points {
            let p = p.read().unwrap_or_else(|e| e.into_inner());
            let Some(z) = p.depth_m else {
                continue;
            };
            let z = z as f64;
            if !(z.is_finite() && z > 1e-6) || p.born_kf != kf_id || p.observations.len() != 1 {
                continue;
            }
            let (obs_kf, kp_idx) = p.observations[0];
            if obs_kf == kf_id && kf.descriptors.descriptors.get(kp_idx).is_some() {
                return true;
            }
        }
    }
    false
}

fn rms(k: &CameraIntrinsics, poses: &[Pose], pts: &[Vector3<f64>], obs: &[Obs]) -> f64 {
    let (s, n) = obs.iter().fold((0.0, 0usize), |(s, n), o| {
        match residual(k, &poses[o.cam], &pts[o.point], &o.uv) {
            Some((r, _)) => (s + r.norm_squared(), n + 1),
            None => (s, n),
        }
    });
    if n == 0 { 0.0 } else { (s / n as f64).sqrt() }
}

/// Adjust the newest `window` keyframes and every map point they observe.
///
/// Keyframes outside the window that observe those points are held fixed, and
/// so is keyframe 0 (the world origin); if fewer than `min_fixed` (at least
/// one) are, the oldest window keyframes are fixed too. One fixed keyframe
/// fixes the rigid gauge; a monocular map also has a scale gauge, which is
/// then removed after the solve by a similarity about the fixed keyframe that
/// keeps its distance to the oldest optimised keyframe (reprojections are
/// invariant to it). That lets the initialisation pair itself be refined,
/// rather than freezing its distortion by fixing both. A depth measurement
/// removes that freedom instead: the similarity is not applied when any point
/// in the window has one. A multi-view depth is a residual, and rescaling
/// would throw away the metric scale it recovered. A depth seen only once is
/// not a variable (it would only repeat its keyframe) but on an optimised
/// keyframe it is still a measurement, so the similarity is not applied then
/// either. Otherwise that pose is rescaled and the point is back-projected at
/// the measured depth on the rescaled ray. After the poses are written the
/// point is placed on the ray of its keypoint at the measured depth, so a
/// moved keyframe does not leave it behind and it does not drag the keyframe.
/// Afterwards observations further than [`OUTLIER_PX`]
/// from their projection are unlinked from both the keyframe and the point.
pub fn local_bundle_adjust(
    k: &CameraIntrinsics,
    map: &mut WorldMap,
    window: usize,
    min_fixed: usize,
    iterations: usize,
) -> Option<LocalBaReport> {
    let n_kf = map.keyframes.len();
    if n_kf < 2 {
        return None;
    }
    let first_local = n_kf.saturating_sub(window);

    // Points observed by the window keyframes.
    let mut point_slot: HashMap<usize, usize> = HashMap::new();
    let mut point_ids: Vec<usize> = Vec::new();
    for kf in &map.keyframes[first_local..] {
        let kf = kf.read().unwrap_or_else(|e| e.into_inner());
        for id in kf.map_points.iter().flatten() {
            let id = *id as usize;
            if id >= map.points.len() || point_slot.contains_key(&id) {
                continue;
            }
            // A point seen once has an unconstrained depth. A depth image
            // fixes that, but the point is then determined by its keyframe, so
            // it is placed from the adjusted pose afterwards rather than
            // pulling the keyframe toward wherever it currently sits.
            let seen = map.points[id]
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .observations
                .len();
            if seen >= 2 {
                point_slot.insert(id, point_ids.len());
                point_ids.push(id);
            }
        }
    }
    if point_ids.is_empty() {
        return None;
    }

    // Every observation of those points, and the keyframes they come from.
    let mut kf_slot: HashMap<u64, usize> = HashMap::new();
    let mut kf_ids: Vec<u64> = Vec::new();
    let mut poses: Vec<Pose> = Vec::new();
    let mut obs: Vec<Obs> = Vec::new();
    let mut pts: Vec<Vector3<f64>> = Vec::with_capacity(point_ids.len());
    for (slot, &pid) in point_ids.iter().enumerate() {
        let p = map.points[pid].read().unwrap_or_else(|e| e.into_inner());
        pts.push(p.world_pos.coords.cast::<f64>());
        for &(kf_id, kp_idx) in &p.observations {
            let Some(kf) = map.keyframes.get(kf_id as usize) else {
                continue;
            };
            let cam = match kf_slot.get(&kf_id) {
                Some(&c) => c,
                None => {
                    let kf = kf.read().unwrap_or_else(|e| e.into_inner());
                    kf_slot.insert(kf_id, kf_ids.len());
                    kf_ids.push(kf_id);
                    poses.push(world_to_camera(&kf.pose));
                    kf_ids.len() - 1
                }
            };
            let kf = kf.read().unwrap_or_else(|e| e.into_inner());
            let Some(d) = kf.descriptors.descriptors.get(kp_idx) else {
                continue;
            };
            obs.push(Obs {
                cam,
                point: slot,
                uv: Point2::new(d.keypoint.x, d.keypoint.y),
                kf_id,
                kp_idx,
            });
        }
    }

    // Depth of each point in the keyframe that created it. `born_kf` is that
    // keyframe; it is in `kf_slot` only while the creating observation remains.
    let mut depths: Vec<DepthRef> = Vec::new();
    for (slot, &pid) in point_ids.iter().enumerate() {
        let p = map.points[pid].read().unwrap_or_else(|e| e.into_inner());
        let Some(z) = p.depth_m else { continue };
        let z = z as f64;
        if !(z.is_finite() && z > 1e-6) {
            continue;
        }
        let Some(&cam) = kf_slot.get(&p.born_kf) else {
            continue;
        };
        depths.push(DepthRef {
            cam,
            point: slot,
            z_meas: z,
        });
    }

    // Which cameras move. Keyframe ids are their indices in `map.keyframes`.
    let mut fixed: Vec<bool> = kf_ids
        .iter()
        .map(|&id| (id as usize) < first_local || id == 0)
        .collect();
    let mut by_age: Vec<usize> = (0..kf_ids.len()).collect();
    by_age.sort_by_key(|&c| kf_ids[c]);
    for &c in &by_age {
        if fixed.iter().filter(|&&f| f).count() >= min_fixed.max(1) {
            break;
        }
        fixed[c] = true;
    }
    let mut cam_param: Vec<Option<usize>> = vec![None; kf_ids.len()];
    let mut n_opt = 0;
    for c in 0..kf_ids.len() {
        if !fixed[c] {
            cam_param[c] = Some(n_opt);
            n_opt += 1;
        }
    }
    if n_opt == 0 {
        return None;
    }

    // Observations of each point, for the Schur complement. Kept as an index
    // list so a map with no depth runs the same reductions as before.
    let mut obs_of_point: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for (i, o) in obs.iter().enumerate() {
        obs_of_point[o.point].push(i);
    }
    let mut depth_of_point: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for (di, d) in depths.iter().enumerate() {
        depth_of_point[d.point].push(di);
    }

    let rms_before = rms(k, &poses, &pts, &obs);
    let poses_before = poses.clone();
    let mut cost = robust_cost(k, &poses, &pts, &obs, &depths);
    let mut lambda = 1e-3;
    let nc = 6 * n_opt;

    for _ in 0..iterations {
        // Per-observation Jacobians and weighted residuals.
        let mut h_cc = DMatrix::<f64>::zeros(nc, nc);
        let mut b_c = DVector::<f64>::zeros(nc);
        let mut v_p = vec![Matrix3::<f64>::zeros(); pts.len()];
        let mut b_p = vec![Vector3::<f64>::zeros(); pts.len()];
        // W blocks, one per observation of an optimised camera.
        let mut w_blocks: Vec<Option<Matrix6x3<f64>>> = vec![None; obs.len()];
        // Depth cross terms, optimised cameras only, parallel to `depths`.
        let mut depth_w: Vec<Option<(usize, Matrix6x3<f64>)>> = vec![None; depths.len()];
        for (i, o) in obs.iter().enumerate() {
            let pose = &poses[o.cam];
            let Some((r, c)) = residual(k, pose, &pts[o.point], &o.uv) else {
                continue;
            };
            let w = huber_weight(r.norm());
            let jp = Matrix2x3::new(
                k.fx / c.z,
                0.0,
                -k.fx * c.x / (c.z * c.z),
                0.0,
                k.fy / c.z,
                -k.fy * c.y / (c.z * c.z),
            );
            let r_mat = pose.rotation_matrix();
            let j_point = jp * r_mat;
            v_p[o.point] += w * j_point.transpose() * j_point;
            b_p[o.point] -= w * j_point.transpose() * r;
            if let Some(ci) = cam_param[o.cam] {
                // Camera update: R <- exp(w) R, t <- t + v.
                let rx = r_mat * pts[o.point];
                let mut dc = Matrix3x6::<f64>::zeros();
                dc.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew(&rx)));
                dc.fixed_view_mut::<3, 3>(0, 3)
                    .copy_from(&Matrix3::identity());
                let j_cam = jp * dc;
                let hcc: Matrix6<f64> = w * j_cam.transpose() * j_cam;
                let bc: Vector6<f64> = -w * j_cam.transpose() * r;
                let mut blk = h_cc.view_mut((6 * ci, 6 * ci), (6, 6));
                blk += hcc;
                let mut bblk = b_c.rows_mut(6 * ci, 6);
                bblk += bc;
                w_blocks[i] = Some(w * j_cam.transpose() * j_point);
            }
        }
        for (di, d) in depths.iter().enumerate() {
            let Some((r, g_point, g_cam)) = depth_jac(k, &poses[d.cam], &pts[d.point], d.z_meas)
            else {
                continue;
            };
            let w = huber_weight(r.abs());
            v_p[d.point] += w * g_point * g_point.transpose();
            b_p[d.point] -= w * g_point * r;
            if let Some(ci) = cam_param[d.cam] {
                let hcc: Matrix6<f64> = w * g_cam * g_cam.transpose();
                let bc: Vector6<f64> = -w * g_cam * r;
                let mut blk = h_cc.view_mut((6 * ci, 6 * ci), (6, 6));
                blk += hcc;
                let mut bblk = b_c.rows_mut(6 * ci, 6);
                bblk += bc;
                depth_w[di] = Some((ci, w * g_cam * g_point.transpose()));
            }
        }

        // Damped Schur complement.
        let mut s = h_cc.clone();
        for d in 0..nc {
            s[(d, d)] *= 1.0 + lambda;
            s[(d, d)] += 1e-9;
        }
        let mut rhs = b_c.clone();
        let mut v_inv = vec![Matrix3::<f64>::zeros(); pts.len()];
        for p in 0..pts.len() {
            let mut v = v_p[p];
            for d in 0..3 {
                v[(d, d)] = v[(d, d)] * (1.0 + lambda) + 1e-9;
            }
            let Some(vi) = v.try_inverse() else {
                continue;
            };
            v_inv[p] = vi;
            let ob = &obs_of_point[p];
            for &a in ob {
                let Some(wa) = w_blocks[a] else { continue };
                let ca = cam_param[obs[a].cam].expect("w block implies optimised camera");
                let wa_vi = wa * vi;
                let mut r = rhs.rows_mut(6 * ca, 6);
                r -= wa_vi * b_p[p];
                for &b in ob {
                    let Some(wb) = w_blocks[b] else { continue };
                    let cb = cam_param[obs[b].cam].expect("w block implies optimised camera");
                    let mut blk = s.view_mut((6 * ca, 6 * cb), (6, 6));
                    blk -= wa_vi * wb.transpose();
                }
                for &di in &depth_of_point[p] {
                    let Some((cb, wb)) = depth_w[di] else {
                        continue;
                    };
                    let mut blk = s.view_mut((6 * ca, 6 * cb), (6, 6));
                    blk -= wa_vi * wb.transpose();
                }
            }
            for &di in &depth_of_point[p] {
                let Some((ca, wa)) = depth_w[di] else {
                    continue;
                };
                let wa_vi = wa * vi;
                let mut r = rhs.rows_mut(6 * ca, 6);
                r -= wa_vi * b_p[p];
                for &b in ob {
                    let Some(wb) = w_blocks[b] else { continue };
                    let cb = cam_param[obs[b].cam].expect("w block implies optimised camera");
                    let mut blk = s.view_mut((6 * ca, 6 * cb), (6, 6));
                    blk -= wa_vi * wb.transpose();
                }
                for &dj in &depth_of_point[p] {
                    let Some((cb, wb)) = depth_w[dj] else {
                        continue;
                    };
                    let mut blk = s.view_mut((6 * ca, 6 * cb), (6, 6));
                    blk -= wa_vi * wb.transpose();
                }
            }
        }
        let Some(dc) = s
            .clone()
            .cholesky()
            .map(|c| c.solve(&rhs))
            .or_else(|| s.lu().solve(&rhs))
        else {
            lambda *= 10.0;
            continue;
        };

        // Back-substitute the points.
        let mut new_pts = pts.clone();
        for p in 0..pts.len() {
            let mut rhs_p = b_p[p];
            for &a in &obs_of_point[p] {
                if let Some(wa) = w_blocks[a] {
                    let ca = cam_param[obs[a].cam].expect("optimised");
                    let dca: Vector6<f64> = dc.fixed_rows::<6>(6 * ca).into_owned();
                    rhs_p -= wa.transpose() * dca;
                }
            }
            for &di in &depth_of_point[p] {
                if let Some((ca, wa)) = depth_w[di] {
                    let dca: Vector6<f64> = dc.fixed_rows::<6>(6 * ca).into_owned();
                    rhs_p -= wa.transpose() * dca;
                }
            }
            new_pts[p] += v_inv[p] * rhs_p;
        }
        let mut new_poses = poses.clone();
        for (c, pose) in new_poses.iter_mut().enumerate() {
            if let Some(ci) = cam_param[c] {
                let w = Vector3::new(dc[6 * ci], dc[6 * ci + 1], dc[6 * ci + 2]);
                let v = Vector3::new(dc[6 * ci + 3], dc[6 * ci + 4], dc[6 * ci + 5]);
                let r = Rotation3::from_scaled_axis(w).into_inner() * pose.rotation_matrix();
                *pose = Pose::new(r, pose.translation + v);
            }
        }
        let new_cost = robust_cost(k, &new_poses, &new_pts, &obs, &depths);
        if new_cost < cost {
            let converged = (cost - new_cost) < 1e-6 * cost;
            poses = new_poses;
            pts = new_pts;
            cost = new_cost;
            lambda = (lambda / 10.0).max(1e-9);
            if converged {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    // With a single fixed keyframe and no depth, the solution is only defined
    // up to a scale about that keyframe. Remove that freedom exactly: a
    // similarity about the fixed camera centre leaves every reprojection
    // unchanged, so rescale to keep the distance from it to the oldest
    // optimised keyframe (for the initialisation pair, the unit baseline).
    // A multi-view depth residual already fixes that scale. A once-seen
    // depth is not a residual, but the back-projection below uses the pose
    // written here; rescaling first would place the point at the old z on
    // the rescaled ray.
    let scale_pinned =
        !depths.is_empty() || once_seen_depth_on_optimised_camera(map, &kf_ids, &fixed);
    if !scale_pinned && fixed.iter().filter(|&&f| f).count() == 1 {
        let f = fixed.iter().position(|&f| f).expect("one fixed");
        let r = (0..kf_ids.len())
            .filter(|&c| !fixed[c])
            .min_by_key(|&c| kf_ids[c])
            .expect("n_opt > 0");
        let centre = |p: &Pose| p.inverse().translation;
        let c0 = centre(&poses[f]);
        let before = (centre(&poses_before[r]) - centre(&poses_before[f])).norm();
        let after = (centre(&poses[r]) - c0).norm();
        if before > 0.0 && after > 0.0 {
            let s = before / after;
            for (c, pose) in poses.iter_mut().enumerate() {
                if !fixed[c] {
                    let cc = c0 + s * (centre(pose) - c0);
                    *pose = Pose::from_quat_translation(pose.rotation, -(pose.rotation * cc));
                }
            }
            for p in pts.iter_mut() {
                *p = c0 + s * (*p - c0);
            }
        }
    }
    let rms_after = rms(k, &poses, &pts, &obs);

    // Write back.
    for (c, &kf_id) in kf_ids.iter().enumerate() {
        if cam_param[c].is_some() {
            let mut kf = map.keyframes[kf_id as usize]
                .write()
                .unwrap_or_else(|e| e.into_inner());
            kf.pose = typed_pose(&poses[c]);
        }
    }
    for (slot, &pid) in point_ids.iter().enumerate() {
        let mut p = map.points[pid].write().unwrap_or_else(|e| e.into_inner());
        p.world_pos = Point3::from(pts[slot].cast::<f32>());
    }
    // A depth point with one observation is the back-projection of its
    // keypoint. The solve does not include it (it would only repeat its
    // keyframe's pose), so place it from the pose just written.
    for (c, &kf_id) in kf_ids.iter().enumerate() {
        if cam_param[c].is_none() {
            continue;
        }
        let c2w = poses[c].inverse();
        let kf = map.keyframes[kf_id as usize]
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mut place: Vec<(usize, f64, f64, f64)> = Vec::new();
        for (i, p) in map.points.iter().enumerate() {
            let p = p.read().unwrap_or_else(|e| e.into_inner());
            let Some(z) = p.depth_m else { continue };
            let z = z as f64;
            if !(z.is_finite() && z > 1e-6) || p.born_kf != kf_id || p.observations.len() != 1 {
                continue;
            }
            let (obs_kf, kp_idx) = p.observations[0];
            if obs_kf != kf_id {
                continue;
            }
            let Some(d) = kf.descriptors.descriptors.get(kp_idx) else {
                continue;
            };
            place.push((i, d.keypoint.x, d.keypoint.y, z));
        }
        drop(kf);
        for (i, u, v, z) in place {
            let x_cam = Point3::new((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z);
            map.points[i]
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .world_pos = c2w.transform_point(&x_cam).cast::<f32>();
        }
    }
    let mut outliers_removed = 0;
    for o in &obs {
        let bad = residual(k, &poses[o.cam], &pts[o.point], &o.uv)
            .is_none_or(|(r, _)| r.norm() > OUTLIER_PX);
        if !bad {
            continue;
        }
        outliers_removed += 1;
        let pid = point_ids[o.point];
        map.points[pid]
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .observations
            .retain(|&(kf, i)| !(kf == o.kf_id && i == o.kp_idx));
        let mut kf = map.keyframes[o.kf_id as usize]
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if kf.map_points.get(o.kp_idx).copied().flatten() == Some(pid as u64) {
            kf.map_points[o.kp_idx] = None;
        }
    }

    Some(LocalBaReport {
        optimised_keyframes: n_opt,
        fixed_keyframes: fixed.iter().filter(|&&f| f).count(),
        points: pts.len(),
        observations: obs.len(),
        rms_before,
        rms_after,
        outliers_removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::MapExt;
    use crate::types::{KeyFrame, MapPoint};
    use cv_core::{Descriptor, Descriptors, KeyPoint, KeyPoints};
    use nalgebra::UnitQuaternion;
    use std::sync::{Arc, RwLock};

    fn k() -> CameraIntrinsics {
        CameraIntrinsics::new(300.0, 300.0, 160.0, 120.0, 320, 240)
    }

    /// Camera centred at `c`, looking down +z, turned by `yaw` radians.
    fn pose(c: Vector3<f64>, yaw: f64) -> Pose {
        let r_cw = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0).inverse();
        Pose::from_quat_translation(r_cw, -(r_cw * c))
    }

    /// Four keyframes along x observing a cloud of points, with exact
    /// observations. Returns the map and the true poses and points.
    fn scene() -> (WorldMap, Vec<Pose>, Vec<Vector3<f64>>) {
        let k = k();
        let truth: Vec<Pose> = (0..4)
            .map(|i| pose(Vector3::new(0.3 * i as f64, 0.0, 0.0), 0.02 * i as f64))
            .collect();
        let points: Vec<Vector3<f64>> = (0..150)
            .map(|i| {
                let a = i as f64;
                Vector3::new(
                    -1.5 + 4.0 * ((a * 0.618) % 1.0),
                    -1.0 + 2.0 * ((a * 0.414) % 1.0),
                    3.0 + 2.0 * ((a * 0.732) % 1.0),
                )
            })
            .collect();
        let mut map = WorldMap::new();
        for (pi, p) in points.iter().enumerate() {
            map.add_point(MapPoint::new(
                pi as u64,
                Point3::from(p.cast::<f32>()),
                vec![0; 32],
            ));
        }
        for (ci, cam) in truth.iter().enumerate() {
            let mut descs = Descriptors::new();
            let mut ids = Vec::new();
            for (pi, p) in points.iter().enumerate() {
                let c = cam.rotation * p + cam.translation;
                let (u, v) = (k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy);
                if c.z > 0.0 && (0.0..320.0).contains(&u) && (0.0..240.0).contains(&v) {
                    descs.push(Descriptor::new(vec![0; 32], KeyPoint::new(u, v)));
                    ids.push(pi);
                }
            }
            let kps = KeyPoints {
                keypoints: descs.descriptors.iter().map(|d| d.keypoint).collect(),
            };
            let mut kf = KeyFrame::new(ci as u64, typed_pose(cam), kps, descs);
            for (kp_idx, &pi) in ids.iter().enumerate() {
                kf.map_points[kp_idx] = Some(pi as u64);
                map.points[pi]
                    .write()
                    .unwrap()
                    .observations
                    .push((ci as u64, kp_idx));
            }
            map.keyframes.push(Arc::new(RwLock::new(kf)));
        }
        (map, truth, points)
    }

    fn kf_pose(map: &WorldMap, i: usize) -> Pose {
        world_to_camera(&map.keyframes[i].read().unwrap().pose)
    }

    fn nudge(map: &mut WorldMap, i: usize, rot: [f64; 3], dt: Vector3<f64>) {
        let p = kf_pose(map, i);
        let q = UnitQuaternion::from_euler_angles(rot[0], rot[1], rot[2]) * p.rotation;
        map.keyframes[i].write().unwrap().pose =
            typed_pose(&Pose::from_quat_translation(q, p.translation + dt));
    }

    /// Perturbed keyframes and points converge back onto exact observations;
    /// keyframe 0 does not move, and neither does the scale (the distance from
    /// keyframe 0 to keyframe 1).
    #[test]
    fn recovers_a_perturbed_window_and_holds_the_gauge() {
        let (mut map, truth, points) = scene();
        for i in 1..4 {
            let s = i as f64;
            nudge(
                &mut map,
                i,
                [0.004 * s, -0.003, 0.002],
                Vector3::new(0.02, -0.015, 0.01 * s),
            );
        }
        for (pi, p) in map.points.iter().enumerate() {
            let off = Vector3::new(0.01, -0.02, 0.03) * (((pi % 7) as f64) - 3.0) / 3.0;
            p.write().unwrap().world_pos = Point3::from((points[pi] + off).cast::<f32>());
        }
        let baseline_before = kf_pose(&map, 1).inverse().translation.norm();

        let report = local_bundle_adjust(&k(), &mut map, 4, 1, 30).expect("adjusted");
        assert_eq!(report.fixed_keyframes, 1);
        assert_eq!(report.optimised_keyframes, 3);
        assert!(
            report.rms_before > 1.0,
            "perturbation too small: {report:?}"
        );
        assert!(report.rms_after < 0.05, "did not converge: {report:?}");
        assert_eq!(report.outliers_removed, 0);

        let p0 = kf_pose(&map, 0);
        assert!((p0.translation - truth[0].translation).norm() < 1e-6);
        let baseline_after = kf_pose(&map, 1).inverse().translation.norm();
        assert!(
            (baseline_after - baseline_before).abs() < 1e-4,
            "scale moved: baseline {baseline_before} -> {baseline_after}"
        );
    }

    /// With a second fixed keyframe the scale is pinned by the data, and the
    /// true poses are recovered.
    #[test]
    fn two_fixed_keyframes_recover_the_true_poses() {
        let (mut map, truth, _) = scene();
        for i in 2..4 {
            nudge(
                &mut map,
                i,
                [0.003, 0.004, -0.002],
                Vector3::new(-0.02, 0.01, 0.03),
            );
        }
        let report = local_bundle_adjust(&k(), &mut map, 4, 2, 30).expect("adjusted");
        assert_eq!(report.fixed_keyframes, 2);
        assert!(report.rms_after < 0.05, "did not converge: {report:?}");
        for (i, gt) in truth.iter().enumerate() {
            let p = kf_pose(&map, i);
            let terr = (p.inverse().translation - gt.inverse().translation).norm();
            let rerr = (p.rotation.inverse() * gt.rotation).angle().to_degrees();
            assert!(
                terr < 1e-3 && rerr < 0.01,
                "keyframe {i}: {terr} m, {rerr} deg"
            );
        }
    }

    /// Scale every optimised camera and every point about keyframe 0.
    /// Reprojection is unchanged; a depth measurement is not.
    fn scale_about_origin(map: &mut WorldMap, s: f64) {
        let origin = kf_pose(map, 0).inverse().translation;
        for p in &map.points {
            let mut p = p.write().unwrap();
            let x = p.world_pos.coords.cast::<f64>();
            p.world_pos = Point3::from((origin + s * (x - origin)).cast::<f32>());
        }
        for i in 1..map.keyframes.len() {
            let pose = kf_pose(map, i);
            let centre = pose.inverse().translation;
            let new_c = origin + s * (centre - origin);
            let scaled = Pose::from_quat_translation(pose.rotation, -(pose.rotation * new_c));
            map.keyframes[i].write().unwrap().pose = typed_pose(&scaled);
        }
    }

    fn stamp_depth_in_keyframe(map: &mut WorldMap, kf: u64, points: &[Vector3<f64>], cam: &Pose) {
        for (pi, p) in points.iter().enumerate() {
            let mut mp = map.points[pi].write().unwrap();
            if !mp.observations.iter().any(|o| o.0 == kf) {
                continue;
            }
            let c = cam.rotation * p + cam.translation;
            mp.depth_m = Some(c.z as f32);
            mp.born_kf = kf;
        }
    }

    /// A 2% scale leaves every reprojection exactly where it was, so monocular
    /// adjustment would keep it (the gauge removal restores the *scaled*
    /// baseline). Depth in an optimised keyframe is what pulls the metric
    /// scale back.
    #[test]
    fn depth_recovers_a_scale_reprojection_cannot_see() {
        let (mut map, truth, points) = scene();
        stamp_depth_in_keyframe(&mut map, 1, &points, &truth[1]);
        let s = 1.02;
        scale_about_origin(&mut map, s);
        let scaled = kf_pose(&map, 1).inverse().translation.norm();
        let true_base = truth[1].inverse().translation.norm();
        assert!(
            (scaled / true_base - s).abs() < 1e-3,
            "scale did not take: {scaled} vs {}",
            s * true_base
        );

        let report = local_bundle_adjust(&k(), &mut map, 4, 1, 40).expect("adjusted");
        assert!(report.rms_after < 0.05, "did not converge: {report:?}");
        for (i, gt) in truth.iter().enumerate() {
            let got = kf_pose(&map, i).inverse().translation;
            let terr = (got - gt.inverse().translation).norm();
            assert!(
                terr < 0.005,
                "keyframe {i} stayed at the scaled pose: {terr} m"
            );
        }
    }

    /// A point with one observation is free along its ray under reprojection.
    /// With a depth measurement it is in the solve, and a point left off its
    /// ray is pulled back. Dropping once-seen points is what left fresh RGB-D
    /// landmarks behind their keyframe.
    #[test]
    fn depth_point_seen_once_returns_to_its_measurement() {
        let (mut map, truth, points) = scene();
        stamp_depth_in_keyframe(&mut map, 1, &points, &truth[1]);
        let pi = (0..points.len())
            .find(|&pi| {
                map.points[pi]
                    .read()
                    .unwrap()
                    .observations
                    .iter()
                    .any(|o| o.0 == 1)
            })
            .expect("keyframe 1 sees a point");
        let keep = map.points[pi]
            .read()
            .unwrap()
            .observations
            .iter()
            .copied()
            .find(|o| o.0 == 1)
            .unwrap();
        let others: Vec<(u64, usize)> = map.points[pi]
            .read()
            .unwrap()
            .observations
            .iter()
            .copied()
            .filter(|o| o.0 != 1)
            .collect();
        for (kf, kp) in others {
            let mut kfw = map.keyframes[kf as usize].write().unwrap();
            if kfw.map_points.get(kp).copied().flatten() == Some(pi as u64) {
                kfw.map_points[kp] = None;
            }
        }
        {
            let mut p = map.points[pi].write().unwrap();
            p.observations.clear();
            p.observations.push(keep);
            p.world_pos =
                Point3::from((points[pi] + Vector3::new(0.01, -0.02, 0.05)).cast::<f32>());
        }

        let report = local_bundle_adjust(&k(), &mut map, 4, 1, 40).expect("adjusted");
        assert!(report.rms_after < 0.05, "did not converge: {report:?}");
        let got = map.points[pi]
            .read()
            .unwrap()
            .world_pos
            .coords
            .cast::<f64>();
        let err = (got - points[pi]).norm();
        assert!(
            err < 0.005,
            "once-seen depth point was left {err} m from its measurement"
        );
        for (i, gt) in truth.iter().enumerate() {
            let terr = (kf_pose(&map, i).inverse().translation - gt.inverse().translation).norm();
            assert!(terr < 0.005, "keyframe {i} moved {terr} m");
        }
    }

    /// The only depth in the window belongs to a point seen once, on an
    /// optimised keyframe. The same pose perturbation without a depth
    /// restores the pre-solve baseline (`recovers_a_perturbed_window_and_holds_the_gauge`).
    /// This measurement must block that similarity, and the point is then
    /// placed on the unscaled keyframe's ray at the measured depth.
    #[test]
    fn once_seen_depth_blocks_the_monocular_similarity() {
        let (mut map, truth, points) = scene();
        for i in 1..4 {
            let s = i as f64;
            nudge(
                &mut map,
                i,
                [0.004 * s, -0.003, 0.002],
                Vector3::new(0.02, -0.015, 0.01 * s),
            );
        }
        for (pi, p) in map.points.iter().enumerate() {
            let off = Vector3::new(0.01, -0.02, 0.03) * (((pi % 7) as f64) - 3.0) / 3.0;
            p.write().unwrap().world_pos = Point3::from((points[pi] + off).cast::<f32>());
        }

        let pi = (0..points.len())
            .find(|&pi| {
                let obs = &map.points[pi].read().unwrap().observations;
                obs.iter().any(|o| o.0 == 1) && obs.len() >= 2
            })
            .expect("keyframe 1 sees a multi-view point");
        let keep = map.points[pi]
            .read()
            .unwrap()
            .observations
            .iter()
            .copied()
            .find(|o| o.0 == 1)
            .unwrap();
        let others: Vec<(u64, usize)> = map.points[pi]
            .read()
            .unwrap()
            .observations
            .iter()
            .copied()
            .filter(|o| *o != keep)
            .collect();
        for (kf, kp) in others {
            let mut kfw = map.keyframes[kf as usize].write().unwrap();
            if kfw.map_points.get(kp).copied().flatten() == Some(pi as u64) {
                kfw.map_points[kp] = None;
            }
        }
        let intr = k();
        let z_meas = {
            let cam = &truth[1];
            (cam.rotation * points[pi] + cam.translation).z
        };
        let (u_obs, v_obs) = {
            let kf = map.keyframes[1].read().unwrap();
            let d = &kf.descriptors.descriptors[keep.1];
            (d.keypoint.x, d.keypoint.y)
        };
        {
            let mut p = map.points[pi].write().unwrap();
            p.observations.clear();
            p.observations.push(keep);
            p.depth_m = Some(z_meas as f32);
            p.born_kf = 1;
        }
        assert!(
            map.points.iter().enumerate().all(|(i, p)| {
                let p = p.read().unwrap();
                i == pi || p.depth_m.is_none()
            }),
            "the window's only depth must be the once-seen point"
        );

        let baseline_before = kf_pose(&map, 1).inverse().translation.norm();
        let report = local_bundle_adjust(&intr, &mut map, 4, 1, 30).expect("adjusted");
        assert!(report.rms_after < 0.05, "did not converge: {report:?}");
        let baseline_after = kf_pose(&map, 1).inverse().translation.norm();
        assert!(
            (baseline_after - baseline_before).abs() > 1e-3,
            "similarity restored the monocular baseline {baseline_before} -> {baseline_after}"
        );

        let got = map.points[pi]
            .read()
            .unwrap()
            .world_pos
            .coords
            .cast::<f64>();
        let cam = kf_pose(&map, 1);
        let c = cam.rotation * got + cam.translation;
        assert!(
            (c.z - z_meas).abs() < 1e-3,
            "back-projected depth {} != measured {z_meas}",
            c.z
        );
        let u = intr.fx * c.x / c.z + intr.cx;
        let v = intr.fy * c.y / c.z + intr.cy;
        assert!(
            (u - u_obs).abs() < 1e-3 && (v - v_obs).abs() < 1e-3,
            "point left the keypoint ray: ({u}, {v}) vs ({u_obs}, {v_obs})"
        );
    }
}
