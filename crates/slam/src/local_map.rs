//! Map maintenance shared by both tracking modes: matching map points to a
//! frame by projection, pose refinement on those matches, the keyframe policy,
//! and monocular map extension by triangulation between keyframes.

use crate::mapping::MapExt;
use crate::types::{MapPoint, WorldMap};
use cv_calib3d::{solve_pnp_refine, triangulate_points};
use cv_core::{CameraIntrinsics, Pose, TypedPose};
use cv_features::Descriptors;
use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, UnitQuaternion, Vector3};

/// Largest Hamming distance (of 256 bits) accepted for a projection match.
pub(crate) const MAX_PROJECTION_HAMMING: u32 = 64;
/// Largest Hamming distance accepted when matching two keyframes for
/// triangulation. Tighter than for tracking: a wrong pair here becomes a wrong
/// map point that every later frame is pulled towards.
const MAX_TRIANGULATION_HAMMING: u32 = 50;
/// Ratio for keyframe-to-keyframe matching along the epipolar line.
const TRIANGULATION_RATIO: f64 = 0.8;
/// Epipolar distance (pixels) for a keyframe-to-keyframe candidate.
const EPIPOLAR_PX: f64 = 2.0;
/// Reprojection error (pixels) a new map point may have in either keyframe.
const TRIANGULATION_REPROJ_PX: f64 = 2.0;
/// Minimum angle between the two viewing rays of a new map point. Below this
/// the depth is too poorly constrained to be worth adding.
const MIN_TRIANGULATION_PARALLAX_DEG: f64 = 1.0;

pub(crate) fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Pixel projection of a world point, or `None` behind the camera.
pub(crate) fn project(k: &CameraIntrinsics, pose: &Pose, p: &Vector3<f64>) -> Option<Point2<f64>> {
    let c = pose.rotation * p + pose.translation;
    if c.z <= 1e-9 {
        return None;
    }
    Some(Point2::new(
        k.fx * c.x / c.z + k.cx,
        k.fy * c.y / c.z + k.cy,
    ))
}

fn reprojection_px(k: &CameraIntrinsics, pose: &Pose, p: &Vector3<f64>, obs: &Point2<f64>) -> f64 {
    project(k, pose, p).map_or(f64::INFINITY, |q| (q - obs).norm())
}

/// World -> camera pose of a keyframe (whose pose is stored camera -> world).
pub(crate) fn keyframe_world_to_camera(p: &TypedPose) -> Pose {
    let r_wc = p.rotation.cast::<f64>();
    let c = p.position.cast::<f64>().coords;
    let r_cw = r_wc.transpose();
    Pose::new(r_cw, -(r_cw * c))
}

/// Match map points to a frame's descriptors by projecting them with `pose`
/// and searching a `radius`-pixel window around each projection.
///
/// This is how a tracked frame finds the map: the motion prediction (or a
/// coarse pose) says where each point should be, so a point only competes
/// with the handful of features near its projection. Matching against the
/// whole map with a global ratio test instead rejects every point that has a
/// near-duplicate anywhere in the map - and a map that grows by keyframes has
/// many, since neighbouring keyframes see the same surfaces.
///
/// Returns `(map index, descriptor index)`; each descriptor is used at most
/// once (the closest map point keeps it).
pub(crate) fn match_by_projection(
    k: &CameraIntrinsics,
    descriptors: &Descriptors,
    map: &WorldMap,
    pose: &Pose,
    radius: f64,
) -> Vec<(usize, usize)> {
    let r2 = radius * radius;
    let kps: Vec<Point2<f64>> = descriptors
        .descriptors
        .iter()
        .map(|d| Point2::new(d.keypoint.x, d.keypoint.y))
        .collect();
    let mut best_for_query: Vec<Option<(u32, usize)>> = vec![None; kps.len()];
    for (mi, p) in map.points.iter().enumerate() {
        let p = p.read().unwrap_or_else(|e| e.into_inner());
        let Some(uv) = project(k, pose, &p.world_pos.coords.cast::<f64>()) else {
            continue;
        };
        // No ratio test inside the window: ORB detects the same corner at
        // several pyramid levels, so the runner-up is usually the same corner
        // and a ratio of 0.9 rejected ~30% of correct matches (measured on
        // the RGB-D scene: 102 -> 146 matches for 401 points). The window and
        // the pose refinement's reprojection test do that job instead.
        let mut best = (u32::MAX, usize::MAX);
        for (qi, kp) in kps.iter().enumerate() {
            if (kp - uv).norm_squared() > r2 {
                continue;
            }
            let d = hamming(&p.descriptor, &descriptors.descriptors[qi].data);
            if d < best.0 {
                best = (d, qi);
            }
        }
        if best.0 > MAX_PROJECTION_HAMMING {
            continue;
        }
        let slot = &mut best_for_query[best.1];
        if slot.map_or(true, |(d, _)| best.0 < d) {
            *slot = Some((best.0, mi));
        }
    }
    best_for_query
        .iter()
        .enumerate()
        .filter_map(|(qi, s)| s.map(|(_, mi)| (mi, qi)))
        .collect()
}

/// Levenberg-Marquardt pose refinement with inlier re-selection: refine on
/// the points within `threshold_px`, re-select, repeat. Returns the pose and a
/// per-correspondence inlier flag.
pub(crate) fn refine_pose(
    k: &CameraIntrinsics,
    initial: &Pose,
    object: &[Point3<f64>],
    image: &[Point2<f64>],
    threshold_px: f64,
) -> (Pose, Vec<bool>) {
    let inliers_of = |pose: &Pose| -> Vec<bool> {
        object
            .iter()
            .zip(image)
            .map(|(o, i)| reprojection_px(k, pose, &o.coords, i) < threshold_px)
            .collect()
    };
    let mut pose = *initial;
    let mut inliers = inliers_of(&pose);
    for _ in 0..3 {
        let (o, i): (Vec<Point3<f64>>, Vec<Point2<f64>>) = object
            .iter()
            .zip(image)
            .zip(&inliers)
            .filter(|(_, &ok)| ok)
            .map(|((o, i), _)| (*o, *i))
            .unzip();
        if o.len() < 6 {
            break;
        }
        match solve_pnp_refine(&pose, &o, &i, k, None, 20) {
            Ok(p) => pose = p,
            Err(_) => break,
        }
        let next = inliers_of(&pose);
        if next == inliers {
            break;
        }
        inliers = next;
    }
    (pose, inliers)
}

/// Median depth of the map points a keyframe observes, in its own camera.
fn keyframe_median_depth(map: &WorldMap, kf_index: usize) -> Option<f64> {
    let kf = map.keyframes[kf_index]
        .read()
        .unwrap_or_else(|e| e.into_inner());
    let w2c = keyframe_world_to_camera(&kf.pose);
    let mut depths: Vec<f64> = kf
        .map_points
        .iter()
        .flatten()
        .filter_map(|&id| map.points.get(id as usize))
        .map(|p| {
            let p = p.read().unwrap_or_else(|e| e.into_inner());
            (w2c.rotation * p.world_pos.coords.cast::<f64>() + w2c.translation).z
        })
        .filter(|z| *z > 0.0)
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.total_cmp(b));
    Some(depths[depths.len() / 2])
}

/// How far a tracked frame has moved from the last keyframe, and how much of
/// that keyframe's map it still tracks.
pub(crate) struct KeyframeDistance {
    /// Established map points the last keyframe observes: those seen by at
    /// least one other keyframe too (ORB-SLAM's reference count). A point a
    /// keyframe has just back-projected from depth has not yet been matched
    /// in any other view, and many never will be (weak corners, the same
    /// corner at several pyramid levels), so counting them makes a well
    /// tracked frame look like it lost most of the map. If the keyframe has
    /// none (the first one), every point it observes counts.
    pub reference: usize,
    /// Camera-centre distance to the last keyframe over its median scene
    /// depth: scale-free, so the same policy works for monocular maps.
    pub baseline_ratio: f64,
    /// Rotation from the last keyframe, degrees.
    pub rotation_deg: f64,
}

pub(crate) fn distance_to_last_keyframe(map: &WorldMap, pose: &Pose) -> Option<KeyframeDistance> {
    let last = map.keyframes.len().checked_sub(1)?;
    let median = keyframe_median_depth(map, last)?;
    let kf = map.keyframes[last]
        .read()
        .unwrap_or_else(|e| e.into_inner());
    let ids: Vec<u64> = kf.map_points.iter().flatten().copied().collect();
    let established = ids
        .iter()
        .filter_map(|&id| map.points.get(id as usize))
        .filter(|p| {
            p.read()
                .unwrap_or_else(|e| e.into_inner())
                .observations
                .len()
                >= 2
        })
        .count();
    let reference = if established > 0 {
        established
    } else {
        ids.len()
    };
    let kf_c2w_r = kf.pose.rotation.cast::<f64>();
    let kf_c = kf.pose.position.cast::<f64>().coords;
    let c2w = pose.inverse();
    let rel = UnitQuaternion::from_matrix(&(kf_c2w_r.transpose() * c2w.rotation_matrix()));
    Some(KeyframeDistance {
        reference,
        baseline_ratio: (c2w.translation - kf_c).norm() / median,
        rotation_deg: rel.angle().to_degrees(),
    })
}

/// Sampson distance of `x2ᵀ E x1 = 0` for normalised homogeneous points.
fn sampson(e: &Matrix3<f64>, a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let ea = e * a;
    let etb = e.transpose() * b;
    let num = b.dot(&ea).powi(2);
    let den = ea.x * ea.x + ea.y * ea.y + etb.x * etb.x + etb.y * etb.y;
    if den > 0.0 {
        num / den
    } else {
        f64::INFINITY
    }
}

fn skew(t: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
}

fn projection_matrix(pose: &Pose) -> Matrix3x4<f64> {
    let mut p = Matrix3x4::zeros();
    p.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&pose.rotation_matrix());
    p.fixed_view_mut::<3, 1>(0, 3).copy_from(&pose.translation);
    p
}

/// Monocular map extension: triangulate new map points between the newest
/// keyframe and up to `neighbours` keyframes before it, from features that
/// neither keyframe has linked to a map point yet. Returns the map indices of
/// the created points.
///
/// Candidates are searched along the epipolar line (both poses are known), so
/// each feature competes only with features that are geometrically possible;
/// the best Hamming match must pass a ratio test against the second best on
/// the line. A pair becomes a map point only if its triangulation lies in
/// front of both cameras, reprojects within [`TRIANGULATION_REPROJ_PX`] in
/// both, and its rays meet at more than [`MIN_TRIANGULATION_PARALLAX_DEG`].
pub(crate) fn triangulate_new_points(
    k: &CameraIntrinsics,
    map: &mut WorldMap,
    neighbours: usize,
) -> Vec<usize> {
    let Some(new_idx) = map.keyframes.len().checked_sub(1) else {
        return Vec::new();
    };
    let f = 0.5 * (k.fx + k.fy);
    let epi_thr = (EPIPOLAR_PX / f).powi(2);
    let norm = |x: f64, y: f64| Vector3::new((x - k.cx) / k.fx, (y - k.cy) / k.fy, 1.0);

    let new_kf = std::sync::Arc::clone(&map.keyframes[new_idx]);
    let mut created = Vec::new();
    for old_idx in (new_idx.saturating_sub(neighbours)..new_idx).rev() {
        let old_kf = std::sync::Arc::clone(&map.keyframes[old_idx]);
        let Some(old_depth) = keyframe_median_depth(map, old_idx) else {
            continue;
        };
        let (pairs, new_pose, old_pose, new_id, old_id) = {
            let new = new_kf.read().unwrap_or_else(|e| e.into_inner());
            let old = old_kf.read().unwrap_or_else(|e| e.into_inner());
            let new_pose = keyframe_world_to_camera(&new.pose);
            let old_pose = keyframe_world_to_camera(&old.pose);
            // Too short a baseline for this pair to triangulate anything.
            let baseline = (new.pose.position - old.pose.position).norm() as f64;
            if baseline / old_depth < 0.01 {
                continue;
            }
            // Relative pose old -> new, and its essential matrix.
            let r = new_pose.rotation_matrix() * old_pose.rotation_matrix().transpose();
            let t = new_pose.translation - r * old_pose.translation;
            let e = skew(&t) * r;

            let free = |m: &Vec<Option<u64>>, i: usize| m.get(i).map_or(false, |x| x.is_none());
            let old_free: Vec<(usize, Vector3<f64>)> = old
                .descriptors
                .descriptors
                .iter()
                .enumerate()
                .filter(|(i, _)| free(&old.map_points, *i))
                .map(|(i, d)| (i, norm(d.keypoint.x, d.keypoint.y)))
                .collect();
            // Best old feature for each new one; then each old feature is
            // kept by its closest new feature.
            let mut best_for_old: Vec<Option<(u32, usize)>> =
                vec![None; old.descriptors.descriptors.len()];
            for (qi, dq) in new.descriptors.descriptors.iter().enumerate() {
                if !free(&new.map_points, qi) {
                    continue;
                }
                let xq = norm(dq.keypoint.x, dq.keypoint.y);
                let mut best = (u32::MAX, usize::MAX);
                let mut second = u32::MAX;
                for (ti, xt) in &old_free {
                    if sampson(&e, xt, &xq) > epi_thr {
                        continue;
                    }
                    let d = hamming(&dq.data, &old.descriptors.descriptors[*ti].data);
                    if d < best.0 {
                        second = best.0;
                        best = (d, *ti);
                    } else if d < second {
                        second = d;
                    }
                }
                if best.0 > MAX_TRIANGULATION_HAMMING {
                    continue;
                }
                if second != u32::MAX && best.0 as f64 > TRIANGULATION_RATIO * second as f64 {
                    continue;
                }
                let slot = &mut best_for_old[best.1];
                if slot.map_or(true, |(d, _)| best.0 < d) {
                    *slot = Some((best.0, qi));
                }
            }
            let pairs: Vec<(usize, usize)> = best_for_old
                .iter()
                .enumerate()
                .filter_map(|(ti, s)| s.map(|(_, qi)| (ti, qi)))
                .collect();
            (pairs, new_pose, old_pose, new.id, old.id)
        };
        if pairs.is_empty() {
            continue;
        }

        let (pts_old, pts_new, n_old, n_new, desc) = {
            let new = new_kf.read().unwrap_or_else(|e| e.into_inner());
            let old = old_kf.read().unwrap_or_else(|e| e.into_inner());
            let px = |d: &Descriptors, i: usize| {
                let kp = d.descriptors[i].keypoint;
                Point2::new(kp.x, kp.y)
            };
            let pts_old: Vec<Point2<f64>> = pairs
                .iter()
                .map(|&(ti, _)| px(&old.descriptors, ti))
                .collect();
            let pts_new: Vec<Point2<f64>> = pairs
                .iter()
                .map(|&(_, qi)| px(&new.descriptors, qi))
                .collect();
            let to_n = |p: &Point2<f64>| Point2::new((p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy);
            let n_old: Vec<Point2<f64>> = pts_old.iter().map(to_n).collect();
            let n_new: Vec<Point2<f64>> = pts_new.iter().map(to_n).collect();
            let desc: Vec<Vec<u8>> = pairs
                .iter()
                .map(|&(_, qi)| new.descriptors.descriptors[qi].data.clone())
                .collect();
            (pts_old, pts_new, n_old, n_new, desc)
        };
        let Ok(tri) = triangulate_points(
            &projection_matrix(&old_pose),
            &projection_matrix(&new_pose),
            &n_old,
            &n_new,
        ) else {
            continue;
        };

        let c_old = old_pose.inverse().translation;
        let c_new = new_pose.inverse().translation;
        let cos_min = MIN_TRIANGULATION_PARALLAX_DEG.to_radians().cos();
        let mut new_w = new_kf.write().unwrap_or_else(|e| e.into_inner());
        let mut old_w = old_kf.write().unwrap_or_else(|e| e.into_inner());
        for (j, x) in tri.iter().enumerate() {
            let x = x.coords;
            if !(x.x.is_finite() && x.y.is_finite() && x.z.is_finite()) {
                continue;
            }
            if reprojection_px(k, &old_pose, &x, &pts_old[j]) > TRIANGULATION_REPROJ_PX
                || reprojection_px(k, &new_pose, &x, &pts_new[j]) > TRIANGULATION_REPROJ_PX
            {
                // Also rejects points behind either camera.
                continue;
            }
            let (a, b) = (x - c_old, x - c_new);
            if a.dot(&b) / (a.norm() * b.norm()) > cos_min {
                continue;
            }
            let (ti, qi) = pairs[j];
            let idx = map.points.len();
            let mut mp = MapPoint::new(idx as u64, Point3::from(x.cast::<f32>()), desc[j].clone());
            mp.observations.push((old_id, ti));
            mp.observations.push((new_id, qi));
            map.add_point(mp);
            old_w.map_points[ti] = Some(idx as u64);
            new_w.map_points[qi] = Some(idx as u64);
            created.push(idx);
        }
    }
    created
}
