//! Map maintenance shared by both tracking modes: matching map points to a
//! frame by projection, pose refinement on those matches, the keyframe policy,
//! monocular map extension by triangulation between keyframes, and fusion and
//! culling of the points that produces.

use crate::mapping::MapExt;
use crate::types::{MapPoint, WorldMap};
use cv_calib3d::{solve_pnp_refine, triangulate_points};
use cv_core::{CameraIntrinsics, Pose, TypedPose};
use cv_features::Descriptors;
use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, UnitQuaternion, Vector3};
use std::collections::HashMap;

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
/// Projection window in which a map point and a keyframe feature are the same
/// corner. Wider than the triangulation check: fusion runs after the pose is
/// known, and a duplicate's own error is what it is absorbing.
const FUSE_PX: f64 = 3.0;
/// Largest Hamming distance at which fusion treats two descriptors as the same
/// corner. The triangulation gate, which is the same decision made earlier.
const FUSE_HAMMING: u32 = 50;
/// Keyframes a point is given, after the one it was born in, to be seen again.
/// A point that still has only its creating observations by then is removed.
const CULL_MIN_AGE: u64 = 2;

pub(crate) fn hamming(a: &[u8], b: &[u8]) -> u32 {
    // Same sentinel as `cv_core::Descriptor::hamming_distance`. `zip` stops at
    // the shorter slice, so an 8-byte descriptor against a 32-byte one can
    // report 0 and pass the fusion and triangulation caps. ORB descriptors
    // are 32 bytes; this only fires when lengths are mixed. Building a
    // `Descriptor` to call that method would copy both slices on the match
    // path, so the check is inlined.
    if a.len() != b.len() {
        return u32::MAX;
    }
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
    // Cell size is the search radius. A disk of that radius overlaps the
    // projection's cell and its neighbours; one further cell covers a bin
    // that rounded across a boundary. The radius test still drops anything
    // outside the disk, so those extra cells are not Hamming'd. An equal
    // Hamming keeps the lower keypoint index, which is what a forward scan
    // kept.
    let grid = (radius.is_finite() && radius > 0.0).then(|| {
        let mut bins: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (qi, kp) in kps.iter().enumerate() {
            bins.entry(projection_bin(kp.x, kp.y, radius))
                .or_default()
                .push(qi);
        }
        bins
    });
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
        if let Some(grid) = &grid {
            let (cx, cy) = projection_bin(uv.x, uv.y, radius);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let key = (cx.saturating_add(dx), cy.saturating_add(dy));
                    let Some(cell) = grid.get(&key) else {
                        continue;
                    };
                    for &qi in cell {
                        consider_projection_candidate(
                            qi,
                            uv,
                            r2,
                            &kps,
                            &p.descriptor,
                            &descriptors.descriptors[qi].data,
                            &mut best,
                        );
                    }
                }
            }
        } else {
            for qi in 0..kps.len() {
                consider_projection_candidate(
                    qi,
                    uv,
                    r2,
                    &kps,
                    &p.descriptor,
                    &descriptors.descriptors[qi].data,
                    &mut best,
                );
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

fn projection_bin(x: f64, y: f64, cell: f64) -> (i32, i32) {
    ((x / cell).floor() as i32, (y / cell).floor() as i32)
}

/// Radius gate, then Hamming. Equal distances keep the lower keypoint index.
fn consider_projection_candidate(
    qi: usize,
    uv: Point2<f64>,
    r2: f64,
    kps: &[Point2<f64>],
    map_desc: &[u8],
    query_desc: &[u8],
    best: &mut (u32, usize),
) {
    if (kps[qi] - uv).norm_squared() > r2 {
        return;
    }
    let d = hamming(map_desc, query_desc);
    if d < best.0 || (d == best.0 && qi < best.1) {
        *best = (d, qi);
    }
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
/// The matcher uses [`sampson_precomputed`]; this form is the reference the
/// gate is checked against.
#[cfg(test)]
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

/// `|E a|²` on the first two coordinates. One per old feature, reused for
/// every query against that keyframe.
fn essential_xy2(e: &Matrix3<f64>, a: &Vector3<f64>) -> f64 {
    let ea = e * a;
    ea.x * ea.x + ea.y * ea.y
}

/// Same distance as [`sampson`], with `l = Eᵀ x_query` and `|E a|_xy²` already known.
fn sampson_precomputed(a: &Vector3<f64>, l: &Vector3<f64>, ea_xy2: f64) -> f64 {
    let num = a.dot(l).powi(2);
    let den = ea_xy2 + l.x * l.x + l.y * l.y;
    if den > 0.0 {
        num / den
    } else {
        f64::INFINITY
    }
}

/// The Sampson distance is `num / (|E a|_xy² + |l|_xy²)`. `|E a|_xy²` is at
/// most `ea_xy2_max` for the features in this keyframe, so
/// `num / (ea_xy2_max + |l|_xy²)` is a lower bound. When that bound is above
/// the threshold, every feature in the set is above it too, and the exact
/// test would reject them. The slack is so a value sitting on the threshold
/// is left for the exact test.
fn sampson_rejected_for_every_candidate(num: f64, ea_xy2_max: f64, lxy2: f64, thr: f64) -> bool {
    let den_max = ea_xy2_max + lxy2;
    if !(den_max.is_finite() && den_max > 0.0) {
        return num > 0.0;
    }
    num > thr * den_max * (1.0 + 1e-8)
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
            let old_free: Vec<(usize, Vector3<f64>, f64)> = old
                .descriptors
                .descriptors
                .iter()
                .enumerate()
                .filter(|(i, _)| free(&old.map_points, *i))
                .map(|(i, d)| {
                    let xt = norm(d.keypoint.x, d.keypoint.y);
                    let ea_xy2 = essential_xy2(&e, &xt);
                    (i, xt, ea_xy2)
                })
                .collect();
            let ea_xy2_max = old_free.iter().map(|(_, _, ea)| *ea).fold(0.0, f64::max);
            // Best old feature for each new one; then each old feature is
            // kept by its closest new feature.
            let mut best_for_old: Vec<Option<(u32, usize)>> =
                vec![None; old.descriptors.descriptors.len()];
            for (qi, dq) in new.descriptors.descriptors.iter().enumerate() {
                if !free(&new.map_points, qi) {
                    continue;
                }
                let xq = norm(dq.keypoint.x, dq.keypoint.y);
                let l = e.transpose() * xq;
                let lxy2 = l.x * l.x + l.y * l.y;
                let mut best = (u32::MAX, usize::MAX);
                let mut second = u32::MAX;
                for (ti, xt, ea_xy2) in &old_free {
                    let num = xt.dot(&l).powi(2);
                    if sampson_rejected_for_every_candidate(num, ea_xy2_max, lxy2, epi_thr) {
                        continue;
                    }
                    if sampson_precomputed(xt, &l, *ea_xy2) > epi_thr {
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

fn read_point(map: &WorldMap, i: usize) -> std::sync::RwLockReadGuard<'_, MapPoint> {
    map.points[i].read().unwrap_or_else(|e| e.into_inner())
}

fn write_point(map: &WorldMap, i: usize) -> std::sync::RwLockWriteGuard<'_, MapPoint> {
    map.points[i].write().unwrap_or_else(|e| e.into_inner())
}

/// Move every observation of `drop` onto `keep`, and clear `drop`.
///
/// A keyframe that already observes `keep` cannot observe it twice, so
/// `drop`'s feature there is unlinked instead of transferred. Returns nothing;
/// the caller culls `drop` once it has no observations.
fn merge_into(map: &mut WorldMap, keep: usize, drop_at: usize) {
    if keep == drop_at || keep >= map.points.len() || drop_at >= map.points.len() {
        return;
    }
    let obs = {
        let mut dropped = write_point(map, drop_at);
        std::mem::take(&mut dropped.observations)
    };
    for (kf_id, kp_idx) in obs {
        let Some(kf) = map.keyframes.get(kf_id as usize) else {
            continue;
        };
        let mut kf = kf.write().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = kf.map_points.get_mut(kp_idx) else {
            continue;
        };
        if *slot != Some(drop_at as u64) {
            continue;
        }
        let keep_observes = read_point(map, keep)
            .observations
            .iter()
            .any(|(k, _)| *k == kf_id);
        if keep_observes {
            *slot = None;
        } else {
            *slot = Some(keep as u64);
            drop(kf);
            write_point(map, keep).observations.push((kf_id, kp_idx));
        }
    }
}

/// Fuse map points into keyframe `kf_index`.
///
/// Every map point is projected into the keyframe. When the projection lands
/// within [`FUSE_PX`] of a feature with a similar descriptor:
///
/// * the feature has no point yet, and the projected point is not already in
///   this keyframe: the feature becomes another observation of that point;
/// * the feature belongs to a different point: the two are the same corner
///   (a second pyramid level, or a point triangulated again), and the one
///   with fewer observations is merged into the other.
///
/// One point is fused to one feature, the closest acceptable one. Returns how
/// many features were attached or merged.
pub(crate) fn fuse_keyframe(k: &CameraIntrinsics, map: &mut WorldMap, kf_index: usize) -> usize {
    if kf_index >= map.keyframes.len() || map.points.is_empty() {
        return 0;
    }
    let (pose, kf_id, features, linked): (
        Pose,
        u64,
        Vec<(Point2<f64>, Vec<u8>)>,
        Vec<Option<u64>>,
    ) = {
        let kf = map.keyframes[kf_index]
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let features = kf
            .descriptors
            .descriptors
            .iter()
            .map(|d| (Point2::new(d.keypoint.x, d.keypoint.y), d.data.clone()))
            .collect();
        (
            keyframe_world_to_camera(&kf.pose),
            kf.id,
            features,
            kf.map_points.clone(),
        )
    };
    if features.is_empty() {
        return 0;
    }

    let mut seen_here = vec![false; map.points.len()];
    for id in linked.iter().flatten() {
        if let Some(slot) = seen_here.get_mut(*id as usize) {
            *slot = true;
        }
    }

    struct Cand {
        point: usize,
        feat: usize,
        ham: u32,
        dist: f64,
    }
    let mut cands = Vec::new();
    for pi in 0..map.points.len() {
        let point = read_point(map, pi);
        if point.observations.is_empty() {
            continue;
        }
        let Some(uv) = project(k, &pose, &point.world_pos.coords.cast::<f64>()) else {
            continue;
        };
        let descriptor = point.descriptor.clone();
        drop(point);
        let mut best: Option<Cand> = None;
        for (fi, (kp, desc)) in features.iter().enumerate() {
            if linked.get(fi).copied().flatten() == Some(pi as u64) {
                continue;
            }
            let dist = (kp - uv).norm();
            if dist > FUSE_PX {
                continue;
            }
            let ham = hamming(&descriptor, desc);
            if ham > FUSE_HAMMING {
                continue;
            }
            let better = match &best {
                None => true,
                Some(b) => ham < b.ham || (ham == b.ham && dist < b.dist),
            };
            if better {
                best = Some(Cand {
                    point: pi,
                    feat: fi,
                    ham,
                    dist,
                });
            }
        }
        if let Some(c) = best {
            cands.push(c);
        }
    }
    cands.sort_by(|a, b| a.ham.cmp(&b.ham).then(a.dist.total_cmp(&b.dist)));

    let mut used_point = vec![false; map.points.len()];
    let mut used_feat = vec![false; features.len()];
    let mut fused = 0usize;
    for c in cands {
        if used_feat[c.feat] || used_point[c.point] {
            continue;
        }
        let live = map.keyframes[kf_index]
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .map_points
            .get(c.feat)
            .copied()
            .flatten();
        match live {
            Some(other) if other as usize != c.point && (other as usize) < map.points.len() => {
                if used_point[other as usize] {
                    continue;
                }
                let (keep, drop_at) = {
                    let na = read_point(map, c.point).observations.len();
                    let nb = read_point(map, other as usize).observations.len();
                    if nb > na || (nb == na && (other as usize) < c.point) {
                        (other as usize, c.point)
                    } else {
                        (c.point, other as usize)
                    }
                };
                merge_into(map, keep, drop_at);
                used_point[drop_at] = true;
                used_feat[c.feat] = true;
                fused += 1;
            }
            None if !seen_here[c.point] => {
                write_point(map, c.point).observations.push((kf_id, c.feat));
                map.keyframes[kf_index]
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .map_points[c.feat] = Some(c.point as u64);
                seen_here[c.point] = true;
                used_feat[c.feat] = true;
                fused += 1;
            }
            _ => {}
        }
    }
    fused
}

/// Remove map points that were never seen again, and compact the map.
///
/// A point with no observations (merged, or every observation rejected) is
/// removed immediately. A point that still has fewer than `min_obs`
/// observations [`CULL_MIN_AGE`] keyframes after the one it was born in is
/// removed too. Monocular tracking passes 3: a point is born from a
/// triangulated pair and has to be seen from one more keyframe. RGB-D does
/// not call this. A depth point is already a metric landmark, and both
/// merging neighbours and dropping points that lacked a second view made
/// the 20-frame sweep drift (2.9 cm / 0.39° to about 5 cm / 0.7°).
///
/// Surviving points keep [`MapPoint::created_at`] and receive a new [`MapPoint::id`]
/// equal to their new index. Keyframe slots are rewritten to match, and the
/// descriptor cache is dropped.
pub(crate) fn cull_under_observed(map: &mut WorldMap, min_obs: usize) -> usize {
    let newest = match map.keyframes.len().checked_sub(1) {
        Some(n) => n as u64,
        None => return 0,
    };
    let mut drop_point = vec![false; map.points.len()];
    for i in 0..map.points.len() {
        let point = read_point(map, i);
        if point.observations.is_empty()
            || (newest.saturating_sub(point.born_kf) >= CULL_MIN_AGE
                && point.observations.len() < min_obs)
        {
            drop_point[i] = true;
        }
    }
    let n_drop = drop_point.iter().filter(|d| **d).count();
    if n_drop == 0 {
        return 0;
    }

    for (i, go) in drop_point.iter().copied().enumerate() {
        if !go {
            continue;
        }
        let obs = std::mem::take(&mut write_point(map, i).observations);
        for (kf_id, kp_idx) in obs {
            let Some(kf) = map.keyframes.get(kf_id as usize) else {
                continue;
            };
            let mut kf = kf.write().unwrap_or_else(|e| e.into_inner());
            if kf.map_points.get(kp_idx).copied().flatten() == Some(i as u64) {
                kf.map_points[kp_idx] = None;
            }
        }
    }

    let mut new_index = vec![None; map.points.len()];
    let mut kept = Vec::with_capacity(map.points.len() - n_drop);
    for (i, point) in map.points.drain(..).enumerate() {
        if drop_point[i] {
            continue;
        }
        {
            let mut p = point.write().unwrap_or_else(|e| e.into_inner());
            p.id = kept.len() as u64;
        }
        new_index[i] = Some(kept.len());
        kept.push(point);
    }
    map.points = kept;
    map.descriptor_cache = None;
    map.descriptor_cache_len = 0;

    for kf in &map.keyframes {
        let mut kf = kf.write().unwrap_or_else(|e| e.into_inner());
        for slot in &mut kf.map_points {
            if let Some(id) = *slot {
                *slot = new_index
                    .get(id as usize)
                    .copied()
                    .flatten()
                    .map(|n| n as u64);
            }
        }
    }
    n_drop
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::MapExt;
    use crate::types::KeyFrame;
    use cv_core::{Descriptor, Descriptors, FrameConvention, KeyPoint, KeyPoints};
    use std::sync::{Arc, RwLock};

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100)
    }

    fn identity_pose() -> TypedPose {
        TypedPose::new(
            Point3::origin(),
            nalgebra::Matrix3::identity(),
            FrameConvention::opencv(),
        )
    }

    fn add_born(map: &mut WorldMap, pos: Point3<f32>, byte: u8, obs: &[(u64, usize)]) -> usize {
        let mut mp = MapPoint::new(0, pos, vec![byte; 32]);
        mp.observations = obs.to_vec();
        map.add_point(mp);
        map.points.len() - 1
    }

    fn push_kf(map: &mut WorldMap, id: u64, feats: &[(f64, f64, u8)], links: &[Option<u64>]) {
        let mut descs = Descriptors::new();
        for &(x, y, byte) in feats {
            descs.push(Descriptor::new(vec![byte; 32], KeyPoint::new(x, y)));
        }
        let kps = KeyPoints {
            keypoints: descs.descriptors.iter().map(|d| d.keypoint).collect(),
        };
        let mut kf = KeyFrame::new(id, identity_pose(), kps, descs);
        assert_eq!(
            kf.map_points.len(),
            links.len(),
            "test keyframe built with one slot per feature"
        );
        kf.map_points = links.to_vec();
        map.keyframes.push(Arc::new(RwLock::new(kf)));
    }

    /// Every keyframe slot points at the point that lists that observation,
    /// and every observation has a slot. This is the invariant compaction
    /// has to keep: `id` is the index, and the two sides name each other.
    fn assert_links(map: &WorldMap) {
        for (i, point) in map.points.iter().enumerate() {
            let point = point.read().unwrap();
            assert_eq!(point.id, i as u64);
            for &(kf_id, kp_idx) in &point.observations {
                let kf = map.keyframes[kf_id as usize].read().unwrap();
                assert_eq!(kf.map_points[kp_idx], Some(i as u64));
            }
        }
        for kf in &map.keyframes {
            let kf = kf.read().unwrap();
            for (kp_idx, slot) in kf.map_points.iter().enumerate() {
                if let Some(id) = *slot {
                    let point = map.points[id as usize].read().unwrap();
                    assert!(
                        point
                            .observations
                            .iter()
                            .any(|&(k, i)| k == kf.id && i == kp_idx),
                        "keyframe {} feature {kp_idx} points at {id}, which does not list it",
                        kf.id
                    );
                }
            }
        }
    }

    #[test]
    fn unequal_lengths_are_not_a_zero_distance_match() {
        let short = [0u8; 8];
        let mut long = [0u8; 32];
        long[8..].fill(0xff);
        let d = hamming(&short, &long);
        assert!(
            d > FUSE_HAMMING && d > MAX_TRIANGULATION_HAMMING && d > MAX_PROJECTION_HAMMING,
            "mixed-length Hamming was {d}, under the fusion and triangulation caps"
        );
    }

    /// Two in-window keypoints with the same descriptor sit in different
    /// cells, and the higher index is in the cell visited first. The match
    /// is still the lower index, and a keypoint outside the radius is not.
    #[test]
    fn projection_grid_keeps_the_lower_index_on_a_tie() {
        let mut map = WorldMap::new();
        // (0, 0, 5) projects to the principal point (50, 50).
        add_born(&mut map, Point3::new(0.0, 0.0, 5.0), 7, &[]);
        let radius = 4.0;
        let mut descs = Descriptors::new();
        // x = 53.5 is the next cell; x = 46.5 is the previous one. Both are
        // 3.5 px from the projection. Index 1 is visited first.
        descs.push(Descriptor::new(vec![7; 32], KeyPoint::new(53.5, 50.0)));
        descs.push(Descriptor::new(vec![7; 32], KeyPoint::new(46.5, 50.0)));
        descs.push(Descriptor::new(vec![7; 32], KeyPoint::new(90.0, 50.0)));
        let pose = keyframe_world_to_camera(&identity_pose());
        let pairs = match_by_projection(&intrinsics(), &descs, &map, &pose, radius);
        assert_eq!(
            pairs,
            vec![(0, 0)],
            "lower index must win an equal Hamming: {pairs:?}"
        );
    }

    #[test]
    fn a_point_never_reobserved_is_removed_and_survivors_are_renumbered() {
        let mut map = WorldMap::new();
        // Born at keyframe 0, then seen twice more.
        let a = add_born(&mut map, Point3::new(0.0, 0.0, 5.0), 1, &[(0, 0)]);
        write_point(&map, a).observations.extend([(1, 0), (2, 0)]);
        // Born at keyframe 0 and never seen again.
        let _b = add_born(&mut map, Point3::new(1.0, 0.0, 5.0), 2, &[(0, 1)]);
        // Born at the newest keyframe: too young to remove, one observation.
        let c = add_born(&mut map, Point3::new(0.0, 1.0, 5.0), 3, &[(2, 1)]);
        push_kf(
            &mut map,
            0,
            &[(50.0, 50.0, 1), (60.0, 50.0, 2)],
            &[Some(0), Some(1)],
        );
        push_kf(&mut map, 1, &[(50.0, 50.0, 1)], &[Some(0)]);
        push_kf(
            &mut map,
            2,
            &[(50.0, 50.0, 1), (40.0, 50.0, 3)],
            &[Some(0), Some(2)],
        );
        let _ = map.descriptor_bytes();

        let removed = cull_under_observed(&mut map, 2);
        assert_eq!(removed, 1);
        assert_eq!(map.points.len(), 2);
        assert!(
            map.descriptor_cache.is_none(),
            "compaction must drop the cache"
        );
        let bytes = map.descriptor_bytes();
        assert_eq!(bytes[0], 1, "the reobserved point stays first");
        assert_eq!(bytes[32], 3, "the young point slides into the hole");
        assert_eq!(read_point(&map, 0).created_at, 0);
        assert_eq!(read_point(&map, 1).created_at, c as u64);
        assert_eq!(read_point(&map, 1).id, 1);
        assert_links(&map);
    }

    #[test]
    fn a_reobserved_point_stays_and_a_young_one_is_not_judged_yet() {
        let mut map = WorldMap::new();
        // Two observations, born at 0. Monocular rule (min 3) removes it once
        // two further keyframes have passed: it never earned a third view.
        add_born(&mut map, Point3::new(0.0, 0.0, 5.0), 1, &[(0, 0)]);
        write_point(&map, 0).observations.push((1, 0));
        // Born at keyframe 1, so with the newest keyframe at 2 it is one
        // keyframe old and must be kept even with two observations. The later
        // observation is recorded after birth, or born_kf would be the newer
        // keyframe and the age check would not be what this test claims.
        let young = add_born(&mut map, Point3::new(0.2, 0.0, 5.0), 2, &[(1, 1)]);
        write_point(&map, young).observations.push((2, 0));
        push_kf(&mut map, 0, &[(50.0, 50.0, 1)], &[Some(0)]);
        push_kf(
            &mut map,
            1,
            &[(50.0, 50.0, 1), (55.0, 50.0, 2)],
            &[Some(0), Some(1)],
        );
        push_kf(&mut map, 2, &[(52.0, 50.0, 2)], &[Some(1)]);

        assert_eq!(cull_under_observed(&mut map, 3), 1);
        assert_eq!(map.points.len(), 1);
        assert_eq!(read_point(&map, 0).created_at, 1);
        assert_links(&map);
    }

    #[test]
    fn an_unmatched_feature_adopts_the_point_that_projects_onto_it() {
        let mut map = WorldMap::new();
        // (0, 0, 5) projects to the principal point (50, 50).
        add_born(&mut map, Point3::new(0.0, 0.0, 5.0), 7, &[(0, 0)]);
        // Twenty pixels away: 100 * 1/5 = 20, so u = 70. Must not be fused.
        add_born(&mut map, Point3::new(1.0, 0.0, 5.0), 7, &[(0, 1)]);
        push_kf(
            &mut map,
            0,
            &[(50.0, 50.0, 7), (70.0, 50.0, 7)],
            &[Some(0), Some(1)],
        );
        push_kf(&mut map, 1, &[(50.0, 50.0, 7)], &[None]);

        assert_eq!(fuse_keyframe(&intrinsics(), &mut map, 1), 1);
        let obs = read_point(&map, 0).observations.clone();
        assert!(
            obs.contains(&(1, 0)),
            "the feature became an observation: {obs:?}"
        );
        assert_eq!(read_point(&map, 1).observations.len(), 1);
        assert_eq!(map.keyframes[1].read().unwrap().map_points[0], Some(0),);
        assert_links(&map);
    }

    #[test]
    fn two_points_on_the_same_corner_become_one() {
        let mut map = WorldMap::new();
        // A at the principal point. B is 0.05 m to the side at depth 5:
        // 100 * 0.05/5 = 1 px, so it lands on A's feature and A lands on B's
        // (2 px away). Same descriptor, so they are one corner seen twice.
        add_born(&mut map, Point3::new(0.0, 0.0, 5.0), 4, &[(0, 0)]);
        add_born(&mut map, Point3::new(0.05, 0.0, 5.0), 4, &[(0, 1)]);
        push_kf(
            &mut map,
            0,
            &[(50.0, 50.0, 4), (52.0, 50.0, 4)],
            &[Some(0), Some(1)],
        );

        assert_eq!(fuse_keyframe(&intrinsics(), &mut map, 0), 1);
        assert_eq!(cull_under_observed(&mut map, 2), 1);
        assert_eq!(map.points.len(), 1);
        assert_eq!(read_point(&map, 0).created_at, 0, "the older point is kept");
        let linked = map.keyframes[0].read().unwrap().map_points.clone();
        assert_eq!(linked.iter().filter(|s| s.is_some()).count(), 1);
        assert_links(&map);
    }

    #[test]
    fn epipolar_gate_keeps_every_sampson_inlier() {
        let e = Matrix3::new(0.0, -0.2, 0.05, 0.2, 0.0, -0.4, -0.05, 0.4, 0.0);
        let thr = (EPIPOLAR_PX / 100.0).powi(2);
        let mut pts = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                pts.push(Vector3::new(
                    (i as f64 - 15.0) / 40.0,
                    (j as f64 - 15.0) / 40.0,
                    1.0,
                ));
            }
        }
        let ea: Vec<f64> = pts.iter().map(|a| essential_xy2(&e, a)).collect();
        let ea_max = ea.iter().copied().fold(0.0, f64::max);
        let queries = [
            Vector3::new(0.1, -0.2, 1.0),
            Vector3::new(-0.4, 0.3, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        for b in queries {
            let l = e.transpose() * b;
            let lxy2 = l.x * l.x + l.y * l.y;
            for (a, ea_xy2) in pts.iter().zip(ea.iter()) {
                let exact = sampson(&e, a, &b);
                let pre = sampson_precomputed(a, &l, *ea_xy2);
                assert!(
                    (exact - pre).abs() < 1e-9 || (exact.is_infinite() && pre.is_infinite()),
                    "precomputed Sampson {pre} != {exact}"
                );
                let num = a.dot(&l).powi(2);
                if sampson_rejected_for_every_candidate(num, ea_max, lxy2, thr) {
                    assert!(
                        exact > thr,
                        "the gate dropped a point whose Sampson distance is {exact} (threshold {thr})"
                    );
                }
            }
        }
    }
}
