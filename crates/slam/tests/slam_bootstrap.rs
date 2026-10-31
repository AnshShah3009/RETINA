//! SLAM has to be able to *start*.
//!
//! `Tracker::process_frame` used to return `Err` whenever the map was empty, and
//! nothing in cv-slam ever added a map point - so every frame of every sequence
//! failed, through every entry point (`process_frame`,
//! `process_frame_with_depth`, `Slam::process_image`, Python `PySlam`).
//!
//! These tests render a textured synthetic scene (a back wall, a floor and two
//! boxes at different depths) by ray casting, from a known camera trajectory,
//! and check that the tracker recovers that trajectory:
//!
//! * RGB-D: the first frame bootstraps the map from depth, the rest track by
//!   PnP. Metric scale, so poses are compared directly.
//! * Monocular: the first frame is held as an initialisation reference, the
//!   second initialises from the essential matrix, the rest track by PnP.
//!   Scale is arbitrary, so translations are compared after fixing the scale
//!   on the initialisation baseline.
//!
//! The 20-frame sweeps check that the map *grows*: the camera travels ~3 m,
//! so the view that built the initial map leaves the image, and tracking has
//! to continue on points added at later keyframes (back-projected from depth,
//! or triangulated between monocular keyframes) without making every frame a
//! keyframe.

use cv_core::storage::CpuStorage;
use cv_core::{CameraIntrinsics, CpuTensor, Pose, Tensor, TensorShape};
use cv_slam::tracking::{Tracker, TrackingState};
use cv_slam::WorldMap;
use nalgebra::{Rotation3, UnitQuaternion, Vector3};

const W: usize = 320;
const H: usize = 240;
const F: f64 = 300.0;

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(F, F, W as f64 / 2.0, H as f64 / 2.0, W as u32, H as u32)
}

struct Aabb {
    min: Vector3<f64>,
    max: Vector3<f64>,
    seed: u32,
}

fn scene() -> Vec<Aabb> {
    let b = |min: [f64; 3], max: [f64; 3], seed| Aabb {
        min: Vector3::from(min),
        max: Vector3::from(max),
        seed,
    };
    vec![
        // Back wall.
        b([-8.0, -6.0, 5.0], [8.0, 6.0, 5.3], 1),
        // Floor (camera y points down, so the floor is at +y).
        b([-8.0, 1.0, -1.0], [8.0, 1.3, 5.0], 2),
        // Two boxes at different depths, for genuine 3D structure.
        b([-1.0, -0.7, 2.6], [0.1, 0.6, 3.4], 3),
        b([0.6, -1.3, 3.4], [1.7, 0.3, 4.2], 4),
    ]
}

fn hash(a: i64, b: i64, seed: u32) -> f64 {
    let mut h = (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (seed as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h & 0xFF) as f64
}

/// Two octaves of random-intensity cells: plenty of corners, and the
/// neighbourhood of each one is distinctive enough for ORB to match.
fn texture(a: f64, b: f64, seed: u32) -> f64 {
    let coarse = hash((a / 0.13).floor() as i64, (b / 0.13).floor() as i64, seed);
    let fine = hash(
        (a / 0.047).floor() as i64,
        (b / 0.047).floor() as i64,
        seed + 100,
    );
    0.65 * coarse + 0.35 * fine
}

/// Nearest hit of a world ray: (distance along `dir`, intensity).
fn cast(origin: &Vector3<f64>, dir: &Vector3<f64>, scene: &[Aabb]) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for b in scene {
        let mut t_near = f64::NEG_INFINITY;
        let mut t_far = f64::INFINITY;
        let mut axis = 0;
        for k in 0..3 {
            if dir[k].abs() < 1e-12 {
                if origin[k] < b.min[k] || origin[k] > b.max[k] {
                    t_near = f64::INFINITY;
                }
                continue;
            }
            let t1 = (b.min[k] - origin[k]) / dir[k];
            let t2 = (b.max[k] - origin[k]) / dir[k];
            let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            if lo > t_near {
                t_near = lo;
                axis = k;
            }
            t_far = t_far.min(hi);
        }
        if t_near > t_far || t_near <= 1e-6 {
            continue;
        }
        if best.map_or(true, |(d, _)| t_near < d) {
            let p = origin + dir * t_near;
            let (u, v) = match axis {
                0 => (p.y, p.z),
                1 => (p.x, p.z),
                _ => (p.x, p.y),
            };
            best = Some((t_near, texture(u, v, b.seed)));
        }
    }
    best
}

/// A world -> camera pose for a camera centred at `c` with yaw/pitch `ypr`.
fn camera_pose(c: Vector3<f64>, yaw: f64, pitch: f64) -> Pose {
    // Camera-to-world rotation.
    let r_wc = Rotation3::from_euler_angles(pitch, yaw, 0.0);
    let r_cw = r_wc.inverse();
    Pose::from_quat_translation(UnitQuaternion::from_rotation_matrix(&r_cw), -(r_cw * c))
}

/// Render the intensity image and the metric depth (camera z) for `pose`.
fn render(pose: &Pose) -> (CpuTensor<u8>, CpuTensor<f32>) {
    let scene = scene();
    let inv = pose.inverse(); // camera -> world
    let origin = inv.translation;
    let r_wc = inv.rotation;
    let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
    let mut img = vec![0u8; W * H];
    let mut depth = vec![0f32; W * H];
    const SS: usize = 3;
    for y in 0..H {
        for x in 0..W {
            let mut acc = 0.0;
            let mut n = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let u = x as f64 + (sx as f64 + 0.5) / SS as f64 - 0.5;
                    let v = y as f64 + (sy as f64 + 0.5) / SS as f64 - 0.5;
                    let d_cam = Vector3::new((u - cx) / F, (v - cy) / F, 1.0);
                    if let Some((_, i)) = cast(&origin, &(r_wc * d_cam), &scene) {
                        acc += i;
                        n += 1.0;
                    }
                }
            }
            img[y * W + x] = if n > 0.0 { (acc / n).round() as u8 } else { 0 };
            // Pixel-centre ray; its camera z is 1, so the ray parameter is depth.
            let d_cam = Vector3::new((x as f64 - cx) / F, (y as f64 - cy) / F, 1.0);
            depth[y * W + x] = cast(&origin, &(r_wc * d_cam), &scene)
                .map(|(t, _)| t as f32)
                .unwrap_or(0.0);
        }
    }
    (
        Tensor::from_vec(img, TensorShape::new(1, H, W)).unwrap(),
        Tensor::from_vec(depth, TensorShape::new(1, H, W)).unwrap(),
    )
}

fn tracker() -> Tracker {
    let group = std::sync::Arc::new(
        cv_runtime::orchestrator::ResourceGroup::new(
            "slam-bootstrap",
            cv_hal::backend::DeviceId(0),
            1,
            None,
            cv_runtime::orchestrator::GroupPolicy::default(),
        )
        .expect("resource group"),
    );
    Tracker::new(group, intrinsics())
}

fn rotation_error_deg(a: &Pose, b: &Pose) -> f64 {
    (a.rotation.inverse() * b.rotation).angle().to_degrees()
}

/// Five RGB-D frames: the first must bootstrap the map, the rest must track,
/// and every pose must match the ground truth (world frame = first camera).
#[test]
fn rgbd_bootstrap_then_tracks_known_trajectory() {
    let truth: Vec<Pose> = (0..5)
        .map(|k| {
            let k = k as f64;
            camera_pose(
                Vector3::new(0.06 * k, -0.02 * k, 0.04 * k),
                (-1.0f64 * k).to_radians(),
                (0.5f64 * k).to_radians(),
            )
        })
        .collect();

    let mut tracker = tracker();
    let mut map = WorldMap::new();
    for (k, gt) in truth.iter().enumerate() {
        let (img, depth) = render(gt);
        let (pose, inliers) = tracker
            .process_frame_with_depth(&img, &depth, &mut map)
            .unwrap_or_else(|e| panic!("RGB-D frame {k} failed: {e}"));
        let rot = rotation_error_deg(&pose, gt);
        // Compare camera centres, which is what "trajectory" means.
        let c_est = pose.inverse().translation;
        let c_gt = gt.inverse().translation;
        let terr = (c_est - c_gt).norm();
        println!(
            "frame {k}: map points {}, tracked {}, centre err {terr:.4} m, rot err {rot:.3} deg",
            map.points.len(),
            inliers.len()
        );
        assert_eq!(tracker.state, TrackingState::Tracking, "frame {k}");
        assert!(terr < 0.02, "frame {k}: camera centre off by {terr:.4} m");
        assert!(rot < 0.5, "frame {k}: rotation off by {rot:.3} deg");
        if k == 0 {
            assert!(
                map.points.len() >= 50,
                "the first RGB-D frame must bootstrap the map, got {} points",
                map.points.len()
            );
            assert!(
                !map.keyframes.is_empty(),
                "bootstrap frame must be a keyframe"
            );
        } else {
            assert!(
                inliers.len() >= 30,
                "frame {k}: only {} inliers",
                inliers.len()
            );
        }
    }
}

/// Five monocular frames: the first is held as the initialisation reference,
/// the second initialises the map by two-view geometry, the rest track by PnP.
#[test]
fn monocular_two_view_init_then_tracks_known_trajectory() {
    let truth: Vec<Pose> = (0..5)
        .map(|k| {
            let k = k as f64;
            camera_pose(
                Vector3::new(0.15 * k, -0.03 * k, 0.05 * k),
                (-1.0f64 * k).to_radians(),
                0.0,
            )
        })
        .collect();

    let mut tracker = tracker();
    let mut map = WorldMap::new();
    let mut scale = None;
    let mut successes = 0;
    for (k, gt) in truth.iter().enumerate() {
        let (img, _) = render(gt);
        let res = tracker.process_frame(&img, &mut map);
        if k == 0 {
            let e = res.expect_err("one monocular frame cannot yield a pose");
            println!("frame 0: {e}");
            assert_eq!(tracker.state, TrackingState::Initializing);
            assert!(map.points.is_empty());
            continue;
        }
        let (pose, inliers) = res.unwrap_or_else(|e| panic!("mono frame {k} failed: {e}"));
        successes += 1;
        assert_eq!(tracker.state, TrackingState::Tracking, "frame {k}");
        let c_est = pose.inverse().translation;
        let c_gt = gt.inverse().translation;
        // Monocular scale is arbitrary: fix it on the initialisation baseline.
        let s = *scale.get_or_insert_with(|| c_gt.norm() / c_est.norm());
        let terr = (c_est * s - c_gt).norm();
        let rot = rotation_error_deg(&pose, gt);
        println!(
            "frame {k}: map points {}, tracked {}, scaled centre err {terr:.4} m, rot err {rot:.3} deg",
            map.points.len(),
            inliers.len()
        );
        // Tolerances are set by the front end, not the geometry: ORB matches
        // only ~20% of features across a 16 cm baseline on this scene, with
        // ~0.6-0.9 px localisation error against ground truth, and the init
        // has ~2-3 deg of parallax.
        //
        // The rotation bound was relaxed to 3.5 deg while monocular mode
        // never grew its map: by frame 3 only ~18 points carried the pose and
        // it measured 2.887 deg (2.968 at frame 4). With keyframes,
        // triangulation and local BA the tracked set grows instead (43-97
        // points) and it measures <= 0.684 deg, centre error <= 8.1% of the
        // distance travelled (was <= 10.4%).
        assert!(rot < 2.5, "frame {k}: rotation off by {rot:.3} deg");
        // Direction of the init baseline, and scale drift afterwards.
        assert!(
            terr < 0.15 * c_gt.norm(),
            "frame {k}: scaled camera centre off by {terr:.4} m (gt {:?}, est {:?})",
            c_gt,
            c_est * s
        );
        if k == 1 {
            assert!(
                map.points.len() >= 50,
                "init made {} points",
                map.points.len()
            );
        }
    }
    assert_eq!(successes, 4);
}

/// Tracking loss must be visible: a frame that cannot be matched to the map
/// returns `Err` and sets the state to `Lost`, rather than silently handing back
/// the previous pose as if it had been measured.
#[test]
fn tracking_loss_is_reported() {
    let gt = camera_pose(Vector3::zeros(), 0.0, 0.0);
    let (img, depth) = render(&gt);
    let mut tracker = tracker();
    let mut map = WorldMap::new();
    tracker
        .process_frame_with_depth(&img, &depth, &mut map)
        .expect("bootstrap");

    // A featureless frame: nothing to match.
    let blank: Tensor<u8, CpuStorage<u8>> =
        Tensor::from_vec(vec![128u8; W * H], TensorShape::new(1, H, W)).unwrap();
    let res = tracker.process_frame(&blank, &mut map);
    assert!(res.is_err(), "lost frame returned {:?}", res.map(|r| r.0));
    assert_eq!(tracker.state, TrackingState::Lost);

    // Recovery: the original view relocalises against the existing map.
    let (pose, _) = tracker
        .process_frame(&img, &mut map)
        .expect("returning to a mapped view must relocalise");
    assert_eq!(tracker.state, TrackingState::Tracking);
    assert!(pose.translation.norm() < 0.02);
}

/// A sweep right across the scene: the camera translates ~3 m while turning
/// to the right, so by the last frame the view that initialised the map has
/// (almost) left the image.
fn sweep(n: usize) -> Vec<Pose> {
    (0..n)
        .map(|k| {
            let k = k as f64;
            camera_pose(
                Vector3::new(0.16 * k, -0.01 * k, 0.03 * k),
                (0.6f64 * k).to_radians(),
                (0.2f64 * k).to_radians(),
            )
        })
        .collect()
}

/// How many of `inliers` were created during initialisation.
///
/// `born_before` is [`WorldMap::points_created`] at the end of initialisation.
/// Culling compacts point indices, so the index itself no longer says when a
/// point was born; [`MapPoint::created_at`] does, and it is not reused.
fn from_initial_map(inliers: &[usize], map: &WorldMap, born_before: u64) -> usize {
    inliers
        .iter()
        .filter(|&&i| {
            map.points.get(i).is_some_and(|p| {
                p.read().unwrap_or_else(|e| e.into_inner()).created_at < born_before
            })
        })
        .count()
}

/// Twenty monocular frames over the sweep. Tracking has to survive on map
/// points triangulated from later keyframes, and the trajectory error has to
/// stay bounded.
#[test]
fn monocular_long_sequence_extends_the_map() {
    const N: usize = 20;
    let truth = sweep(N);

    let mut tracker = tracker();
    let mut map = WorldMap::new();
    let mut scale = None;
    let mut n_initial = 0usize;
    let mut born_before = 0u64;
    let mut max_rot = 0.0f64;
    let mut max_rel_terr = 0.0f64;
    let mut last_inliers = Vec::new();
    for (k, gt) in truth.iter().enumerate() {
        let (img, _) = render(gt);
        let res = tracker.process_frame(&img, &mut map);
        if k == 0 {
            assert!(res.is_err());
            continue;
        }
        let (pose, inliers) = res.unwrap_or_else(|e| {
            panic!(
                "mono frame {k} failed ({} map points, {} keyframes): {e}",
                map.points.len(),
                map.keyframes.len()
            )
        });
        if k == 1 {
            n_initial = map.points.len();
            born_before = map.points_created();
        }
        let c_est = pose.inverse().translation;
        let c_gt = gt.inverse().translation;
        // Monocular scale is arbitrary: fix it on the initialisation baseline.
        let s = *scale.get_or_insert_with(|| c_gt.norm() / c_est.norm());
        let terr = (c_est * s - c_gt).norm();
        let rot = rotation_error_deg(&pose, gt);
        let travelled = c_gt.norm();
        println!(
            "frame {k}: map points {}, keyframes {}, tracked {} ({} from init), scaled \
             centre err {terr:.4} m ({:.1}% of travel), rot err {rot:.3} deg",
            map.points.len(),
            map.keyframes.len(),
            inliers.len(),
            from_initial_map(&inliers, &map, born_before),
            100.0 * terr / travelled
        );
        assert_eq!(tracker.state, TrackingState::Tracking, "frame {k}");
        assert!(
            inliers.len() >= 35,
            "frame {k}: only {} inliers",
            inliers.len()
        );
        max_rot = max_rot.max(rot);
        max_rel_terr = max_rel_terr.max(terr / travelled);
        last_inliers = inliers;
    }
    println!(
        "max rot err {max_rot:.3} deg, max centre err {:.1}% of travel",
        100.0 * max_rel_terr
    );
    assert!(
        map.points.len() > 2 * n_initial,
        "the map never grew: {} points, {n_initial} at initialisation",
        map.points.len()
    );
    // By the end the pose rests on points mapped after initialisation.
    assert!(
        from_initial_map(&last_inliers, &map, born_before) * 4 < last_inliers.len(),
        "the last frame still tracks mostly initial points: {} of {}",
        from_initial_map(&last_inliers, &map, born_before),
        last_inliers.len()
    );
    // Measured with detect_ctx calling detect: rotation <= 0.318 deg, centre
    // error <= 5.3% of the distance travelled (4.2% at the last frame), 20
    // keyframes, 619 map points. Frame 2 tracks 38. Before the detectors
    // agreed: 0.555 deg, 6.6% (0.9% at the last frame), 606 points.
    // Before culling the same sweep held 1338 points (rotation <= 0.659 deg,
    // 5.3% at the last frame). Before map extension the sequence was lost
    // at frame 3 (28 inliers, 3.3 deg, 66% of travel).
    assert!(max_rot < 1.5, "rotation drifted to {max_rot:.3} deg");
    assert!(
        max_rel_terr < 0.10,
        "camera centre drifted to {:.1}% of the distance travelled",
        100.0 * max_rel_terr
    );
}

/// Twenty RGB-D frames over the same sweep. The map must keep up, but not by
/// making every frame a keyframe, and the metric trajectory must stay tight.
#[test]
fn rgbd_long_sequence_keyframes_sparingly() {
    const N: usize = 20;
    let truth = sweep(N);

    let mut tracker = tracker();
    let mut map = WorldMap::new();
    let mut max_rot = 0.0f64;
    let mut max_terr = 0.0f64;
    for (k, gt) in truth.iter().enumerate() {
        let (img, depth) = render(gt);
        let (pose, inliers) = tracker
            .process_frame_with_depth(&img, &depth, &mut map)
            .unwrap_or_else(|e| panic!("RGB-D frame {k} failed: {e}"));
        let rot = rotation_error_deg(&pose, gt);
        let terr = (pose.inverse().translation - gt.inverse().translation).norm();
        println!(
            "frame {k}: map points {}, keyframes {}, tracked {}, centre err {terr:.4} m, \
             rot err {rot:.3} deg",
            map.points.len(),
            map.keyframes.len(),
            inliers.len()
        );
        assert_eq!(tracker.state, TrackingState::Tracking, "frame {k}");
        max_rot = max_rot.max(rot);
        max_terr = max_terr.max(terr);
    }
    println!(
        "{} keyframes for {N} frames, max centre err {max_terr:.4} m, max rot err {max_rot:.3} deg",
        map.keyframes.len()
    );
    assert!(
        map.keyframes.len() <= N / 2,
        "{} keyframes for {N} frames: nearly every frame became a keyframe",
        map.keyframes.len()
    );
    assert!(map.keyframes.len() >= 2, "the map never extended");
    // Measured with the depth residual and detect_ctx calling detect:
    // 5 keyframes, centre error <= 1.89 cm, rotation <= 0.234 deg. Before the
    // detectors agreed the same sweep was 3.80 cm / 0.472 deg. Tracking
    // without bundle adjustment was 2.94 cm / 0.391 deg; reprojection alone,
    // with no depth residual, jumped 0.4–0.6 m at keyframes. Before the
    // keyframe policy change: 20 keyframes, 3.79 cm, 0.513 deg.
    assert!(max_terr < 0.04, "camera centre drifted {max_terr:.4} m");
    assert!(max_rot < 0.5, "rotation drifted {max_rot:.3} deg");
}
