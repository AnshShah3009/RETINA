//! Synthetic recovery: each `OdometryMethod` must recover a known relative
//! pose from a rendered pair. `Intensity` and `Hybrid` used to return `None`
//! unconditionally - public enum variants that could not succeed - so there
//! was nothing to test. The scenes are rendered analytically (surface points +
//! texture as closed forms), so the ground truth is exact.
//!
//! The hybrid scene is deliberately **bumpy**: point-to-plane on a flat plane
//! is rank-1 (it sees only the normal-direction translation), so a plane alone
//! would prove nothing about the geometric half of the system.

#![forbid(unsafe_code)]

use cv_3d::odometry::{compute_rgbd_odometry, OdometryMethod};
use cv_3d::tsdf::CameraIntrinsics;
use nalgebra::{Matrix4, Point3, Vector3};

const W: usize = 64;
const H: usize = 48;

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(
        100.0,
        100.0,
        (W as f32 - 1.0) / 2.0,
        (H as f32 - 1.0) / 2.0,
        W as u32,
        H as u32,
    )
}

/// Texture on the scene surface, as a function of the surface's X/Y.
fn texture(x: f32, y: f32) -> f32 {
    (127.0 + 90.0 * (20.0 * x).sin() * (15.0 * y).cos()).clamp(0.0, 255.0)
}

fn color(v: f32) -> Vector3<u8> {
    let b = v.round().clamp(0.0, 255.0) as u8;
    Vector3::new(b, b, b)
}

/// A flat textured plane at z = 1: source frames as exact closed forms.
fn flat_pair(dx: f32, dy: f32) -> (Vec<f32>, Vec<f32>, Vec<Vector3<u8>>, Vec<Vector3<u8>>) {
    let k = intrinsics();
    let (mut sd, mut td) = (vec![0.0f32; W * H], vec![0.0f32; W * H]);
    let (mut sc, mut tc) = (
        vec![Vector3::new(0, 0, 0); W * H],
        vec![Vector3::new(0, 0, 0); W * H],
    );
    for v in 0..H {
        for u in 0..W {
            let idx = v * W + u;
            let (x, y) = ((u as f32 - k.cx) / k.fx, (v as f32 - k.cy) / k.fy);
            sd[idx] = 1.0;
            td[idx] = 1.0;
            sc[idx] = color(texture(x, y));
            // Camera translated by (dx, dy, 0): the surface point under the
            // target pixel (u, v) is the plane point (x - dx, y - dy).
            tc[idx] = color(texture(x - dx, y - dy));
        }
    }
    (sd, td, sc, tc)
}

/// Bumpy textured surface, rendered into the target by splatting the same 3D
/// points through the known transform - consistent depth and colour by
/// construction.
fn bumpy_pair(t_true: &Matrix4<f32>) -> (Vec<f32>, Vec<f32>, Vec<Vector3<u8>>, Vec<Vector3<u8>>) {
    let k = intrinsics();
    let (mut sd, mut td) = (vec![0.0f32; W * H], vec![0.0f32; W * H]);
    let (mut sc, mut tc) = (
        vec![Vector3::new(0, 0, 0); W * H],
        vec![Vector3::new(0, 0, 0); W * H],
    );
    let mut world = Vec::with_capacity(W * H);
    for v in 0..H {
        for u in 0..W {
            let (x, y) = ((u as f32 - k.cx) / k.fx, (v as f32 - k.cy) / k.fy);
            let z = 1.0 + 0.05 * (12.0 * x).sin() * (9.0 * y).cos();
            let idx = v * W + u;
            sd[idx] = z;
            sc[idx] = color(texture(5.0 * x, 4.0 * y));
            world.push(Point3::new(x * z, y * z, z));
        }
    }
    for (i, w) in world.iter().enumerate() {
        let p = t_true.transform_point(w);
        if p.z <= 1e-6 {
            continue;
        }
        let u = (p.x * k.fx / p.z + k.cx).round() as isize;
        let v = (p.y * k.fy / p.z + k.cy).round() as isize;
        if u < 0 || u >= W as isize || v < 0 || v >= H as isize {
            continue;
        }
        let idx = v as usize * W + u as usize;
        if td[idx] == 0.0 || p.z < td[idx] {
            td[idx] = p.z;
            tc[idx] = sc[i];
        }
    }
    // Forward splatting leaves holes (~8% of pixels at a 2 px shift), and a
    // hole reads as black - a photometric residual of ~127 grey against a
    // scene that never contained it, which drags the solver into a bogus
    // minimum. Real renders have no holes; fill them by neighbour dilation so
    // the test scene has none either.
    for _ in 0..3 {
        let (prev_d, prev_c) = (td.clone(), tc.clone());
        for y in 0..H {
            for x in 0..W {
                let idx = y * W + x;
                if td[idx] != 0.0 {
                    continue;
                }
                for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (ny, nx) = (y as isize + dy, x as isize + dx);
                    if ny < 0 || ny >= H as isize || nx < 0 || nx >= W as isize {
                        continue;
                    }
                    let nidx = ny as usize * W + nx as usize;
                    if prev_d[nidx] != 0.0 {
                        td[idx] = prev_d[nidx];
                        tc[idx] = prev_c[nidx];
                        break;
                    }
                }
            }
        }
    }
    (sd, td, sc, tc)
}

fn assert_recovers(name: &str, got: &Matrix4<f32>, t_true: &Matrix4<f32>) {
    let dt = (got[(0, 3)] - t_true[(0, 3)])
        .abs()
        .max((got[(1, 3)] - t_true[(1, 3)]).abs())
        .max((got[(2, 3)] - t_true[(2, 3)]).abs());
    assert!(
        dt < 1e-2,
        "{name}: translation off by {dt:.4} (got [{}, {}, {}])",
        got[(0, 3)],
        got[(1, 3)],
        got[(2, 3)]
    );
    let mut rot_err = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            rot_err = rot_err.max((got[(i, j)] - t_true[(i, j)]).abs());
        }
    }
    assert!(rot_err < 1e-2, "{name}: rotation off by {rot_err:.4}");
}

#[test]
fn intensity_odometry_recovers_an_in_plane_translation() {
    let (dx, dy) = (0.03, 0.015);
    let (sd, td, sc, tc) = flat_pair(dx, dy);
    let k = intrinsics();
    let result = compute_rgbd_odometry(
        &sd,
        &td,
        Some(&sc),
        Some(&tc),
        &k,
        W,
        H,
        OdometryMethod::Intensity,
    )
    .expect("Intensity odometry must converge on a textured plane");

    let mut t_true = Matrix4::identity();
    t_true[(0, 3)] = dx;
    t_true[(1, 3)] = dy;
    assert_recovers("intensity", &result.transformation, &t_true);
}

#[test]
fn hybrid_odometry_recovers_a_pose_on_a_bumpy_scene() {
    let mut t_true = Matrix4::identity();
    t_true[(0, 3)] = 0.02;
    t_true[(1, 3)] = -0.01;
    t_true[(2, 3)] = 0.01;
    let (sd, td, sc, tc) = bumpy_pair(&t_true);
    let k = intrinsics();
    let result = compute_rgbd_odometry(
        &sd,
        &td,
        Some(&sc),
        Some(&tc),
        &k,
        W,
        H,
        OdometryMethod::Hybrid,
    )
    .expect("Hybrid odometry must converge when both residual kinds are present");

    assert_recovers("hybrid", &result.transformation, &t_true);
}

#[test]
fn point_to_plane_still_recovers_on_the_same_bumpy_scene() {
    // Control: the depth-only method works the same scene, so the hybrid
    // result is not the geometric rows carrying an absent photometric half.
    let mut t_true = Matrix4::identity();
    t_true[(0, 3)] = 0.02;
    t_true[(1, 3)] = -0.01;
    t_true[(2, 3)] = 0.01;
    let (sd, td, _sc, _tc) = bumpy_pair(&t_true);
    let k = intrinsics();
    let result =
        compute_rgbd_odometry(&sd, &td, None, None, &k, W, H, OdometryMethod::PointToPlane)
            .expect("PointToPlane must converge on the bumpy scene");
    assert_recovers("point_to_plane", &result.transformation, &t_true);
}

#[test]
fn missing_inputs_fail_rather_than_guess() {
    let (sd, td, sc, tc) = flat_pair(0.02, 0.0);
    let k = intrinsics();
    // Intensity without colours and Hybrid without colours are not those
    // problems: `None`, not a depth-only answer wearing their name.
    assert!(
        compute_rgbd_odometry(&sd, &td, None, None, &k, W, H, OdometryMethod::Intensity).is_none(),
        "Intensity without colour must fail"
    );
    assert!(
        compute_rgbd_odometry(&sd, &td, Some(&sc), None, &k, W, H, OdometryMethod::Hybrid)
            .is_none(),
        "Hybrid without both colours must fail"
    );
    let _ = tc;
}
