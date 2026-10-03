//! Numerical-parity number generator: camera composition, projection and
//! distortion (`cv-calib3d` / `cv-core::geometry`) vs OpenCV 4.13.
//!
//! Companion reference side: `parity/parity_calib.py`.
//!
//! Same line protocol as `parity/parity_filters.rs`, plus
//!
//! ```text
//! #FR <name> <n> <v0> ...        n f64 values (a point list or a 3x3 row-major)
//! #FK <name> <v0> <v1> ...       an RGB triple
//! ```
//!
//! **Input identity is the harness's sharpest contract.** Every camera, pose,
//! point and matrix below is defined by a closed form that `parity_calib.py`
//! re-derives independently and asserts equality against. Nothing here is read
//! from disk, so a drifting generator fails loudly instead of producing a
//! meaningless comparison.

use std::io::Write;

use cv_calib3d::{
    calibrate_camera_planar_with_options, find_essential_mat, find_essential_mat_ransac,
    find_fundamental_mat, find_fundamental_mat_ransac, generate_chessboard_object_points,
    project_points, project_points_with_distortion, recover_pose_from_essential,
    CameraCalibrationOptions, HomographySolver,
};
use cv_core::geometry::{
    CameraIntrinsics, CameraIntrinsicsF32, CameraModel, Distortion, DistortionF32,
    FisheyeDistortion, FisheyeDistortionF32, PinholeModel, PinholeModelF32, Pose,
};
use nalgebra::{Matrix3, Point2, Point3, Vector3};

// ── output plumbing ─────────────────────────────────────────────────────────

fn emit_f64s(name: &str, vals: &[f64]) {
    let mut line = format!("#FR {} {}", name, vals.len());
    for v in vals {
        line.push_str(&format!(" {:.17e}", v));
    }
    println!("{line}");
}

fn emit_f32(name: &str, v: f32) {
    println!("#F {} {:.17e}", name, v);
}

fn emit_scalar(name: &str, v: f64) {
    println!("#V {} {:.17e}", name, v);
}

fn emit_image(name: &str, img: &[f32], w: usize, h: usize) {
    let mut line = format!("#IM {} {} {}", name, w, h);
    for v in img {
        line.push_str(&format!(" {:.1}", v));
    }
    println!("{line}");
}

/// A pose's translation plus its 3x3 rotation, row-major, as a single record.
fn emit_pose(name: &str, pose: &Pose) {
    let r = pose.rotation_matrix();
    let mut vals = vec![pose.translation.x, pose.translation.y, pose.translation.z];
    for i in 0..3 {
        for j in 0..3 {
            vals.push(r[(i, j)]);
        }
    }
    emit_f64s(name, &vals);
}

fn matrix_row_major(m: &Matrix3<f64>) -> Vec<f64> {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| m[(i, j)]))
        .collect()
}

fn scaled_matrix(m: &Matrix3<f64>, s: f64) -> Vec<f64> {
    matrix_row_major(m).into_iter().map(|v| v / s).collect()
}

// ── shared scenario definitions (mirrored exactly in parity_calib.py) ────────

/// Deliberately anisotropic, non-square image: a symmetric or square target
/// would let a swapped or transposed intrinsic pass unnoticed.
const IMG_W: u32 = 640;
const IMG_H: u32 = 480;

const CAM: [f64; 4] = [500.0, 510.0, 320.0, 240.0];

/// Brown-Conrady: strong barrel + real tangential terms + a k3 term, so any
/// dropped coefficient or reordering of (p1, p2) shows up.
const RADTAN: [f64; 5] = [0.0231, -0.00514, 0.00073, -0.00041, 0.000091];

/// Kannala-Brandt. OpenCV orders fisheye D as (k1, k2, k3, k4).
const KANNALA: [f64; 4] = [0.0312, -0.00214, 0.00017, -0.0000094];

fn intrinsics() -> CameraIntrinsics {
    CameraIntrinsics::new(CAM[0], CAM[1], CAM[2], CAM[3], IMG_W, IMG_H)
}

fn intrinsics_f32() -> CameraIntrinsicsF32 {
    CameraIntrinsicsF32::new(
        CAM[0] as f32,
        CAM[1] as f32,
        CAM[2] as f32,
        CAM[3] as f32,
        IMG_W,
        IMG_H,
    )
}

fn rodrigues(rvec: [f64; 3]) -> Matrix3<f64> {
    let t = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if t < 1e-12 {
        return Matrix3::identity();
    }
    let (rx, ry, rz) = (rvec[0] / t, rvec[1] / t, rvec[2] / t);
    let (c, s) = (t.cos(), t.sin());
    let q = 1.0 - c;
    Matrix3::new(
        q * rx * rx + c,
        q * rx * ry - s * rz,
        q * rx * rz + s * ry,
        q * rx * ry + s * rz,
        q * ry * ry + c,
        q * ry * rz - s * rx,
        q * rx * rz - s * ry,
        q * ry * rz + s * rx,
        q * rz * rz + c,
    )
}

fn pose(rvec: [f64; 3], tvec: [f64; 3]) -> Pose {
    Pose::new(rodrigues(rvec), Vector3::new(tvec[0], tvec[1], tvec[2]))
}

/// 9x6 pinhole target on a 0.03 m grid.
const GRID_COLS: usize = 9;
const GRID_ROWS: usize = 6;
const SQUARE: f64 = 0.03;

fn grid_object_points() -> Vec<Point3<f64>> {
    generate_chessboard_object_points((GRID_COLS, GRID_ROWS), SQUARE)
}

/// The rotation vectors behind the named poses, in the same order.
const POSE_RVECS: [(&str, [f64; 3]); 3] = [
    ("fronto", [0.0, 0.0, 0.0]),
    ("tilted", [0.26, -0.19, 0.11]),
    ("behind", [0.31, -0.24, 0.14]),
];

/// The three poses the report calls out by name.
fn poses() -> Vec<(&'static str, Pose)> {
    vec![
        // Straight-on: the target is perpendicular to the optical axis, so the
        // projection is the textbook one and any deviation is a bug.
        ("fronto", pose([0.0, 0.0, 0.0], [-0.12, -0.09, 0.62])),
        // Deliberately tilted about both horizontal axes; this is the case
        // where a tangential-term ordering error becomes visible.
        ("tilted", pose([0.26, -0.19, 0.11], [-0.18, -0.14, 0.55])),
        // Tilted *and* translated so that part of the target falls behind the
        // camera plane. `project_points` must refuse the whole batch.
        ("behind", pose([0.31, -0.24, 0.14], [0.28, 0.05, -0.026])),
    ]
}

/// One hard-edged geometric scene for the feature detectors: a regular lattice
/// of squares plus a set of filled discs at irrational-ish centres. Generated
/// from a closed form on both sides and asserted byte-identical.
const FW: usize = 320;
const FH: usize = 240;

const DISCS: [(f64, f64, f64); 6] = [
    (63.13, 58.71, 17.0),
    (188.42, 71.29, 12.0),
    (99.87, 151.6, 14.0),
    (241.5, 133.09, 20.0),
    (160.05, 205.44, 11.0),
    (283.7, 44.2, 15.0),
];

/// The rotated square's half-extent and centre, emitted so the reference side
/// does not have to re-derive the same magic numbers.
const ROT_CX: f64 = 209.3;
const ROT_CY: f64 = 63.7;
const ROT_HALF: f64 = 24.6;
const ROT_DEG: f64 = 40.0;

/// One hard-edged geometric scene for the feature detectors: a 12-on/12-off
/// diagonal lattice of squares, six filled discs at non-lattice centres, and
/// one square rotated 40 degrees. Generated from this closed form on both
/// sides and asserted byte-identical.
fn feature_scene() -> Vec<u8> {
    let mut img = vec![28u8; FW * FH];
    // 24-pixel lattice of filled squares, 12 on, 12 off, along the diagonal.
    for y in 0..FH {
        for x in 0..FW {
            if ((x + y) / 12) % 2 == 0 {
                img[y * FW + x] = 205;
            }
        }
    }
    // Discs.
    for &(cx, cy, r) in DISCS.iter() {
        for y in 0..FH {
            for x in 0..FW {
                let dx = x as f64 + 0.5 - cx;
                let dy = y as f64 + 0.5 - cy;
                if dx * dx + dy * dy <= r * r {
                    img[y * FW + x] = 245;
                }
            }
        }
    }
    // One square rotated 40 degrees: four edges at a single non-axis-aligned
    // angle, which is what a corner detector's orientation estimate has to see.
    let (cs, sn) = (ROT_DEG.to_radians().cos(), ROT_DEG.to_radians().sin());
    for y in 0..FH {
        for x in 0..FW {
            let bx = x as f64 + 0.5 - ROT_CX;
            let by = y as f64 + 0.5 - ROT_CY;
            let ux = cs * bx + sn * by;
            let uy = -sn * bx + cs * by;
            if ux.abs() <= ROT_HALF && uy.abs() <= ROT_HALF {
                img[y * FW + x] = 15;
            }
        }
    }
    img
}

fn main() {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    // Echo the inputs the reference side has to reproduce.
    emit_f64s("cam", &CAM);
    emit_f64s("radtan", &RADTAN);
    emit_f64s("kannala", &KANNALA);
    emit_scalar("img_w", IMG_W as f64);
    emit_scalar("img_h", IMG_H as f64);
    emit_scalar("square", SQUARE);
    emit_scalar("grid_cols", GRID_COLS as f64);
    emit_scalar("grid_rows", GRID_ROWS as f64);

    // ── 1. camera-matrix composition ───────────────────────────────────────
    emit_f64s("K_rowmajor", &matrix_row_major(&intrinsics().matrix()));
    emit_f64s(
        "K_ideal_rowmajor",
        &matrix_row_major(&CameraIntrinsics::new_ideal(IMG_W, IMG_H).matrix()),
    );
    match intrinsics().try_inverse_matrix() {
        Some(k_inv) => emit_f64s("Kinv_rowmajor", &matrix_row_major(&k_inv)),
        None => emit_f64s("Kinv_rowmajor", &[]),
    }
    {
        let (dx, dy) = (7.0, -3.0);
        let a = intrinsics().matrix();
        let k_inv = intrinsics().try_inverse_matrix().unwrap();
        let comp = a * k_inv;
        emit_f64s("K_Kinv", &matrix_row_major(&comp));
        // Every opencv-style composition of the same factors.
        emit_f64s("K_T_Kinv", &matrix_row_major(&(a.transpose() * k_inv)));
        let full = Matrix3::new(
            intrinsics().fx,
            0.0,
            intrinsics().cx, //
            0.0,
            intrinsics().fy,
            intrinsics().cy, //
            0.0,
            0.0,
            1.0,
        );
        emit_f64s("Kfull_Kinv", &matrix_row_major(&(full * k_inv)));
        let v = full * Vector3::new(dx, dy, 1.0);
        emit_f64s("Kfull_px", &[v[0] / v[2], v[1] / v[2]]);
        let v2 = a * Vector3::new(dx, dy, 1.0);
        emit_f64s("K_px", &[v2[0] / v2[2], v2[1] / v2[2]]);
    }
    // f32 composition: measured against an f64 reference, tolerance is a few ULP.
    {
        let kf = intrinsics_f32().matrix();
        let kinv32 = intrinsics_f32()
            .matrix()
            .try_inverse()
            .expect("f32 intrinsics are non-singular");
        let comp = kf * kinv32;
        let mut vals = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                vals.push(comp[(i, j)] as f64);
            }
        }
        emit_f64s("K32_Kinv32", &vals);
        emit_f32("fx32", intrinsics_f32().fx);
        emit_f32("cx32", intrinsics_f32().cx);
    }

    // ── 2. projection ──────────────────────────────────────────────────────
    let obj = grid_object_points();
    let obj_f32: Vec<Point3<f32>> = obj
        .iter()
        .map(|p| Point3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    emit_f64s(
        "grid_obj",
        &obj.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
    );

    // Kannala-Brandt. OpenCV orders fisheye D as (k1, k2, k3, k4), which is
    // also the field order of `FisheyeDistortion`.
    let kb = FisheyeDistortion::new(KANNALA[0], KANNALA[1], KANNALA[2], KANNALA[3]);

    let radtan = Distortion::new(RADTAN[0], RADTAN[1], RADTAN[2], RADTAN[3], RADTAN[4]);
    let radtan32 = DistortionF32::new(
        RADTAN[0] as f32,
        RADTAN[1] as f32,
        RADTAN[2] as f32,
        RADTAN[3] as f32,
        RADTAN[4] as f32,
    );

    for (pname, p) in poses() {
        // `Rspec_*` is the Rodrigues matrix *before* it enters `Pose`. `Pose`
        // stores a UnitQuaternion, so `rotation_matrix()` round-trips through a
        // quaternion and differs from the constructed matrix in the last bits.
        // The reference asserts byte-identity on `Rspec_*` (the input contract)
        // and compares `R_*` at 1e-14 (the round-trip).
        {
            let raw = rodrigues(
                POSE_RVECS
                    .iter()
                    .find(|(n, _)| *n == pname)
                    .map(|(_, rv)| *rv)
                    .expect("named pose"),
            );
            let mut rv = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    rv.push(raw[(i, j)]);
                }
            }
            emit_f64s(&format!("Rspec_{}", pname), &rv);
        }
        let r = p.rotation_matrix();
        let mut rv = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                rv.push(r[(i, j)]);
            }
        }
        emit_f64s(&format!("R_{}", pname), &rv);
        emit_f64s(
            &format!("t_{}", pname),
            &[p.translation.x, p.translation.y, p.translation.z],
        );

        match project_points(&obj, &intrinsics(), &p) {
            Ok(pts) => emit_f64s(
                &format!("proj_{}", pname),
                &pts.iter().flat_map(|q| [q.x, q.y]).collect::<Vec<_>>(),
            ),
            Err(_) => emit_f64s(&format!("proj_{}", pname), &[]),
        }

        // Projection of a single known camera-frame ray. `PinholeModel::project`
        // takes camera coordinates, so no pose is involved here.
        let pin = PinholeModel::new(intrinsics(), Distortion::none());
        let pin32 = PinholeModelF32::new(intrinsics_f32(), DistortionF32::none());
        let ray = Point3::new(0.21, -0.13, 0.9);
        let q = pin.project(&ray);
        emit_f64s(
            &format!("pinhole_ray_{}", pname),
            &[ray.x, ray.y, ray.z, q.x, q.y],
        );
        let ray32 = Point3::new(0.21f32, -0.13f32, 0.9f32);
        let q32 = pin32.project(&ray32);
        emit_f64s(
            &format!("pinhole_ray32_{}", pname),
            &[
                ray32.x as f64,
                ray32.y as f64,
                ray32.z as f64,
                q32.x as f64,
                q32.y as f64,
            ],
        );
        // Zero-depth behaviour: documented as returning the principal point.
        let z0 = pin.project(&Point3::new(0.4, -0.7, 0.0));
        emit_f64s(&format!("pinhole_z0_{}", pname), &[z0.x, z0.y]);

        // Distorted projection: OpenCV `projectPoints` with D != 0.
        match project_points_with_distortion(&obj, &intrinsics(), &p, &radtan) {
            Ok(pts) => emit_f64s(
                &format!("projdist_{}", pname),
                &pts.iter().flat_map(|q| [q.x, q.y]).collect::<Vec<_>>(),
            ),
            Err(_) => emit_f64s(&format!("projdist_{}", pname), &[]),
        }
        {
            let m = PinholeModel::new(intrinsics(), radtan);
            let mut vals = Vec::with_capacity(obj.len() * 2);
            for o in obj.iter() {
                let pc = p.rotation * o.coords + p.translation;
                let q = m.project(&Point3::new(pc[0], pc[1], pc[2]));
                vals.push(q.x);
                vals.push(q.y);
            }
            emit_f64s(&format!("pinnradtan_{}", pname), &vals);
        }
        {
            // f32 twin, same inputs narrowed.
            let m = PinholeModelF32::new(intrinsics_f32(), radtan32);
            let mut vals = Vec::with_capacity(obj_f32.len() * 2);
            for o in obj_f32.iter() {
                let pc = p.rotation_matrix().cast::<f32>() * o.coords
                    + Vector3::new(
                        p.translation.x as f32,
                        p.translation.y as f32,
                        p.translation.z as f32,
                    );
                let q = m.project(&Point3::new(pc[0], pc[1], pc[2]));
                vals.push(q.x as f64);
                vals.push(q.y as f64);
            }
            emit_f64s(&format!("pinnradtan32_{}", pname), &vals);
        }

        // Kannala-Brandt: the crate's FisheyeDistortion with OpenCV ordering.
        let mut vals = Vec::with_capacity(obj.len() * 2);
        for o in obj.iter() {
            let pc = p.rotation * o.coords + p.translation;
            if pc[2].abs() <= 1e-12 {
                continue;
            }
            let (xd, yd) = kb.apply(pc[0] / pc[2], pc[1] / pc[2]);
            vals.push(intrinsics().fx * xd + intrinsics().cx);
            vals.push(intrinsics().fy * yd + intrinsics().cy);
        }
        emit_f64s(&format!("projkb_{}", pname), &vals);
        {
            let kb32 = FisheyeDistortionF32::new(
                KANNALA[0] as f32,
                KANNALA[1] as f32,
                KANNALA[2] as f32,
                KANNALA[3] as f32,
            );
            let mut vals = Vec::with_capacity(obj_f32.len() * 2);
            for o in obj_f32.iter() {
                let pc = p.rotation_matrix().cast::<f32>() * o.coords
                    + Vector3::new(
                        p.translation.x as f32,
                        p.translation.y as f32,
                        p.translation.z as f32,
                    );
                if pc[2].abs() <= 1e-7 {
                    continue;
                }
                let (xd, yd) = kb32.apply(pc[0] / pc[2], pc[1] / pc[2]);
                vals.push((intrinsics_f32().fx * xd + intrinsics_f32().cx) as f64);
                vals.push((intrinsics_f32().fy * yd + intrinsics_f32().cy) as f64);
            }
            emit_f64s(&format!("projkb32_{}", pname), &vals);
        }
    }

    // ── 3. distortion forward / inverse on a dense normalized grid ─────────
    {
        const N: usize = 41;
        let mut rng = Vec::new();
        for iy in 0..N {
            for ix in 0..N {
                let a = ix as f64 / (N - 1) as f64;
                let b = iy as f64 / (N - 1) as f64;
                rng.push(-1.35 + 2.7 * a);
                rng.push(-1.0 + 2.0 * b);
            }
        }
        emit_f64s("dist_grid", &rng);

        let mut fwd = Vec::with_capacity(rng.len());
        let mut inv = Vec::with_capacity(rng.len() + 1);
        let mut inv32 = Vec::with_capacity(rng.len() + 1);
        for c in rng.chunks(2) {
            let (dx, dy) = radtan.apply(c[0], c[1]);
            fwd.push(dx);
            fwd.push(dy);
            match radtan.remove_checked(dx, dy) {
                Some((ux, uy)) => {
                    inv.push(ux);
                    inv.push(uy);
                }
                None => {
                    inv.push(f64::NAN);
                    inv.push(f64::NAN);
                }
            }
            match radtan32.remove_checked(dx as f32, dy as f32) {
                Some((ux, uy)) => {
                    inv32.push(ux as f64);
                    inv32.push(uy as f64);
                }
                None => {
                    inv32.push(f64::NAN);
                    inv32.push(f64::NAN);
                }
            }
        }
        emit_f64s("radtan_fwd", &fwd);
        emit_f64s("radtan_inv", &inv);
        emit_f64s("radtan_inv32", &inv32);

        let mut kb_fwd = Vec::with_capacity(rng.len());
        let mut kb_inv = Vec::with_capacity(rng.len() + 1);
        for c in rng.chunks(2) {
            let (dx, dy) = kb.apply(c[0], c[1]);
            kb_fwd.push(dx);
            kb_fwd.push(dy);
            let (ux, uy) = kb.remove(dx, dy);
            kb_inv.push(ux);
            kb_inv.push(uy);
        }
        emit_f64s("kb_fwd", &kb_fwd);
        emit_f64s("kb_inv", &kb_inv);

        // Round-trip residual of the *distorted* grid through remove(): this is
        // the quantity that is model-independent and can be checked against
        // scipy.optimize.brentq on the same forward map.
        let mut rt = Vec::new();
        for c in fwd.chunks(2) {
            let (ux, uy) = radtan.remove(c[0], c[1]);
            rt.push((c[0] - ux).hypot(c[1] - uy));
        }
        emit_f64s("radtan_roundtrip_resid", &rt);
    }

    // ── 4. planar (Zhang) camera-matrix composition ────────────────────────
    {
        // Five views of the planar target. The rotations are deliberately
        // moderate (no near-degenerate homography) and include two tilts.
        let view_r = [
            [0.00, 0.00, 0.00],
            [0.22, -0.15, 0.04],
            [-0.19, 0.17, -0.06],
            [0.31, 0.24, 0.12],
            [-0.11, -0.28, 0.09],
        ];
        let view_t = [
            [-0.12, -0.09, 0.62],
            [-0.21, -0.02, 0.55],
            [0.03, -0.25, 0.70],
            [-0.02, -0.18, 0.48],
            [-0.28, 0.06, 0.66],
        ];
        for v in 0..5 {
            let p = pose(view_r[v], view_t[v]);
            emit_pose(&format!("zview_{}", v), &p);
            let proj = project_points(&obj, &intrinsics(), &p).expect("fronto-parallel view");
            emit_f64s(
                &format!("zobs_{}", v),
                &proj.iter().flat_map(|q| [q.x, q.y]).collect::<Vec<_>>(),
            );
        }
        let img_pts: Vec<Vec<Point2<f64>>> = (0..5)
            .map(|v| {
                let p = pose(view_r[v], view_t[v]);
                project_points(&obj, &intrinsics(), &p).expect("view")
            })
            .collect();
        let obj_pts: Vec<Vec<Point3<f64>>> = (0..5).map(|_| obj.clone()).collect();
        let opts = CameraCalibrationOptions {
            fix_principal_point: Some((CAM[2], CAM[3])),
            ..Default::default()
        };
        match calibrate_camera_planar_with_options(&obj_pts, &img_pts, (IMG_W, IMG_H), opts) {
            Ok(res) => {
                let k = res.intrinsics.matrix();
                emit_f64s("zhang_K", &matrix_row_major(&k));
                emit_scalar("zhang_rms", res.rms_reprojection_error);
                for (i, e) in res.extrinsics.iter().enumerate() {
                    emit_pose(&format!("zhang_pose_{}", i), e);
                }
            }
            Err(_) => {
                emit_f64s("zhang_K", &[]);
            }
        }
        // And the observations the reference must match: the *constructed*
        // Rodrigues matrices, not `Pose`'s quaternion round-trip.
        for v in 0..5 {
            let raw = rodrigues(view_r[v]);
            let mut rv = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    rv.push(raw[(i, j)]);
                }
            }
            emit_f64s(&format!("zR_{}", v), &rv);
            emit_f64s(&format!("zt_{}", v), &view_t[v]);
        }
    }

    // ── 5. homography / DLT ────────────────────────────────────────────────
    {
        let h_true = Matrix3::new(
            1.08, 0.041, 118.5, //
            -0.023, 0.91, 86.25, //
            0.000083, -0.00021, 1.0,
        );
        emit_f64s("H_true", &matrix_row_major(&h_true));

        let src: Vec<[f64; 2]> = (0..14)
            .map(|i| [45.0 + (i % 7) as f64 * 78.0, 30.0 + (i / 7) as f64 * 61.0])
            .collect();
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                let v = h_true * Vector3::new(p[0], p[1], 1.0);
                [v[0] / v[2], v[1] / v[2]]
            })
            .collect();
        emit_f64s(
            "H_src",
            &src.iter().flat_map(|p| [p[0], p[1]]).collect::<Vec<_>>(),
        );
        emit_f64s(
            "H_dst",
            &dst.iter().flat_map(|p| [p[0], p[1]]).collect::<Vec<_>>(),
        );

        match HomographySolver::estimate(&src, &dst) {
            Ok(h) => {
                let s = h[(2, 2)];
                emit_f64s("H_est", &scaled_matrix(&h, s));
                emit_f64s("H_true_norm", &scaled_matrix(&h_true, h_true[(2, 2)]));
            }
            Err(_) => emit_f64s("H_est", &[]),
        }

        // Degenerate: all source points collinear, destination not. No homography
        // exists and both sides must refuse.
        let col_src: Vec<[f64; 2]> = (0..8)
            .map(|i| [20.0 + 60.0 * i as f64, 2.0 * i as f64])
            .collect();
        let col_dst: Vec<[f64; 2]> = (0..8)
            .map(|i| [100.0 + 21.0 * i as f64, 30.0 + 44.0 * i as f64])
            .collect();
        emit_f64s(
            "H_collinear_src",
            &col_src
                .iter()
                .flat_map(|p| [p[0], p[1]])
                .collect::<Vec<_>>(),
        );
        emit_f64s(
            "H_collinear_dst",
            &col_dst
                .iter()
                .flat_map(|p| [p[0], p[1]])
                .collect::<Vec<_>>(),
        );
        match HomographySolver::estimate(&col_src, &col_dst) {
            Ok(h) => {
                let s = h[(2, 2)];
                emit_f64s("H_collinear_est", &scaled_matrix(&h, s));
            }
            Err(_) => emit_f64s("H_collinear_est", &[]),
        }

        // Degenerate: the minimal sample of exactly four points.
        let four_src: Vec<[f64; 2]> =
            vec![[10.0, 20.0], [610.0, 25.0], [600.0, 455.0], [15.0, 460.0]];
        let four_dst: Vec<[f64; 2]> = four_src
            .iter()
            .map(|p| {
                let v = h_true * Vector3::new(p[0], p[1], 1.0);
                [v[0] / v[2], v[1] / v[2]]
            })
            .collect();
        emit_f64s(
            "H_four_src",
            &four_src
                .iter()
                .flat_map(|p| [p[0], p[1]])
                .collect::<Vec<_>>(),
        );
        emit_f64s(
            "H_four_dst",
            &four_dst
                .iter()
                .flat_map(|p| [p[0], p[1]])
                .collect::<Vec<_>>(),
        );
        match HomographySolver::estimate(&four_src, &four_dst) {
            Ok(h) => {
                let s = h[(2, 2)];
                emit_f64s("H_four_est", &scaled_matrix(&h, s));
            }
            Err(_) => emit_f64s("H_four_est", &[]),
        }

        // Too few points.
        let three = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]];
        match HomographySolver::estimate(&three, &three) {
            Ok(h) => emit_f64s("H_three_est", &matrix_row_major(&h)),
            Err(_) => emit_f64s("H_three_est", &[]),
        }
    }

    // ── 6. essential / fundamental from correspondences ────────────────────
    {
        let r_true = rodrigues([0.041, -0.037, 0.026]);
        let t_true = Vector3::new(0.31, 0.048, 0.055);
        let p2 = Pose::new(r_true.clone(), t_true.clone());
        emit_pose("ep_relpose", &p2);

        // 24 synthetic 3D points on a deterministic lattice, all in front of
        // both cameras, plus 4 held out (indices >= 20) that neither estimator
        // is allowed to see.
        let mut world = Vec::new();
        for i in 0..24 {
            let a = (i % 6) as f64;
            let b = (i / 6) as f64;
            world.push(Point3::new(
                -0.45 + 0.18 * a + 0.013 * a * a,
                -0.30 + 0.15 * b + 0.009 * b * b,
                2.2 + 0.31 * ((i * 7) % 5) as f64 + 0.05 * (a - b),
            ));
        }
        let proj2 = project_points(&world, &intrinsics(), &p2).expect("all points in front");
        let proj1: Vec<Point2<f64>> = world
            .iter()
            .map(|p| {
                let q = intrinsics().project(p);
                Point2::new(q.x, q.y)
            })
            .collect();
        let flat1: Vec<f64> = proj1.iter().flat_map(|p| [p.x, p.y]).collect();
        let flat2: Vec<f64> = proj2.iter().flat_map(|p| [p.x, p.y]).collect();
        emit_f64s("ep_pts1", &flat1);
        emit_f64s("ep_pts2", &flat2);
        emit_f64s(
            "ep_world",
            &world
                .iter()
                .flat_map(|p| [p.x, p.y, p.z])
                .collect::<Vec<_>>(),
        );

        let e_true = cv_calib3d::essential_from_extrinsics(&p2);
        emit_f64s("E_true", &matrix_row_major(&e_true));
        let f_true = cv_calib3d::fundamental_from_essential(&e_true, &intrinsics(), &intrinsics());
        emit_f64s("F_true", &matrix_row_major(&f_true));

        // Ground-truth epipolar residuals, so the reference can compute
        // reference-side residuals with its own recovered matrices and compare
        // like with like.
        let sampson = |m: &Matrix3<f64>, a: &Point2<f64>, b: &Point2<f64>| -> f64 {
            let x1 = Vector3::new(a.x, a.y, 1.0);
            let x2 = Vector3::new(b.x, b.y, 1.0);
            let ex1 = m * x1;
            let etx2 = m.transpose() * x2;
            let num = x2.dot(&ex1);
            let den = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
            if den <= 1e-18 {
                f64::INFINITY
            } else {
                (num * num / den).sqrt()
            }
        };
        let mut res = Vec::new();
        for (a, b) in proj1.iter().zip(proj2.iter()) {
            res.push(sampson(&f_true, a, b));
        }
        emit_f64s("ep_resid_true", &res);

        // Estimation on the fit subset (first 20 correspondences).
        let fit1 = &proj1[..20];
        let fit2 = &proj2[..20];
        let held1 = &proj1[20..];
        let held2 = &proj2[20..];

        match find_essential_mat(fit1, fit2, &intrinsics()) {
            Ok(e) => {
                let s = e.norm().max(1e-300);
                emit_f64s("E_fit", &scaled_matrix(&e, s));
                emit_f64s("E_fit_rowmajor", &matrix_row_major(&e));
                let f = cv_calib3d::fundamental_from_essential(&e, &intrinsics(), &intrinsics());
                emit_f64s("F_from_E_fit", &matrix_row_major(&f));
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                for (a, b) in held1.iter().zip(held2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Efit", &r);
                match recover_pose_from_essential(&e, fit1, fit2, &intrinsics()) {
                    Ok(rec) => emit_pose("E_pose_recovered", &rec),
                    Err(_) => emit_f64s("E_pose_recovered", &[]),
                }
            }
            Err(_) => emit_f64s("E_fit", &[]),
        }

        match find_fundamental_mat(fit1, fit2) {
            Ok(f) => {
                let s = f.norm().max(1e-300);
                emit_f64s("F_fit", &scaled_matrix(&f, s));
                emit_f64s("F_fit_rowmajor", &matrix_row_major(&f));
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                for (a, b) in held1.iter().zip(held2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Ffit", &r);
            }
            Err(_) => emit_f64s("F_fit", &[]),
        }

        // Minimal 8-point sample: the documented historical weak spot.
        match find_fundamental_mat(&fit1[..8], &fit2[..8]) {
            Ok(f) => {
                let s = f.norm().max(1e-300);
                emit_f64s("F_fit8", &scaled_matrix(&f, s));
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Ffit8", &r);
            }
            Err(_) => emit_f64s("F_fit8", &[]),
        }
        match find_essential_mat(&fit1[..8], &fit2[..8], &intrinsics()) {
            Ok(e) => {
                let f = cv_calib3d::fundamental_from_essential(&e, &intrinsics(), &intrinsics());
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Efit8", &r);
            }
            Err(_) => emit_f64s("ep_resid_Efit8", &[]),
        }

        // Contaminated sample: 40% outliers, for the RANSAC comparison.
        let mut c1 = fit1.to_vec();
        let mut c2 = fit2.to_vec();
        for i in (0..c1.len()).step_by(5).take(6) {
            c1[i] = Point2::new(17.0 + 29.0 * i as f64, 431.0 - 13.0 * i as f64);
            c2[i] = Point2::new(403.0 - 21.0 * i as f64, 41.0 + 37.0 * i as f64);
        }
        emit_f64s(
            "ep_contam_pts1",
            &c1.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        );
        emit_f64s(
            "ep_contam_pts2",
            &c2.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        );
        match find_fundamental_mat_ransac(&c1, &c2, 1.5, 4000) {
            Ok((f, inl)) => {
                let s = f.norm().max(1e-300);
                emit_f64s("F_ransac", &scaled_matrix(&f, s));
                emit_f64s(
                    "F_ransac_inliers",
                    &inl.iter()
                        .map(|b| if *b { 1.0 } else { 0.0 })
                        .collect::<Vec<_>>(),
                );
                emit_scalar(
                    "F_ransac_n_inliers",
                    inl.iter().filter(|b| **b).count() as f64,
                );
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                for (a, b) in held1.iter().zip(held2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Fransac", &r);
            }
            Err(_) => emit_f64s("F_ransac", &[]),
        }
        match find_essential_mat_ransac(&c1, &c2, &intrinsics(), 1.5, 4000) {
            Ok((e, inl)) => {
                let f = cv_calib3d::fundamental_from_essential(&e, &intrinsics(), &intrinsics());
                emit_scalar(
                    "E_ransac_n_inliers",
                    inl.iter().filter(|b| **b).count() as f64,
                );
                let mut r = Vec::new();
                for (a, b) in fit1.iter().zip(fit2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                for (a, b) in held1.iter().zip(held2.iter()) {
                    r.push(sampson(&f, a, b));
                }
                emit_f64s("ep_resid_Eransac", &r);
            }
            Err(_) => emit_scalar("E_ransac_n_inliers", -1.0),
        }
    }

    // ── 7. the synthetic image for the features crate ──────────────────────
    {
        let img = feature_scene();
        emit_image(
            "feat_img",
            &img.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            FW,
            FH,
        );
    }

    out.flush().expect("stdout");
}
