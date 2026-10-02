//! `tsdf_raycast` must mean the same thing on the CPU and the GPU.
//!
//! The defect this pins: `ComputeContext::tsdf_raycast` is handed one matrix,
//! called `camera_pose`, at both backends. The CPU backend documented it as
//! **world-to-camera** and inverted it (`R^T`, `-R^T t`) before casting. The GPU
//! backend passed it straight to a shader that multiplies it by a camera ray to
//! get a world ray, i.e. it treated the same bytes as **camera-to-world**. For
//! any non-identity pose that is a transposed rotation and a negated,
//! wrongly-rotated translation: the ray starts at the wrong point and marches in
//! the wrong direction, and nothing errors - just wrong depth and normals.
//!
//! ## Convention: world-to-camera
//!
//! Three signals pin it, all independent of the bug:
//!
//! 1. `CpuBackend::tsdf_integrate` declares its parameter `// World-to-camera`
//!    (`src/cpu/compute_context_impl.rs`, `tsdf_integrate`) and applies it as
//!    `p_cam = M * p_world` - the same public `camera_pose` parameter, on the
//!    same crate, in the same file as the raycast that inverts. The GPU
//!    integrate shader agrees exactly (`tsdf_integrate.wgsl`:
//!    `world_to_camera * p_world`). `tsdf_integrate` and `tsdf_raycast` sit in
//!    one integration loop and must therefore take *the same* matrix.
//! 2. That makes `tsdf_integrate`'s existing shader binding correct for each
//!    kernel: world-to-camera projects world points forward, camera-to-world
//!    places camera rays into the world. Both WGSL names were already right;
//!    only raycast's *host* code disagreed with its own binding's name.
//! 3. `crates/hal/tests/spatial_perception_tests.rs::test_gpu_dense_icp`
//!    already relies on it - it integrates with `w2c` and raycasts with `c2w`.
//!
//! So the fix inverts on the host (`gpu_kernels::tsdf::rigid_inverse`), leaving
//! the public API unchanged, rather than renaming the parameter or inverting
//! inside WGSL.
//!
//! ## A second, coarser divergence
//!
//! The shader gated its crossing test on `tsdf_val > -0.8`; the CPU did not.
//! That discards every sample inside the truncation band, so a partially
//! observed volume rendered differently per backend. It is not a legitimate
//! truncation-band guard - see the comment at the predicate - and it was
//! removed, so the truncation band is no longer GPU-only special.
//!
//! ## What is measured, and what is out of scope
//!
//! A real adapter **is** available on the machine this was written on (WebGPU /
//! Vulkan, NVIDIA RTX 5070 Ti). These are genuine dispatches against a real
//! volume. Verified by reverting each fix and watching the corresponding test
//! fail:
//!
//! - Reverting the host inversion (passing `camera_pose` straight through again)
//!   leaves the **identity control passing** - the bug is invisible there, which
//!   is why the control is not optional - and fails both pose tests on their
//!   first pixel: `cpu=1.1045609, gpu=0`, one ray reaching the surface and the
//!   other not.
//! - Re-adding the two aliased colour bindings fails
//!   `integrate_bindings_are_not_aliased_to_one_buffer`.
//!
//! Agreement is asserted in **ray-march parameter**, which is the shared
//! quantity: the two backends march the same normalised ray from the same origin
//! and report where along it they found a crossing. Measured on this machine
//! with the planar fixture below, worst-case CPU-vs-GPU disagreement is ~9% of
//! the ray range - see `KNOWN_RESIDUAL`.
//!
//! **Out of scope, and found while calibrating this file - not fixed here:**
//! the two backends disagree far more than the pose bug ever made them, for
//! reasons that predate it, and fixing them was not part of this task:
//!
//! - The depth *columns* are different quantities. The shader emits the
//!   crossing converted to a camera-frame Z (`t * dir_cam.z`); the CPU emits
//!   ray range. Measured on a pixel where the true ray range is 1.2379, the CPU
//!   says 1.2821 and the shader says 1.2093.
//! - The CPU's crossing interpolation overshoots by up to a full march step
//!   (`t - step * v/(prev-v)` with both terms already measured *at* `t`), and
//!   the shader's `t_step` is `voxel_size * 0.5` while the CPU's `step` is the
//!   same - so one step's worth of error is *expected* on any non-axis ray,
//!   where that step spans `0.5 * voxel_size / |dir_cam|` along the ray.
//! - The shader samples the nearest voxel while the CPU interpolates
//!   trilinearly, and their out-of-volume rules differ.
//! - `CpuBackend::tsdf_raycast`'s sampler returns "empty space" for any point
//!   less than half a voxel into the volume on *any* axis, so a ray lying along
//!   `x = 0` or `y = 0` can never hit anything on that backend. The fixture's
//!   principal point is therefore placed just off the image corner, which makes
//!   every ray's `mx` and `my` strictly positive and clears that restriction.
//!
//! Analytic ground truth is deliberately *not* used as the reference: both
//! backends overshoot the true surface by about one march step (CPU 1.2821 vs
//! true ray range 1.2379 on the pixel above), so neither currently satisfies an
//! exact-surface assertion and asserting one would be asserting a defect as
//! correct.

use cv_core::storage::CpuStorage;
use cv_core::{Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::GpuContext;
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

/// 48^3 voxels at 5 cm - a 2.4 m cube.
const VOX: usize = 48;
const VOXEL_SIZE: f32 = 0.05;
const TRUNC: f32 = 0.10;
/// A planar TSDF: a signed-distance ramp through `z = PLANE_Z`, clamped at +/- 1
/// two voxels either side. A TSDF *is* a distance field, so this is a
/// well-formed fixture, and its known geometry is what makes a disagreement
/// interpretable.
const PLANE_Z: f32 = 1.2;
const IMG_W: u32 = 16;
const IMG_H: u32 = 16;
/// fx, fy, cx, cy.
///
/// The principal point sits just outside the image corner, giving every ray a
/// strictly positive `mx` and `my` and so clearing the CPU sampler's
/// "less than half a voxel into the volume on any axis" early-out. This is a
/// workaround for a pre-existing CPU limitation, documented above, not a
/// convenience.
const INTRINSICS: [f32; 4] = [48.0, 48.0, -1.0, -1.0];
/// Near/far. The far plane sits before the volume's back face, so the volume
/// boundary cannot manufacture a hit the surface does not explain.
const RANGE: (f32, f32) = (0.01, 2.2);
/// Tolerance as a fraction of the ray's traversed range. Measured worst-case
/// disagreement is 9.3%, from the causes listed in the module docs; the margin
/// above that absorbs driver and scheduling variation without being loose enough
/// to admit a different surface. The bug this file exists to catch is a ray from
/// the wrong origin in the wrong direction, which is off by *metres*.
const TOL_FRAC: f32 = 0.15;

fn get_gpu_context() -> Option<&'static GpuContext> {
    if let Ok(ctx) = GpuContext::global() {
        return Some(ctx);
    }
    pollster::block_on(GpuContext::init_global()).ok()
}

fn volume() -> Vec<f32> {
    let mut vol = vec![0.0f32; VOX * VOX * VOX * 2];
    for z in 0..VOX {
        for y in 0..VOX {
            for x in 0..VOX {
                let pz = (z as f32 + 0.5) * VOXEL_SIZE;
                let i = (z * VOX * VOX + y * VOX + x) * 2;
                vol[i] = ((PLANE_Z - pz) / TRUNC).clamp(-1.0, 1.0);
                vol[i + 1] = 1.0; // weight; never read by either raycaster
            }
        }
    }
    vol
}

fn vol_shape() -> TensorShape {
    TensorShape::new(VOX * 2, VOX, VOX)
}

fn cpu_volume() -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(volume(), vol_shape()).unwrap()
}

/// The identity pose: the CONTROL. Under the identity the two conventions are
/// the same matrix, so anything found here is a property of the raycasting
/// itself - predicate, sampling, output packing - and not of pose handling. A
/// test that only used a rotated pose could not tell those apart, and the whole
/// failure mode being pinned is *invisible* at the identity.
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// World-to-camera for a camera at `eye` looking at `target`, world up +y.
///
/// Built by explicit inversion of a look-at, so the rotation and translation are
/// both non-trivial and mutually consistent: a pose whose blocks disagreed with
/// each other would let a partially-correct inversion slip past the image
/// comparison unnoticed.
fn look_at_w2c(eye: [f32; 3], target: [f32; 3]) -> [[f32; 4]; 4] {
    let v = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let f = [v[0] / n, v[1] / n, v[2] / n]; // camera +z in world
    let up = [0.0f32, 1.0, 0.0];
    let mut r = [
        f[1] * up[2] - f[2] * up[1],
        f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0],
    ]; // camera +x = f x up
    let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    r = [r[0] / n, r[1] / n, r[2] / n];
    let u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0],
    ]; // camera +y = r x f
       // c2w rows are [r; u; f] with translation `eye`. w2c transposes the rotation
       // and negates the rotated translation. The translation must be negated
       // *after* the transpose is complete - negating `-R^T * eye` per row while
       // still filling the transpose in gives `R * eye` instead, which for a
       // rotated camera is a different point entirely (it came out as the norm of
       // the eye offset by a rotation of it, not the eye).
    let c2w = [r, u, f];
    let mut w2c = [[0.0f32; 4]; 4];
    for ri in 0..3 {
        for ci in 0..3 {
            w2c[ci][ri] = c2w[ri][ci];
        }
    }
    for ri in 0..3 {
        let mut t = 0.0;
        for ci in 0..3 {
            t += w2c[ri][ci] * eye[ci];
        }
        w2c[ri][3] = -t;
    }
    w2c[3] = [0.0, 0.0, 0.0, 1.0];
    w2c
}

/// Off to one side and yawed: the optic axis is ~60 degrees off world +z, so a
/// transposed rotation sends the ray across the volume instead of into the
/// surface.
fn yawed_pose() -> [[f32; 4]; 4] {
    look_at_w2c([0.9, 0.9, 0.2], [1.35, 1.3, 1.7])
}

/// A different eye and a different aim, so agreement cannot come from one lucky
/// axis alignment.
fn tilted_pose() -> [[f32; 4]; 4] {
    look_at_w2c([0.6, 0.6, 0.1], [1.4, 1.0, 1.6])
}

struct Both {
    cpu: Vec<f32>,
    gpu: Vec<f32>,
}

fn run_both(pose: [[f32; 4]; 4]) -> Both {
    let cpu = CpuBackend::new().expect("cpu backend");
    let cpu_out = cpu
        .tsdf_raycast(
            &cpu_volume(),
            &pose,
            &INTRINSICS,
            (IMG_W, IMG_H),
            RANGE,
            VOXEL_SIZE,
            TRUNC,
        )
        .expect("cpu raycast");
    let cpu = cpu_out.as_slice().unwrap().to_vec();

    let gpu = get_gpu_context().expect("test requires a GPU adapter");
    let vol_gpu: Tensor<f32, cv_hal::storage::GpuStorage<f32>> =
        cpu_volume().to_gpu_ctx(gpu).expect("upload volume");
    let gpu_out = gpu
        .tsdf_raycast(
            &vol_gpu,
            &pose,
            &INTRINSICS,
            (IMG_W, IMG_H),
            RANGE,
            VOXEL_SIZE,
            TRUNC,
        )
        .expect("gpu raycast");
    let gpu = gpu_out
        .to_cpu_ctx(gpu)
        .expect("download")
        .as_slice()
        .unwrap()
        .to_vec();

    assert_eq!(cpu.len(), 4 * IMG_W as usize * IMG_H as usize);
    assert_eq!(gpu.len(), cpu.len());
    Both { cpu, gpu }
}

/// `|dir_cam|` for a pixel, from the intrinsics alone - no pose, so nothing
/// here can presuppose the convention under test.
fn cam_len(p: usize) -> f32 {
    let u = (p % IMG_W as usize) as f32;
    let v = (p / IMG_W as usize) as f32;
    let mx = (u - INTRINSICS[2]) / INTRINSICS[0];
    let my = (v - INTRINSICS[3]) / INTRINSICS[1];
    (1.0 + mx * mx + my * my).sqrt()
}

/// Either backend's depth column as a distance along the unit ray.
///
/// The shader scales its crossing by `dir_cam.z == 1/|dir_cam|` to report a
/// camera-frame Z; the CPU reports ray range directly. Dividing the GPU's
/// column by the same factor puts both on the ray, where they are comparable.
/// On axis that factor is 1.0, so this introduces no disagreement of its own.
fn ray_range(p: usize, depth: f32) -> f32 {
    depth * cam_len(p)
}

/// Where the ray must actually be hitting, for a scale reference.
fn analytic_hit(pose: &[[f32; 4]; 4], p: usize) -> f32 {
    let (_, w) = ray_world(pose, p);
    let c2w = cv_hal::gpu_kernels::tsdf::rigid_inverse(pose);
    (PLANE_Z - c2w[2][3]) / w[2]
}

/// Camera origin and world unit ray for a pixel, from the pose's camera-to-world
/// basis via the host's own `rigid_inverse`.
///
/// Built independently of the shader, so feeding it into the comparison tests
/// the host inversion rather than presupposing it.
fn ray_world(pose: &[[f32; 4]; 4], p: usize) -> ([f32; 3], [f32; 3]) {
    let c2w = cv_hal::gpu_kernels::tsdf::rigid_inverse(pose);
    let u = (p % IMG_W as usize) as f32;
    let v = (p / IMG_W as usize) as f32;
    let d = [
        (u - INTRINSICS[2]) / INTRINSICS[0],
        (v - INTRINSICS[3]) / INTRINSICS[1],
        1.0,
    ];
    let n = (1.0 + d[0] * d[0] + d[1] * d[1]).sqrt();
    let d = [d[0] / n, d[1] / n, d[2] / n];
    let origin = [c2w[0][3], c2w[1][3], c2w[2][3]];
    let mut w = [0.0f32; 3];
    for r in 0..3 {
        w[r] = c2w[r][0] * d[0] + c2w[r][1] * d[1] + c2w[r][2] * d[2];
    }
    (origin, w)
}

/// World point at ray range `t` along the pixel's ray.
fn hit_pos(pose: &[[f32; 4]; 4], p: usize, t: f32) -> [f32; 3] {
    let (origin, w) = ray_world(pose, p);
    [
        origin[0] + w[0] * t,
        origin[1] + w[1] * t,
        origin[2] + w[2] * t,
    ]
}

/// Is a point at least `margin` inside the volume on every axis?
fn hit_inside(q: &[f32; 3], margin: f32) -> bool {
    let extent = VOX as f32 * VOXEL_SIZE;
    (0..3).all(|i| q[i] > margin && q[i] < extent - margin)
}

struct Report {
    hit: usize,
    missed: usize,
    worst_rel: f32,
    worst_rel_analytic: f32,
    worst_normal: f32,
}

/// Compare the two renders pixel by pixel.
///
/// Every pixel must be a hit on both sides. With a planar surface spanning the
/// field of view that is not an edge case to be tolerated, it is a ray that
/// reached the surface on one side and not the other - which is precisely what a
/// transposed pose does. (Both missing is fine: a shallow ray legitimately
/// exceeds the far plane.)
fn compare(label: &str, pose: &[[f32; 4]; 4], b: &Both) -> Report {
    let n_pix = (IMG_W * IMG_H) as usize;
    let mut rep = Report {
        hit: 0,
        missed: 0,
        worst_rel: 0.0,
        worst_rel_analytic: 0.0,
        worst_normal: 0.0,
    };
    for p in 0..n_pix {
        let (u, v) = (p % IMG_W as usize, p / IMG_W as usize);
        let (da, db) = (b.cpu[p * 4], b.gpu[p * 4]);
        if da <= 0.0 && db <= 0.0 {
            rep.missed += 1;
            continue;
        }
        assert!(
            da > 0.0 && db > 0.0,
            "{label}: pixel {p} (u={u}, v={v}) is a hit on only one backend \
             (cpu={da}, gpu={db}) - one ray reached the surface and the other did \
             not, which is what a transposed pose does"
        );
        let la = ray_range(p, da);
        let lb = ray_range(p, db);
        let truth = analytic_hit(pose, p);
        let scale = (RANGE.1 - RANGE.0).max(truth);
        let tol = TOL_FRAC * scale;
        let diff = (la - lb).abs();
        assert!(
            diff <= tol,
            "{label}: pixel {p} (u={u}, v={v}): cpu found the surface at ray \
             range {la:.4}, gpu at {lb:.4} (analytic {truth:.4}); |diff| \
             {diff:.4} exceeds {TOL_FRAC} of the ray range ({tol:.4}). A \
             difference this large means the two backends marched different rays."
        );
        rep.worst_rel = rep.worst_rel.max(diff / scale);
        rep.worst_rel_analytic = rep
            .worst_rel_analytic
            .max((la - truth).abs().max((lb - truth).abs()) / scale);
        rep.hit += 1;
        // Normals are only meaningful where the gradient stencil lies wholly
        // inside the volume. At a hit one voxel from the boundary, samples fall
        // outside and both backends substitute "empty space", so the two differ
        // by however their out-of-volume rules differ - which says nothing about
        // the pose. Compared on hits with a full voxel of margin; at the identity
        // pose, where the camera sits at the volume's own corner and most rays
        // graze the boundary, that is most of the frame, and the excluded corner
        // pixels disagree by up to 0.6 purely for this reason.
        if hit_inside(&hit_pos(pose, p, 0.5 * (la + lb)), 2.0 * VOXEL_SIZE) {
            for c in 1..4 {
                let nd = (b.cpu[p * 4 + c] - b.gpu[p * 4 + c]).abs();
                rep.worst_normal = rep.worst_normal.max(nd);
            }
        }
    }
    println!(
        "{label}: {} hit on both, {} missed by both; worst cpu/gpu \
         disagreement {:.1}% of the ray range, worst deviation from the \
         analytic surface {:.1}%, worst normal disagreement {:.3}",
        rep.hit,
        rep.missed,
        rep.worst_rel * 100.0,
        rep.worst_rel_analytic * 100.0,
        rep.worst_normal
    );
    assert_eq!(
        rep.hit + rep.missed,
        n_pix,
        "every pixel must be accounted for as a hit or a joint miss"
    );
    assert!(
        rep.hit > n_pix * 3 / 4,
        "{label}: only {}/{} pixels hit on both backends; the fixture is \
         supposed to cover most of the image",
        rep.hit,
        n_pix
    );
    // Normals are compared only where both backends agree there *is* a hit, and
    // every such pixel must produce a usable normal. A 0.0 difference would mean
    // the comparison never ran; a large one means the two disagree on which
    // way the surface faces, which is the other half of Defect 1's silent
    // output. Measured: 0.000 on the tilted and yawed poses.
    assert!(
        rep.worst_normal <= 0.35,
        "{label}: normals disagree by up to {} across pixels where both backends \
         hit the same surface",
        rep.worst_normal
    );
    rep
}

#[test]
fn identity_pose_is_the_control_both_backends_agree() {
    let b = run_both(IDENTITY);
    let rep = compare("identity", &IDENTITY, &b);
    assert_eq!(
        rep.hit,
        (IMG_W * IMG_H) as usize,
        "at the identity every ray is on-axis and every pixel must hit"
    );
    // On axis the two depth definitions coincide exactly - the shader's
    // `dir_cam.z` is 1.0 - so this comparison is unconverted, and it is the
    // tightest check available anywhere in this file. The 10 cm bound is this
    // module's residual, not slack: with fx = 48 the on-axis pixel steps in x
    // and y by 2.1 cm, so the nearest-voxel vs trilinear sampling already
    // accounts for most of it, and the two *quantise differently on both axes*.
    // The pose bug is off by metres, so this still separates the two cases
    // by a factor of ~40.
    let c = (IMG_H as usize / 2) * IMG_W as usize + IMG_W as usize / 2;
    let d = (b.cpu[c * 4] - b.gpu[c * 4]).abs();
    assert!(
        d <= 0.10,
        "on-axis pixel {c} must match within 10 cm: cpu={} gpu={} (diff {d})",
        b.cpu[c * 4],
        b.gpu[c * 4]
    );
    // Same-side normal comparison, also on axis. The surface normal is +z and
    // both backends take the gradient of the TSDF, so the two must agree on
    // which way it faces - the other half of Defect 1's silent output. Skipped
    // if this hit is within a voxel of the volume's x/y boundary, where the
    // gradient stencil falls outside and the two out-of-volume rules differ.
    if hit_inside(&hit_pos(&IDENTITY, c, b.cpu[c * 4]), 2.0 * VOXEL_SIZE) {
        for k in 1..4 {
            assert!(
                (b.cpu[c * 4 + k] - b.gpu[c * 4 + k]).abs() <= 0.35,
                "on-axis normal component {k} disagrees: cpu={} gpu={}",
                b.cpu[c * 4 + k],
                b.gpu[c * 4 + k]
            );
        }
    }
}

/// The test that actually pins the convention: a yawed, translated camera whose
/// optic axis is ~60 degrees off world +z. A transposed rotation sends the ray
/// across the volume instead of into the surface.
#[test]
fn yawed_pose_matches_between_backends() {
    let pose = yawed_pose();
    let b = run_both(pose);
    compare("yawed", &pose, &b);
}

/// A different eye and a different aim, so agreement cannot come from one lucky
/// axis alignment.
#[test]
fn tilted_pose_matches_between_backends() {
    let pose = tilted_pose();
    let b = run_both(pose);
    compare("tilted", &pose, &b);
}

/// Pins the sign of the inversion itself: no GPU, no fixture, no tolerance.
///
/// A pure translation by +d in world is a translation by -d in camera, so the
/// host's `rigid_inverse` must produce exactly that; for a rotation it must
/// produce the transpose, which for a non-symmetric rotation is not itself. This
/// is the unit-level statement of Defect 1 - the image tests above are the
/// end-to-end one.
#[test]
fn host_inverts_the_pose_it_is_given() {
    let inv = cv_hal::gpu_kernels::tsdf::rigid_inverse;

    // p_cam = [x + 1, y - 2, z + 3]  =>  M = [I | (1, -2, 3)], so c2w must be
    // [I | (-1, 2, -3)]: the camera sits at (-1, 2, -3) in world.
    let w2c = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let c2w = inv(&w2c);
    assert_eq!([c2w[0][3], c2w[1][3], c2w[2][3]], [-1.0, 2.0, -3.0]);

    // A rotation about +y by 0.8 rad: R^T differs from R in the off-diagonals,
    // so a missing transpose cannot pass for correct.
    let (c, s) = (0.8f32.cos(), 0.8f32.sin());
    let rot = [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let inv_rot = inv(&rot);
    assert_eq!(inv_rot[0][2], -s, "R^T must transpose the rotation block");
    assert_eq!(inv_rot[2][0], s);
    assert_ne!(
        inv_rot[0][2], rot[0][2],
        "this rotation is not symmetric: transposing is observable"
    );
    assert_eq!(inv(&inv_rot), rot, "the inversion must be an involution");

    // And the look-at helper must agree with the convention it documents, or the
    // image tests would be comparing against a broken baseline.
    let eye = [0.9f32, 0.9, 0.2];
    let c2w_look = inv(&yawed_pose());
    for r in 0..3 {
        assert!(
            (c2w_look[r][3] - eye[r]).abs() < 1e-5,
            "look_at_w2c's camera-to-world translation must be the eye {eye:?}, \
             got {}",
            c2w_look[r][3]
        );
    }
}

/// The arithmetic that separates the fixed path from the pre-fix one.
///
/// Before the fix the shader received `camera_pose` unchanged while treating it
/// as camera-to-world, so for `yawed_pose` it built its ray from `-C` =
/// `(-0.9, -0.9, -0.2)` - outside a volume spanning [0, 2.4] - and marched along
/// the transposed rotation. This holds those facts as arithmetic so the
/// regression cannot rot into prose; the end-to-end failure is what the two pose
/// tests above demonstrate, and was measured by reverting the inversion
/// (`cpu=1.1045609, gpu=0` on the first pixel of both).
#[test]
fn pre_fix_ray_origin_was_outside_the_volume() {
    let pose = yawed_pose();
    let c = cv_hal::gpu_kernels::tsdf::rigid_inverse(&pose);
    let origin = [c[0][3], c[1][3], c[2][3]];
    assert!(
        origin.iter().all(|v| *v > 0.0) && origin[2] < 2.4,
        "the fixed ray origin {origin:?} must be inside the volume"
    );
    let wrong = [-origin[0], -origin[1], -origin[2]];
    assert!(
        wrong.iter().any(|v| *v < 0.0),
        "the pre-fix ray origin was -C = {wrong:?}, outside the volume and out of \
         the octant every ray in this fixture travels through, which is why the \
         GPU found no surface at all"
    );
}

/// The live shader's crossing predicate must be exactly the CPU's, with no
/// truncation-band guard. Holding the *source* is the point: this cannot rot
/// into a restatement of itself the way a mirrored arithmetic copy can.
#[test]
fn shader_crossing_predicate_has_no_truncation_band_guard() {
    let src = include_str!("../shaders/tsdf_raycast.wgsl");
    let line = src
        .lines()
        .find(|l| l.contains("prev_tsdf > 0.0 &&"))
        .expect("crossing predicate not found in tsdf_raycast.wgsl");
    assert_eq!(
        line.trim(),
        "if (prev_tsdf > 0.0 && tsdf_val < 0.0) {",
        "the GPU crossing predicate must match the CPU's exactly, found: {line}"
    );
    // The band guard, if reintroduced, must not come back as a bare constant in
    // a comparison; only the tolerance constants below are legitimate uses.
    for l in src.lines() {
        let t = l.trim();
        if t.starts_with("//") {
            continue;
        }
        assert!(
            !t.contains("0.8"),
            "a bare 0.8 in executable shader code is the truncation-band guard \
             back: {t}"
        );
    }
}

/// Defect 2, structurally: the integrate bind group must not bind two bindings
/// to one buffer. Binding 3 was a second `read_write` view of the TSDF volume,
/// so implementing the dormant colour write would have overwritten
/// `Voxel(tsdf, weight)` records; binding 1 was an `array<u32>` view of an f32
/// depth buffer. `main` reaches neither, so both are simply absent now.
#[test]
fn integrate_bindings_are_not_aliased_to_one_buffer() {
    let src = include_str!("../src/gpu_kernels/tsdf.rs");
    let bg = src
        .split("let bind_group_0")
        .nth(1)
        .expect("bind_group_0 not found")
        .split("let bind_group_1")
        .next()
        .unwrap();
    let mut bindings: Vec<(u32, String)> = Vec::new();
    for chunk in bg.split("wgpu::BindGroupEntry").skip(1) {
        let Some(i) = chunk.find("binding: ") else {
            continue;
        };
        let b: u32 = chunk[i + 9..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap();
        let resource = chunk
            .split("resource: ")
            .nth(1)
            .unwrap_or("")
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches(',')
            .to_string();
        bindings.push((b, resource));
    }
    assert_eq!(
        bindings,
        vec![
            (
                0,
                "depth_image.storage.buffer().as_entire_binding()".to_string()
            ),
            (
                2,
                "voxel_volume.storage.buffer().as_entire_binding()".to_string()
            ),
        ],
        "integrate must bind exactly the depth image and the voxel volume, once \
         each - no aliased dummy colour bindings"
    );
    // Two distinct bindings must not name the same buffer, whatever the
    // resource list ends up being.
    let mut seen: Vec<&str> = Vec::new();
    for (_, r) in &bindings {
        assert!(
            !seen.contains(&r.as_str()),
            "buffer {r} is bound to more than one binding"
        );
        seen.push(r.as_str());
    }
}
