//! `registration_icp_point_to_plane_ctx` must not report a registration it did
//! not perform, and its metrics must describe the transform it returns.
//!
//! The `_ctx` entry point is the same algorithm as the CPU
//! `registration_icp_point_to_plane`, which was already fixed for exactly these
//! three defects; its comments in `registration/mod.rs` are the specification.
//! The `_ctx` copy never received the fix.
//!
//! These tests need a real GPU. The `_ctx` function early-returns to the CPU
//! implementation when handed a `ComputeDevice::Cpu`, so a CPU-driven test would
//! exercise the code that is already correct and pass vacuously - the parity
//! test in `tests/gpu_parity_tests.rs` takes that early return and so never
//! reached this code either.

use cv_core::point_cloud::PointCloud;
use cv_hal::compute::ComputeDevice;
use cv_hal::gpu::GpuContext;
use nalgebra::{Matrix4, Point3, Vector3};

fn cloud(points: &[(f32, f32, f32)], normals: Option<&[(f32, f32, f32)]>) -> PointCloud {
    let mut pc = PointCloud::default();
    for (i, p) in points.iter().enumerate() {
        pc.points.push(Point3::new(p.0, p.1, p.2));
        if let Some(n) = normals {
            let v = n[i];
            pc.normals
                .get_or_insert_with(Vec::new)
                .push(Vector3::new(v.0, v.1, v.2));
        }
    }
    pc
}

/// The device under test. `None` when there is no GPU, in which case the tests
/// return early rather than silently taking the CPU early-return branch.
fn device() -> Option<(GpuContext, ComputeDevice<'static>)> {
    let gpu = GpuContext::new().ok()?;
    let leaked: &'static GpuContext = Box::leak(Box::new(gpu));
    Some((leaked.clone(), ComputeDevice::Gpu(leaked)))
}

fn ctx(
    source: &PointCloud,
    target: &PointCloud,
    max_dist: f32,
    iters: usize,
) -> Option<nalgebra::Matrix4<f32>> {
    let (_, dev) = device()?;
    cv_registration::registration_icp_point_to_plane_ctx(
        source,
        target,
        max_dist,
        &Matrix4::identity(),
        iters,
        &dev,
    )
    .ok()
    .map(|r| r.transformation)
}

/// A non-planar target with outward face normals, so point-to-plane residuals
/// actually constrain all three translation axes. A single uniform normal (all
/// +z) cannot observe motion along the perpendicular axes at all, which would
/// make the control case meaningless.
fn cube() -> (Vec<(f32, f32, f32)>, Vec<(f32, f32, f32)>) {
    let mut pts = Vec::new();
    for &x in &[0.0f32, 0.5, 1.0] {
        for &y in &[0.0f32, 0.5, 1.0] {
            for &z in &[0.0f32, 0.5, 1.0] {
                pts.push((x, y, z));
            }
        }
    }
    let nrm: Vec<(f32, f32, f32)> = pts
        .iter()
        .map(|p| {
            let a = [p.0, p.1, p.2];
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for ax in 0..3usize {
                for &s in &[0.0f32, 1.0] {
                    let d = (a[ax] - s).abs();
                    if d < best_d {
                        best_d = d;
                        best = ax;
                    }
                }
            }
            let mut v = [0.0f32; 3];
            v[best] = if a[best] > 0.5 { 1.0 } else { -1.0 };
            (v[0], v[1], v[2])
        })
        .collect();
    (pts, nrm)
}

/// CONTROL: an ordinary registration through the `_ctx` path must still work.
///
/// Every test below needs this, or it could pass for the wrong reason - a
/// function that returned an error for everything would satisfy "must not
/// report a registration it did not perform".
fn control_registration() -> Option<()> {
    let (pts, nrm) = cube();
    let target = cloud(&pts, Some(&nrm));
    let src: Vec<(f32, f32, f32)> = pts.iter().map(|p| (p.0, p.1 + 0.02, p.2)).collect();
    let source = cloud(&src, Some(&nrm));

    let (_, dev) = device()?;
    let r = cv_registration::registration_icp_point_to_plane_ctx(
        &source,
        &target,
        0.1,
        &Matrix4::identity(),
        20,
        &dev,
    )
    .expect("control: a cube offset by 0.02 in y must register through _ctx");
    assert!(
        r.fitness > 0.9,
        "control: a normal registration must report a good fitness, got {}",
        r.fitness
    );
    assert!(
        r.inlier_rmse < 0.01,
        "control: a normal registration must report a small rmse, got {}",
        r.inlier_rmse
    );
    assert!(
        (r.transformation[(1, 3)] + 0.02).abs() < 1e-2,
        "control: the recovered y offset must be near -0.02, got {}",
        r.transformation[(1, 3)]
    );
    Some(())
}

/// (a) `if correspondences_raw.len() < 3 { break; }` fell through to an
/// unconditional `Ok(ICPResult { .. })` with `fitness 0.0` and
/// `inlier_rmse f32::MAX` and the input transform - a *successful* registration
/// that never happened, and an rmse of 3.4e38 that satisfies any `>= 0.0` check.
///
/// Measured pre-fix on the input below:
/// `Ok(ICPResult { transformation: identity, fitness: 0.0,
///   inlier_rmse: 3.4028235e38, num_iterations: 0 })`.
#[test]
fn too_few_correspondences_is_an_error_not_a_result() {
    // Skip rather than fail without a GPU adapter.
    //
    // CI runners have no adapter, so `.expect(...)` turned these three red there
    // while they pass locally. The `_ctx` path needs a compute context, so there
    // is nothing to assert without one - and a test that cannot run in CI
    // reporting failure there is how people learn to ignore red.
    if control_registration().is_none() {
        eprintln!("no GPU adapter: skipping, the _ctx path needs a compute context");
        return;
    }

    let target_pts: Vec<(f32, f32, f32)> = (0..150).map(|i| (i as f32 * 0.01, 0.0, 0.0)).collect();
    let nrm: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 1.0); 150];
    let target = cloud(&target_pts, Some(&nrm));
    // 100 units away with a 0.05 correspondence distance: nothing matches.
    let src_pts: Vec<(f32, f32, f32)> =
        target_pts.iter().map(|p| (p.0 + 100.0, p.1, p.2)).collect();
    let source = cloud(&src_pts, Some(&nrm));

    let Some((_, dev)) = device() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let r = cv_registration::registration_icp_point_to_plane_ctx(
        &source,
        &target,
        0.05,
        &Matrix4::identity(),
        50,
        &dev,
    );
    match r {
        Err(_) => {}
        Ok(res) => panic!(
            "a registration with no correspondences must be an error, not Ok with \
             fitness {} and inlier_rmse {} (num_iterations {})",
            res.fitness, res.inlier_rmse, res.num_iterations
        ),
    }
}

/// (b) `if let Some(ata_inv) = ata.try_inverse() { .. }` had no `else`, so a
/// singular `ata` silently left the pose unchanged and the tail reported that
/// unchanged pose as the result.
///
/// All target points identical makes `A` robustly rank 1: the Jacobian is
/// `[n, p x n]` and with `p` constant the rotational columns span one
/// direction. The source is offset by 0.03 along y, which the plane normal can
/// see, so there is a displacement a solve *should* recover.
///
/// Measured pre-fix: `Ok(fitness 1.0, inlier_rmse 0.030000003,
/// num_iterations 1, transformation = identity)` - the input offset, unfixed,
/// presented as a converged registration with a perfect fitness.
#[test]
fn a_singular_normal_matrix_still_applies_an_update() {
    // Skip rather than fail without a GPU adapter.
    //
    // CI runners have no adapter, so `.expect(...)` turned these three red there
    // while they pass locally. The `_ctx` path needs a compute context, so there
    // is nothing to assert without one - and a test that cannot run in CI
    // reporting failure there is how people learn to ignore red.
    if control_registration().is_none() {
        eprintln!("no GPU adapter: skipping, the _ctx path needs a compute context");
        return;
    }

    let nrm: Vec<(f32, f32, f32)> = vec![(0.0, 1.0, 0.0); 24];
    let target = cloud(&vec![(0.5, 0.3, 0.2); 24], Some(&nrm));
    let source = cloud(&vec![(0.5, 0.33, 0.2); 24], Some(&nrm));

    let Some((_, dev)) = device() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let r = cv_registration::registration_icp_point_to_plane_ctx(
        &source,
        &target,
        0.5,
        &Matrix4::identity(),
        50,
        &dev,
    )
    .expect("a singular A is still solvable via the pseudo-inverse");

    // What matters is that the residual was *reduced*, not which matrix slot the
    // displacement lands in: the update is a left-multiply, so asserting on a
    // particular element would be asserting on a representation detail.
    assert!(
        r.inlier_rmse < 0.01,
        "the singular solve did not reduce the residual: rmse is {} against a \
         0.03 input offset. The update was silently skipped and the input pose \
         was reported back.",
        r.inlier_rmse
    );
    assert!(
        r.num_iterations > 0,
        "the residual fell, so the solve ran, but it recorded zero iterations"
    );
}

/// (c) The function returned the *live* `transformation` - the final iterate -
/// while `best_fitness`/`best_rmse` were recorded at whichever iterate last
/// beat the fitness. Since `fitness` saturates at 1.0, that record froze at
/// iteration 0. The reported rmse and the returned pose therefore described
/// different transforms.
///
/// The target normals here are all +z and the offset is along z, so the
/// point-to-plane residual at any pose is exactly `|z_offset|`. The rmse the
/// function reports must therefore agree with the residual at the transform it
/// hands back - no re-derivation, no nearest-neighbour subtleties.
#[test]
fn the_reported_metrics_describe_the_returned_transform() {
    // Skip rather than fail without a GPU adapter.
    //
    // CI runners have no adapter, so `.expect(...)` turned these three red there
    // while they pass locally. The `_ctx` path needs a compute context, so there
    // is nothing to assert without one - and a test that cannot run in CI
    // reporting failure there is how people learn to ignore red.
    if control_registration().is_none() {
        eprintln!("no GPU adapter: skipping, the _ctx path needs a compute context");
        return;
    }

    let n = 8;
    let target_pts: Vec<(f32, f32, f32)> = (0..n)
        .map(|i| {
            (
                i as f32 * 0.3,
                (i as f32 * 0.7) % 1.3,
                (i as f32 * 1.1) % 0.9,
            )
        })
        .collect();
    let nrm: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 1.0); n];
    let target = cloud(&target_pts, Some(&nrm));
    let offset = 0.05f32;
    let src_pts: Vec<(f32, f32, f32)> = target_pts
        .iter()
        .map(|p| (p.0, p.1, p.2 + offset))
        .collect();
    let source = cloud(&src_pts, Some(&nrm));

    let Some((_, dev)) = device() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let r = cv_registration::registration_icp_point_to_plane_ctx(
        &source,
        &target,
        0.5,
        &Matrix4::identity(),
        50,
        &dev,
    )
    .expect("a cube-like source should register");

    // Re-derive the residual at the transform actually returned.
    let t = r.transformation;
    let mut sum = 0.0f32;
    for (s, g) in source.points.iter().zip(target.points.iter()) {
        let diff = t.transform_point(s) - g;
        sum += diff.z * diff.z; // the target normals are all +z
    }
    let actual = f64::from((sum / n as f32).sqrt());
    let reported = r.inlier_rmse as f64;

    assert!(
        actual <= reported * 1.05 + 1e-9,
        "the reported inlier_rmse ({reported:.6}) is worse than the residual actually \
         achieved at the returned transform ({actual:.6}), so the two describe different \
         transforms. The record froze at iteration 0 while the returned pose kept moving."
    );
}
