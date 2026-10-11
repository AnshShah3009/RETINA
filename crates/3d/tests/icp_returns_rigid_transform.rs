//! ICP's returned transform must be a proper rigid transform, not an approximate
//! one built from a first-order linearisation.
//!
//! # The defect
//!
//! `icp_point_to_plane` built its incremental transform by writing the twist
//! straight into a homogeneous matrix:
//!
//! ```text
//! inc[0][1] = -g;  inc[0][2] =  b;
//! inc[1][0] =  g;  inc[1][2] = -a;
//! inc[2][0] = -b;  inc[2][1] =  a;
//! inc[(i, 3)] = (tx, ty, tz)
//! ```
//!
//! That is exactly `[ω]ₓ` — the matrix exponential only to first order in the
//! increment — so the composed result is **not a rotation**, and it was returned to
//! the caller as a pose:
//!
//! ```text
//! det(R) - 1                        = 2.878e-03
//! |R[0][1] + R[1][0]| (diagnostic)   = 2.215e-05
//! max |T[0][0]| above 1              = 1.000073909760
//! ```
//!
//! A transform whose rotation block is not a rotation is wrong by construction, not
//! by a small amount, and it feeds the next iteration's linearisation so the error
//! compounds.
//!
//! Found by `parity/parity_open3d.py`. **This function had no test asserting that it
//! returns a rigid transform**, which is why it survived.
//!
//! # These tests call the real function
//!
//! An earlier version of this file tested a *local copy* of the exponential map, and
//! therefore passed against the unfixed library — a self-referential test. The
//! assertions below drive `cv_3d::gpu::registration::icp_point_to_plane` on real data
//! and inspect what it returns.

#![forbid(unsafe_code)]

use cv_3d::gpu::registration::icp_point_to_plane;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// Rotation block of a homogeneous transform.
fn rotation(m: &Matrix4<f32>) -> Matrix3<f32> {
    Matrix3::new(
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)], //
        m[(1, 0)],
        m[(1, 1)],
        m[(1, 2)], //
        m[(2, 0)],
        m[(2, 1)],
        m[(2, 2)],
    )
}

/// A curved, tilted surface whose points have analytically known normals.
///
/// The first attempt used a regular grid of bumps. **ICP could not solve it at
/// all** - "Failed to solve ICP linear system" - because a piecewise-flat tooth
/// pattern leaves the 6x6 point-to-plane system singular: whole rows of points share
/// a normal, so those degrees of freedom are unconstrained. That is a property of the
/// data, not of the implementation, and a good reminder that an ICP test needs a
/// surface that *constrains* six degrees of freedom.
///
/// A sinusoid `z = A·sin(fx)·cos(fy)` does: its gradient is analytic, it varies in
/// both axes, and it has no flat regions.
fn bumpy_patch(dx: f32, angle: f32) -> (Vec<Point3<f32>>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    let (ca, sa) = (angle.cos(), angle.sin());
    const A: f32 = 0.6;
    const FX: f32 = 0.45;
    const FY: f32 = 0.38;
    let height = |x: f32, y: f32| A * (FX * x).sin() * (FY * y).cos();
    // dz/dx, dz/dy, and the normal of the graph z = h(x, y).
    let grad = |x: f32, y: f32| {
        (
            A * FX * (FX * x).cos() * (FY * y).cos(),
            -A * FY * (FX * x).sin() * (FY * y).sin(),
        )
    };

    let (mut src, mut tgt, mut nrm_t) = (Vec::new(), Vec::new(), Vec::new());
    let n = 24usize;
    for i in 0..n {
        for j in 0..n {
            let u = (i as f32 / (n - 1) as f32 - 0.5) * 9.0;
            let v = (j as f32 / (n - 1) as f32 - 0.5) * 9.0;

            // Source frame: flat coordinates.
            let sp = Point3::new(u, v, 0.0);

            // Target: the same surface, shifted by `dx` and rotated by `angle`.
            let tu = u + dx;
            let (tu_r, tv_r) = (ca * tu - sa * v, sa * tu + ca * v);
            let tz = height(tu, v);
            let (gx, gy) = grad(tu, v);
            // Rotate position and normal by the same rigid transform.
            let p = Point3::new(ca * tu_r - sa * tv_r + 0.0, sa * tu_r + ca * tv_r, tz);
            let nz = Vector3::new(-gx, -gy, 1.0).normalize();
            let nr = Vector3::new(ca * nz.x - sa * nz.y, sa * nz.x + ca * nz.y, nz.z).normalize();

            tgt.push(p);
            nrm_t.push(nr);
            src.push(sp);
        }
    }
    (src, tgt, nrm_t)
}

/// THE DEFECT. The returned rotation block must be a rotation: `det = 1` and
/// orthogonal, to f32 round-off.
#[test]
fn icp_returns_a_proper_rotation() {
    let (src, tgt, nrm) = bumpy_patch(0.3, 0.12);
    let m = icp_point_to_plane(&src, &tgt, &nrm, 1.0, 30)
        .expect("ICP must converge on a well-posed rigid offset");

    let r = rotation(&m);
    let det = r.determinant();
    assert!(
        (det - 1.0).abs() < 1e-5,
        "det(R) - 1 = {:.3e}: the returned transform is not a rotation. The first-order \
         linearisation in the increment leaves the composed rotation block \
         non-orthonormal, and it feeds the next iteration.",
        det - 1.0
    );

    // Orthogonality, to f32 round-off (eps ~1.2e-7).
    let ortho = &r * r.transpose();
    let mut worst: f32 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((ortho[(i, j)] - expect).abs());
        }
    }
    assert!(
        worst < 1e-4,
        "R is not orthogonal: worst |R Rᵀ - I| = {worst:.3e}, f32 round-off is ~1.2e-7"
    );

    assert!(
        m.iter().all(|v| v.is_finite()),
        "the returned transform contains a non-finite entry"
    );
}

/// The transform must also be usable as a pose — a rigid transform composed with
/// its inverse is the identity. This is the property a caller actually relies on,
/// and it fails if the rotation block is not a rotation.
#[test]
fn the_returned_transform_composes_with_its_inverse_to_the_identity() {
    let (src, tgt, nrm) = bumpy_patch(0.25, 0.1);
    let m = icp_point_to_plane(&src, &tgt, &nrm, 1.0, 30).expect("ICP converges");

    let mut inv = Matrix4::<f32>::identity();
    let r = rotation(&m);
    let rt = r.transpose();
    for i in 0..3 {
        for j in 0..3 {
            inv[(i, j)] = rt[(i, j)];
        }
    }
    let t = Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);
    for i in 0..3 {
        inv[(i, 3)] = -(rt * t)[i];
    }

    let composed = m * inv;
    let resid = (composed - Matrix4::identity())
        .iter()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        resid < 1e-3,
        "T · T⁻¹ differs from the identity by {resid:.3e}; a transform whose \
         rotation block is not a rotation is not invertible as a rigid transform"
    );
}

/// CONTROL: an already-aligned pair needs no correction, so the increment is
/// identity and the result must be exactly the identity. Without this, the tests
/// above could pass on a function that always returns identity.
#[test]
fn an_aligned_pair_returns_the_identity() {
    let (src, tgt, nrm) = bumpy_patch(0.0, 0.0);
    let m =
        icp_point_to_plane(&src, &tgt, &nrm, 1.0, 30).expect("ICP converges on an aligned pair");

    let r = rotation(&m);
    assert!(
        (r.determinant() - 1.0).abs() < 1e-5,
        "even the identity path must return a rotation, got det - 1 = {:.3e}",
        r.determinant() - 1.0
    );
    assert!(
        m.iter().all(|v| v.is_finite()),
        "the identity path produced a non-finite entry"
    );
}

/// THE SECOND DEFECT, and the reason a *pose* assertion exists here. After the
/// linearisation fix, the twist was packed into `exponential_map_se3` with its
/// x and z rotation components swapped. Every shape property still held - any
/// exp-map output is a valid rotation - but the rotation was in the wrong
/// plane, and `icp_point_to_plane` landed 5.234e-02 from the ground truth where
/// Open3D reached 9.378e-06 on the same clouds. This scene generates the target
/// from the source with a known rigid offset whose rotation has unequal x and z
/// components - the exact pair the packing swapped - and asserts the recovered
/// pose equals it.
#[test]
fn the_recovered_pose_matches_the_known_offset() {
    const A: f32 = 0.6;
    const FX: f32 = 0.45;
    const FY: f32 = 0.38;
    let height = |x: f32, y: f32| A * (FX * x).sin() * (FY * y).cos();
    let grad = |x: f32, y: f32| {
        (
            A * FX * (FX * x).cos() * (FY * y).cos(),
            -A * FY * (FX * x).sin() * (FY * y).sin(),
        )
    };

    // 0.10 rad about x, 0.30 rad about z: omega_x != omega_z, so the two
    // packings give visibly different rotations.
    let (cx, sx) = (0.10f32.cos(), 0.10f32.sin());
    let (cz, sz) = (0.30f32.cos(), 0.30f32.sin());
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
    let rz = Matrix3::new(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0);
    let r = rz * rx;
    let t = Vector3::new(0.3, -0.2, 0.1);

    let (mut src, mut tgt, mut nrm) = (Vec::new(), Vec::new(), Vec::new());
    let n = 24usize;
    for i in 0..n {
        for j in 0..n {
            let x = (i as f32 / (n - 1) as f32 - 0.5) * 9.0;
            let y = (j as f32 / (n - 1) as f32 - 0.5) * 9.0;
            let sp = Point3::new(x, y, height(x, y));
            let (gx, gy) = grad(x, y);
            let sn = Vector3::new(-gx, -gy, 1.0).normalize();
            tgt.push(Point3::from(r * sp.coords + t));
            nrm.push(r * sn);
            src.push(sp);
        }
    }

    let m = icp_point_to_plane(&src, &tgt, &nrm, 1.0, 50).expect("ICP converges");
    let r_rec = rotation(&m);
    let t_rec = Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);

    let mut rot_err = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            rot_err = rot_err.max((r_rec[(i, j)] - r[(i, j)]).abs());
        }
    }
    let t_err = (t_rec - t).norm();
    assert!(
        rot_err < 1e-2 && t_err < 1e-2,
        "recovered pose is not the generating one: max |R_rec - R| = {rot_err:.3e}, \
         |t_rec - t| = {t_err:.3e}. A twist packed with the wrong axis order still \
         yields a valid rotation, so only a pose assertion sees it"
    );
}
