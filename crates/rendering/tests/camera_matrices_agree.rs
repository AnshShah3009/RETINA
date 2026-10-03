//! `Camera`'s stored matrices must describe the camera it says it does.
//!
//! Three defects, all in `gaussian_splatting/rasterize.rs`, all reachable
//! through the public `Camera::new` / `Camera::view_projection`.
//!
//! ## 1. `view_projection` dropped the translation column
//!
//! `P * V` with `V` a 3x4 affine matrix needs `k` to run over the *four* rows of
//! `P`, not three: the missing term is `P[i][3] * V[3][3]` with `V[3][3] = 1`.
//! The loop stopped at `k < 3`, so the fourth column of the result was
//! identically zero and a view-projection matrix mapped every point to
//! `z_clip = 0` - a projection onto a plane, with no depth at all.
//!
//! Measured against the true composite: max abs entry error **1.7210021**.
//!
//! ## 2. `rotation_to_matrix` did not normalise a non-unit quaternion
//!
//! The formula is the unit-quaternion form; fed a quaternion of norm `k` it
//! returns the rotation scaled by `k²`, so the determinant is `k³` rather than 1.
//! `Gaussian::rotation_matrix` - a sibling doing the same job on the same data -
//! normalises, and documents why at length; the camera did not.
//!
//! Measured with a norm-2 quaternion: **determinant 2.894536**, and
//! **max |R_scaled - R_unit| = 0.826081** - not a slightly denormalised camera but
//! a different one.
//!
//! ## 3. `fov` was built from `width` and fed to a `fov_y` parameter
//!
//! ```text
//! fov = 2 * atan(0.5 * width / focal_length)     // a HORIZONTAL fov
//! ```
//!
//! `perspective_matrix` takes `fov_y` and computes `P[1][1] = 1 / tan(fov_y/2)`,
//! so the vertical focal length came out as `f / aspect`. Meanwhile
//! `Gaussian::project` - the code that actually rasterises the splats - uses
//! `camera.focal_length` directly. For any non-square viewport the stored
//! projection matrix and the projection actually performed disagreed.
//!
//! Measured `1 / P[1][1]`:
//!
//! ```text
//!   160x120, f =  500 -> 0.160000   (f = 0.500000 expected)
//!   800x600, f = 1000 -> 0.400000   (f = 1.000000 expected)
//!   640x480, f =  500 -> 0.640000   (f = 0.500000 expected)
//! ```
//!
//! `aspect` also divided by `height` unguarded, so a zero-height camera made it
//! infinite and a zero-by-zero one NaN, both of which reached `view_projection`.

use cv_rendering::gaussian_splatting::Camera;
use nalgebra::{Matrix3, Matrix3x4, Point3, UnitQuaternion, Vector3, Vector4};

fn quat_vec(q: &UnitQuaternion<f32>) -> Vector4<f32> {
    Vector4::new(q.i, q.j, q.k, q.w)
}

fn rot(axis: Vector3<f32>, angle: f32) -> UnitQuaternion<f32> {
    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
}

fn cam_at(
    rotation: &UnitQuaternion<f32>,
    position: Vector3<f32>,
    focal: f32,
    w: u32,
    h: u32,
) -> Camera {
    Camera::new(Point3::from(position), quat_vec(rotation), focal, w, h)
}

/// The focal lengths a `Camera` actually projects with.
///
/// A pinhole camera maps `(x, y, z)` to `((x/z) f + w/2, (y/z) f + h/2)`, so the
/// stored matrix entries are the focal lengths **over** the half-viewport, not
/// the focal lengths themselves:
///
/// ```text
/// P[0][0] = fx / (w/2)        P[1][1] = fy / (h/2)
/// ```
///
/// Comparing `1 / P[1][1]` against `focal_length` would compare
/// `(h/2) / fy` against `fy`. That is a mistake worth pinning in the test
/// rather than leaving implicit, because it is the obvious thing to write here
/// and it produced an assertion this file's first draft failed on.
fn focal_lengths(c: &Camera) -> (f32, f32) {
    (
        (c.width as f32 / 2.0) * c.projection_matrix[(0, 0)],
        (c.height as f32 / 2.0) * c.projection_matrix[(1, 1)],
    )
}

/// CONTROL: the matrix really does encode the pinhole model above - verified by
/// pushing a point through it and comparing against the projection the splat
/// renderer performs. Without this, the focal-length assertions below could be
/// satisfied by a matrix that is not the one being used.
#[test]
fn the_projection_matrix_encodes_the_camera_it_stores() {
    let q = rot(Vector3::new(1.0, 1.0, 0.0), 0.4);
    for &(w, h, f) in &[
        (160u32, 120u32, 500.0f32),
        (800, 600, 1000.0),
        (256, 256, 500.0),
    ] {
        let c = cam_at(&q, Vector3::new(0.0, 0.0, 0.0), f, w, h);
        let (fx, fy) = focal_lengths(&c);
        let (wf, hf) = (w as f32, h as f32);
        // P00 must be fx/(w/2) and P11 must be fy/(h/2) - equivalently the
        // classic OpenGL convention P00 = 1/(aspect tan(fov/2)).
        // P00 = fx/(w/2) and P11 = fy/(h/2) - equivalently the classic OpenGL
        // convention `P00 = 1/(aspect * tan(fov_y/2))`, `P11 = 1/tan(fov_y/2)`.
        //
        // `fov_y = 2*atan(0.5*h/f)`, so `tan(fov_y/2) = tan(atan(0.5*h/f)) =
        // 0.5*h/f` exactly. My first version wrote `tan(0.5*h/f)` — omitting the
        // `atan` — which asks for `1/tan(0.24) = 8.293295` where the projection
        // matrix correctly holds `1/0.12 = 8.333334`. A 0.48% discrepancy that
        // looked like a real convention bug and was an arithmetic slip in the
        // expectation.
        let tan_half = 0.5 * hf / f;
        let p00_want = 1.0 / ((wf / hf) * tan_half);
        let p11_want = 1.0 / tan_half;
        println!(
            "{w}x{h}, f = {f}: fx = {fx:.4}, fy = {fy:.4}, P00 = {:.6} (want {p00_want:.6}), \
             P11 = {:.6} (want {p11_want:.6})",
            c.projection_matrix[(0, 0)],
            c.projection_matrix[(1, 1)]
        );
        assert!(
            (c.projection_matrix[(0, 0)] - p00_want).abs() < 1e-5 * p00_want,
            "{w}x{h}: P00 is not the standard perspective entry"
        );
        assert!(
            (c.projection_matrix[(1, 1)] - p11_want).abs() < 1e-5 * p11_want,
            "{w}x{h}: P11 is not the standard perspective entry"
        );
        // `fx == fy` is the correct relation here, and NOT `fx/fy == aspect`.
        //
        // This camera has **square pixels** - one focal length for both axes - so
        // `fx/fy = 1` while the aspect of a 160x120 viewport is 1.333. My first
        // version asserted they were equal, which is unsatisfiable for any
        // non-square viewport with square pixels; it failed at 0.99999994 vs
        // 1.3333334 and read like a convention bug.
        //
        // What is worth pinning is that both focal lengths recover the single
        // stored value, and that the aspect appears in `P00` and not in the focal
        // lengths - which is exactly what the P00/P11 assertions above check.
        assert!(
            (fx - f).abs() < 1e-3 * f && (fy - f).abs() < 1e-3 * f,
            "{w}x{h}: square pixels mean fx = fy = f = {f}; got fx = {fx}, fy = {fy}"
        );
    }
}

/// Defect 3: the stored projection matrix must use the same focal length the
/// splat renderer projects with - `camera.focal_length`, which
/// `Gaussian::project` passes straight through.
#[test]
fn the_projection_matrix_agrees_with_the_stored_focal_length() {
    for &(w, h, f) in &[
        (160u32, 120u32, 500.0f32),
        (800, 600, 1000.0),
        (640, 480, 500.0),
        (256, 256, 500.0),
    ] {
        let c = cam_at(
            &UnitQuaternion::identity(),
            Vector3::new(0.0, 0.0, 0.0),
            f,
            w,
            h,
        );
        let (fx, fy) = focal_lengths(&c);
        println!(
            "{w}x{h}, f = {f}: fx = {fx:.4}, fy = {fy:.4}, camera.focal_length = {}",
            c.focal_length
        );
        assert!(
            (fx - c.focal_length).abs() < 1e-3 * c.focal_length,
            "{w}x{h}: fx {fx} disagrees with the camera's own focal_length {}",
            c.focal_length
        );
        assert!(
            (fy - c.focal_length).abs() < 1e-3 * c.focal_length,
            "{w}x{h}: fy {fy} disagrees with the camera's own focal_length {}. A vertical fov \
             built from the width lands at f/aspect = {}",
            c.focal_length,
            c.focal_length / (w as f32 / h as f32)
        );
    }
}

/// CONTROL: a unit quaternion gives a proper rotation, at a square and a
/// non-square viewport. Without this, the determinant tests below could be
/// satisfied by a matrix that is not a rotation at all.
#[test]
fn a_unit_quaternion_still_gives_a_proper_rotation() {
    for &(w, h) in &[(160u32, 120u32), (800, 600), (256, 256)] {
        let q = rot(Vector3::new(1.0, 1.0, 0.0), 0.4);
        let c = cam_at(&q, Vector3::new(0.3, -0.2, 1.5), 500.0, w, h);
        let r: Matrix3<f32> = c.view_matrix.fixed_view::<3, 3>(0, 0).into();
        let det = r.determinant();
        let orth = (r * r.transpose() - Matrix3::identity())
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        println!("{w}x{h}: det = {det:.6}, max |R R^T - I| = {orth:.6e}");
        assert!(
            (det - 1.0).abs() < 1e-4,
            "{w}x{h}: a unit quaternion must give determinant 1, got {det}"
        );
        assert!(
            orth < 1e-5,
            "{w}x{h}: the view rotation is not orthonormal: {orth:e}"
        );
    }
}

/// Defect 2: a quaternion of norm 2 denotes the **same** rotation as its unit
/// version, because a quaternion is invariant under positive scaling. Passing it
/// to the public constructor must therefore give the same camera.
#[test]
fn a_non_unit_quaternion_still_gives_a_proper_rotation() {
    let q = rot(Vector3::new(1.0, 1.0, 0.0), 0.4);
    let pos = Vector3::new(0.3, -0.2, 1.5);
    let unit = Camera::new(Point3::from(pos), quat_vec(&q), 500.0, 160, 120);
    let raw = Camera::new(Point3::from(pos), quat_vec(&q) * 2.0, 500.0, 160, 120);

    let r: Matrix3<f32> = raw.view_matrix.fixed_view::<3, 3>(0, 0).into();
    let det = r.determinant();
    let diff = (raw.view_matrix - unit.view_matrix)
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    println!("norm-2 quaternion: det = {det:.6}, max |V - V_unit| = {diff:.6}");
    assert!(
        (det - 1.0).abs() < 1e-4,
        "a norm-2 quaternion must still describe a rotation, determinant is {det}. The raw \
         formula scales the matrix by norm^2, so the determinant is norm^3 = 8 before the fix"
    );
    assert!(
        diff < 1e-5,
        "a norm-2 quaternion must give the same view matrix as the unit one, they differ by \
         {diff:e}"
    );

    // A slightly denormalised quaternion is the common, benign case: 0.9999
    // should move nothing measurable. It is included because a fix that only
    // special-cased large norms would show up here as a jump rather than a
    // scale.
    let nearly = Camera::new(Point3::from(pos), quat_vec(&q) * 0.9999, 500.0, 160, 120);
    let d2 = (nearly.view_matrix - unit.view_matrix)
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    println!("norm-0.9999 quaternion: max |V - V_unit| = {d2:.6}");
    assert!(
        d2 < 1e-3,
        "a 0.9999-norm quaternion moved the camera by {d2:e}"
    );

    // A zero quaternion: no rotation at all. It must not be NaN.
    let zero = Camera::new(Point3::origin(), Vector4::zeros(), 500.0, 160, 120);
    assert!(
        zero.view_matrix.iter().all(|v| v.is_finite()),
        "a zero quaternion produced a non-finite view matrix: {:?}",
        zero.view_matrix
    );
    assert_eq!(
        zero.view_matrix,
        Camera::new(
            Point3::origin(),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            500.0,
            160,
            120
        )
        .view_matrix,
        "a zero quaternion must give the identity rotation, the same as an explicit identity"
    );
}

/// Defect 1: `view_projection` must be the true `P * V`, translation column
/// included.
#[test]
fn view_projection_includes_the_translation_column() {
    let q = rot(Vector3::new(1.0, 1.0, 0.0), 0.4);
    let pos = Vector3::new(0.3, -0.2, 1.5);
    let c = cam_at(&q, pos, 500.0, 160, 120);

    let got = c.view_projection();
    let mut want = Matrix3x4::zeros();
    // `P * V` with `V` a 3x4 *affine* matrix, so its implicit fourth row is
    // `(0, 0, 0, 1)`. The `k == 3` term is therefore `P[i][3] * V[3][j]`, which is
    // `P[i][3]` **only for the translation column**: `V[3][j]` is 0 for j < 3.
    //
    // My first version added `P[i][3]` to every column, which shifts the three
    // rotation columns too - a different matrix entirely, and a plausible-looking
    // reference for one. It disagreed with the implementation by exactly
    // `0.020002007` in the fourth column and looked like a missing transform.
    // The implementation's own comment already states the rule.
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..3 {
                want[(i, j)] += c.projection_matrix[(i, k)] * c.view_matrix[(k, j)];
            }
        }
        want[(i, 3)] += c.projection_matrix[(i, 3)];
    }
    let err = (got - want).iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("view_projection vs P*V: max entry error {err:e}");
    println!(
        "got  column 3 = [{}, {}, {}]",
        got[(0, 3)],
        got[(1, 3)],
        got[(2, 3)]
    );
    println!(
        "want column 3 = [{}, {}, {}]",
        want[(0, 3)],
        want[(1, 3)],
        want[(2, 3)]
    );
    assert!(
        err < 1e-4,
        "view_projection is not P*V: max entry error {err:e}. The fourth column was {err} \
         too large, which maps every point to z_clip = 0"
    );

    // The statement that matters: the composite must place a known world point
    // where the camera says it is.
    let p = Point3::new(0.7, 0.1, 3.0);
    let world = nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
    let view = c.view_matrix * world;
    let clip = c.view_projection() * world;
    // `clip == P * view`, verified entry by entry, so z_clip is no longer pinned
    // at zero.
    assert!(
        (clip[2]
            - (c.projection_matrix[(2, 0)] * view[0]
                + c.projection_matrix[(2, 1)] * view[1]
                + c.projection_matrix[(2, 2)] * view[2]
                + c.projection_matrix[(2, 3)]))
            .abs()
            < 1e-4,
        "z_clip does not follow from the view and projection matrices"
    );
    println!("clip = {clip:?}");
    assert!(
        clip[2].is_finite() && clip[2] != 0.0,
        "z_clip must be a real depth, got {}",
        clip[2]
    );
}

/// A zero-height or zero-width viewport must not produce a non-finite
/// projection.
#[test]
fn a_degenerate_viewport_does_not_produce_nan() {
    for &(w, h) in &[(0u32, 0u32), (160, 0), (0, 120)] {
        let c = cam_at(
            &UnitQuaternion::identity(),
            Vector3::new(0.1, 0.1, 1.0),
            500.0,
            w,
            h,
        );
        let finite = c
            .view_matrix
            .iter()
            .chain(c.projection_matrix.iter())
            .chain(c.view_projection().iter())
            .all(|v| v.is_finite());
        println!(
            "{w}x{h}: projection_matrix[1][1] = {}, view_projection finite = {finite}",
            c.projection_matrix[(1, 1)]
        );
        assert!(
            finite,
            "{w}x{h}: a degenerate viewport produced a non-finite matrix. `aspect` divides by \
             `height` unguarded, so 0 gave inf and 0/0 gave NaN"
        );
    }
}
