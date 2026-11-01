//! `Camera::from(CameraIntrinsicsF32)` must store a projection matrix whose
//! **principal point** is the intrinsics' principal point.
//!
//! The two focal lengths are already pinned (`camera_from_projection.rs`,
//! `anisotropic_intrinsics.rs`), and this file controls for them explicitly so a
//! failure here cannot be mistaken for that defect. What was left is the other
//! half of the pinhole:
//!
//! ```text
//! u = fx * X/Z + cx        v = fy * Y/Z + cy
//! ```
//!
//! `Camera::from` stores `cx` and `cy` and `Gaussian::project` - the code that
//! actually rasterises - reads them. The stored `projection_matrix`, however,
//! comes from `Camera::new`, which builds the *symmetric* perspective matrix
//! with `P[0][2] = P[1][2] = 0`. That is the correct matrix for the centred
//! principal point `Camera::new` itself uses (`cx = width/2`), but for an
//! off-centre principal point it is a different camera: the optical axis
//! (`X = Y = 0`) satisfies `P_row · [X, Y, Z, 1] = 0`, so anything reading
//! `projection_matrix` / `view_projection` puts it at the image centre while the
//! splat renderer puts it at `(cx, cy)`.
//!
//! Measured with `fx, fy = 500, 200` and `cx, cy = 100, 400` on a 640x480
//! image, for the camera-space point `(0.3, -0.2, 2.0)`:
//!
//! ```text
//! Gaussian::project (what is rendered)  -> (175.0, 380.0)
//! view_projection   (what is stored)    -> (395.0, 220.0)
//!                                            220 px and 160 px apart
//! ```

use cv_core::CameraIntrinsicsF32;
use cv_rendering::gaussian_splatting::{Camera, Gaussian, GaussianCloud, GaussianRasterizer};
use nalgebra::{Point3, Vector3, Vector4};

/// Where `Camera::view_projection` says a camera-space point lands, in pixels.
///
/// `view_projection` is a **3x4** matrix, so clip coordinates stop at `xyz`: the
/// fourth row of a perspective transform (`(0, 0, -1, 0)` in the classic OpenGL
/// convention) is not stored, and the perspective divide is by the eye depth
/// `w_clip = Z`. Dividing by `clip[2]` instead was a mistake in this file's first
/// draft - `P[2][2]`/`P[2][3]` carry the near/far mapping, not the depth - and
/// the centred control below caught it, which is exactly what it is there for.
///
/// NDC in `[-1, 1]` then maps to pixels as `(ndc + 1) * size/2` with **no y
/// flip**, the orientation `Gaussian::project` uses (`v = fy * Y/Z + cy` with
/// `fy > 0`, no negation). Together those two steps reproduce the pinhole
/// `(fx X/Z + cx, fy Y/Z + cy)` from the stored matrix alone.
fn pixel_through_view_projection(c: &Camera, p_cam: Vector3<f32>) -> (f32, f32) {
    let vp = c.view_projection();
    let clip = vp * Vector4::new(p_cam.x, p_cam.y, p_cam.z, 1.0);
    let w = p_cam.z;
    (
        (clip[0] / w + 1.0) * c.width as f32 / 2.0,
        (clip[1] / w + 1.0) * c.height as f32 / 2.0,
    )
}

/// Where the rasteriser actually draws a splat at that point, in pixels - the
/// forward pass's own answer, not a re-derivation of the formula.
fn pixel_through_the_renderer(c: &Camera, p_cam: Vector3<f32>) -> (f32, f32) {
    let mut cloud = GaussianCloud::new();
    cloud.push(Gaussian::new(
        Point3::from(p_cam),
        Vector3::new(0.05, 0.05, 0.05),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
    ));
    let pg = GaussianRasterizer::new(c.clone(), 16, 16).project_gaussians(&cloud)[0].clone();
    assert!(pg.is_valid(), "the probe splat must project");
    (pg.center.x, pg.center.y)
}

/// The focal lengths recovered from the stored matrix, by the same relation
/// `camera_from_projection.rs` uses: `P[0][0] = fx/(w/2)`, `P[1][1] = fy/(h/2)`.
fn focal_lengths(c: &Camera) -> (f32, f32) {
    (
        (c.width as f32 / 2.0) * c.projection_matrix[(0, 0)],
        (c.height as f32 / 2.0) * c.projection_matrix[(1, 1)],
    )
}

/// CONTROL: with the centred principal point `Camera::new` uses, the stored
/// matrix and the renderer already agree. This is what makes the off-centre
/// case a defect in the *principal point* rather than in the convention this
/// test decodes the matrix with - the decoder is exercised on a camera that is
/// known to be self-consistent.
#[test]
fn a_centred_camera_stores_a_matrix_that_agrees_with_the_renderer() {
    let c = Camera::new(
        Point3::origin(),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        500.0,
        640,
        480,
    );
    assert_eq!((c.cx, c.cy), (320.0, 240.0));
    let p = Vector3::new(0.3, -0.2, 2.0);
    let (sx, sy) = pixel_through_view_projection(&c, p);
    let (rx, ry) = pixel_through_the_renderer(&c, p);
    println!("centred: stored ({sx}, {sy}) vs rendered ({rx}, {ry})");
    assert!(
        (sx - rx).abs() < 1e-2 && (sy - ry).abs() < 1e-2,
        "a centred camera must be self-consistent: stored ({sx}, {sy}) vs rendered ({rx}, {ry})"
    );
}

/// CONTROL: the focal lengths the intrinsics carry are already stored correctly
/// - the defect below is not the one `camera_from_projection.rs` pins.
#[test]
fn the_off_centre_camera_still_stores_the_right_focal_lengths() {
    let intrinsics = CameraIntrinsicsF32::new(500.0, 200.0, 100.0, 400.0, 640, 480);
    let c = Camera::from(intrinsics);
    let (fx, fy) = focal_lengths(&c);
    assert!(
        (fx - 500.0).abs() < 1e-2 && (fy - 200.0).abs() < 1e-2,
        "the focal lengths are a separate, already-fixed part of this contract: got fx={fx}, fy={fy}"
    );

    // Both agree on the *scale* of the projection: a point on the optical axis
    // and one off it differ by `fx / Z`, in both readings. Only the origin of
    // the two readings differs.
    let depth = 2.0f32;
    let (s0, _) = pixel_through_view_projection(&c, Vector3::new(0.0, 0.0, depth));
    let (s1, _) = pixel_through_view_projection(&c, Vector3::new(1.0, 0.0, depth));
    assert!(
        (s1 - s0 - fx / depth).abs() < 1e-2,
        "the stored matrix must still scale by fx/Z: 1 unit at Z={depth} moved {:.4} px, \
         fx/Z = {}",
        s1 - s0,
        fx / depth
    );
}

/// DEFECT: the stored projection matrix must place the optical axis at the
/// intrinsics' principal point, because `Gaussian::project` does.
#[test]
fn an_off_centre_principal_point_is_in_the_stored_projection() {
    let intrinsics = CameraIntrinsicsF32::new(500.0, 200.0, 100.0, 400.0, 640, 480);
    let c = Camera::from(intrinsics);
    assert_eq!((c.cx, c.cy), (100.0, 400.0), "the camera must keep cx, cy");

    // The optical axis projects to the principal point, in both readings.
    let (axis_x, axis_y) = pixel_through_view_projection(&c, Vector3::new(0.0, 0.0, 2.0));
    println!(
        "optical axis: stored ({axis_x}, {axis_y}), camera says ({}, {})",
        c.cx, c.cy
    );
    assert!(
        (axis_x - c.cx).abs() < 1e-2 && (axis_y - c.cy).abs() < 1e-2,
        "the optical axis must land on the principal point: the stored matrix put it at \
         ({axis_x}, {axis_y}) but the camera says (cx, cy) = ({}, {}). `Camera::from` keeps the \
         intrinsics' principal point while the projection matrix it inherits is the *symmetric* \
         one from `Camera::new`, whose P[0][2] = P[1][2] = 0 encodes cx = width/2, \
         cy = height/2 = ({}, {})",
        c.cx,
        c.cy,
        c.width as f32 / 2.0,
        c.height as f32 / 2.0
    );

    // And the general point: the two readings of the same camera must agree on
    // where it looks, not only on the axis.
    let p = Vector3::new(0.3, -0.2, 2.0);
    let (sx, sy) = pixel_through_view_projection(&c, p);
    let (rx, ry) = pixel_through_the_renderer(&c, p);
    println!("off-centre: stored ({sx}, {sy}) vs rendered ({rx}, {ry})");
    assert!(
        (sx - rx).abs() < 1e-2 && (sy - ry).abs() < 1e-2,
        "the stored matrix and the splat renderer disagree by ({} px, {} px) on the same \
         camera-frame point: stored ({sx}, {sy}) vs rendered ({rx}, {ry})",
        sx - rx,
        sy - ry
    );
}
