//! `Camera::from(CameraIntrinsicsF32)` must store a projection matrix that
//! encodes the same focal lengths the camera itself stores.
//!
//! `Camera::from` keeps `fx` in `focal_length` and puts the vertical focal in
//! its own `fy` field, which is what `Gaussian::project` actually rasterises
//! with. The stored `projection_matrix`, however, is built by `Camera::new`
//! from a single focal length, so both of its diagonal entries end up encoding
//! `fx`:
//!
//! ```text
//! P[0][0] = 2 fx / width      P[1][1] = 2 fy / height
//! ```
//!
//! For `fx`, `fy` = `500`, `200` on a 640x480 image the vertical entry comes
//! out as `2 * 500 / 480`, i.e. an effective `fy` of 500 instead of 200. The
//! camera renders with 200 (`Gaussian::project` passes `camera.fy` straight
//! through) while anything that consumes `projection_matrix` / `view_projection`
//! - a depth test, a mesh draw, a reprojection - verticalises with 500.

use cv_core::CameraIntrinsicsF32;
use cv_rendering::gaussian_splatting::Camera;

/// Focal lengths recovered from the stored perspective matrix: a pinhole stores
/// `P[0][0] = fx / (width/2)` and `P[1][1] = fy / (height/2)`.
fn focal_lengths(c: &Camera) -> (f32, f32) {
    (
        (c.width as f32 / 2.0) * c.projection_matrix[(0, 0)],
        (c.height as f32 / 2.0) * c.projection_matrix[(1, 1)],
    )
}

#[test]
fn projection_matrix_keeps_the_anisotropic_focal_lengths() {
    let intrinsics = CameraIntrinsicsF32::new(500.0, 200.0, 100.0, 400.0, 640, 480);
    let camera = Camera::from(intrinsics);
    assert_eq!(camera.focal_length, 500.0);
    assert_eq!(camera.fy, 200.0);

    let (fx, fy) = focal_lengths(&camera);
    println!(
        "projection matrix encodes fx = {fx}, fy = {fy}; camera says fx = {}, fy = {}",
        camera.focal_length, camera.fy
    );

    // Control: the horizontal entry is already correct, so a fix must not be
    // "swap the entries" and must leave `P[0][0]` alone.
    assert!(
        (fx - camera.focal_length).abs() < 1e-2,
        "P[0][0] must encode fx = {}, got {fx}",
        camera.focal_length
    );
    // The defect: the vertical entry uses the same focal length as the
    // horizontal one, not the camera's `fy`.
    assert!(
        (fy - camera.fy).abs() < 1e-2,
        "P[1][1] must encode camera.fy = {}, got {fy}. The stored matrix and the \
         projection the splat renderer performs disagree by fx/fy = 2.5",
        camera.fy
    );
}
