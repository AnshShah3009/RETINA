//! Projecting a Gaussian is a change of frame, and must not depend on which
//! frame the world is written in.
//!
//! `Gaussian::project` built its screen-space covariance as
//!
//! ```text
//! screen_cov = J_v · Σ_world · J_vᵀ
//! ```
//!
//! where `J_v` is the Jacobian of the pixel projection with respect to **view**
//! coordinates and `Σ_world` is the covariance in **world** coordinates. The
//! camera's rotation never entered, so the two frames were mixed: the standard
//! form is `J_v · (R Σ_world Rᵀ) · J_vᵀ` with `R` the world-to-camera rotation.
//!
//! The mistake is invisible for a camera at the identity orientation - which is
//! what every existing test uses - and it is invisible at the pixel *centre*,
//! which does use the full view matrix. It shows up as a wrong ellipse
//! orientation and wrong extent for any rotated camera, and it also feeds
//! `compute_tile_bounds`, so a large splat can be clipped out of pixels it
//! actually covers.
//!
//! The invariant: rotating the whole world (scene *and* camera) about any axis
//! through the camera origin is a change of coordinates, not a change of image.
//! A camera looking at a scene must render the same picture either way.

use cv_rendering::gaussian_splatting::{
    Camera, Gaussian, GaussianCloud, GaussianRasterizer, SphericalHarmonics,
};
use nalgebra::{Matrix3, Matrix3x4, Point3, Rotation3, UnitQuaternion, Vector3, Vector4};

fn rot(axis: Vector3<f32>, angle: f32) -> UnitQuaternion<f32> {
    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
}

fn quat_vec(q: &UnitQuaternion<f32>) -> Vector4<f32> {
    Vector4::new(q.i, q.j, q.k, q.w)
}

/// `Camera::new` takes the world-to-camera rotation and the camera position, and
/// builds `[R | -R·p]`. Spell that out so the test world change is applied to the
/// same quantities the camera is made of.
fn camera(r_world_to_camera: &UnitQuaternion<f32>, position: Vector3<f32>) -> Camera {
    Camera::new(
        Point3::from(position),
        quat_vec(r_world_to_camera),
        500.0,
        160,
        120,
    )
}

fn expected_view_matrix(
    r_world_to_camera: &UnitQuaternion<f32>,
    p: Vector3<f32>,
) -> Matrix3x4<f32> {
    let r: Matrix3<f32> = r_world_to_camera.to_rotation_matrix().into_inner();
    let mut m = Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = r[(i, j)];
        }
        m[(i, 3)] = -(r[(i, 0)] * p.x + r[(i, 1)] * p.y + r[(i, 2)] * p.z);
    }
    m
}

fn scene() -> Gaussian {
    let q = rot(Vector3::new(0.0, 1.0, 0.3), 0.7);
    Gaussian {
        position: Point3::new(0.5, 0.2, 3.0),
        // Deliberately anisotropic: an isotropic covariance cannot show a
        // projection that ignores the camera's orientation.
        scale: Vector3::new(0.30, 0.08, 0.20),
        rotation: quat_vec(&q),
        opacity: 0.5,
        spherical_harmonics: SphericalHarmonics::from_dc(Vector3::new(1.0, 0.0, 0.0)),
        features: Vector3::zeros(),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn rotating_the_world_does_not_change_the_projected_splat() {
    // Base world.
    let cam_rot = rot(Vector3::new(1.0, 1.0, 0.0), 0.4);
    let cam_pos = Vector3::new(0.3, -0.2, 1.5);
    let camera0 = camera(&cam_rot, cam_pos);

    // Control: the camera the test rotates was built as intended.
    let view0 = expected_view_matrix(&cam_rot, cam_pos);
    let view_err = (camera0.view_matrix - view0)
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        view_err < 1e-5,
        "camera construction is not [R | -R·p]: max entry error {view_err:e}"
    );

    let mut cloud0 = GaussianCloud::new();
    cloud0.push(scene());

    // The same world, written in a frame rotated by `a` about the camera origin.
    let a = rot(Vector3::new(0.0, 1.0, 0.0), 0.6);
    let a_mat: Matrix3<f32> = a.to_rotation_matrix().into_inner();
    let r_c0: Matrix3<f32> = cam_rot.to_rotation_matrix().into_inner();
    // A world-to-camera rotation has to pre-absorb the inverse world change:
    // R_new = R_old · Aᵀ, and the camera's centre moves with the world.
    let cam_rot1 = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        r_c0 * a_mat.transpose(),
    ));
    let camera1 = camera(&cam_rot1, a_mat * cam_pos);

    let g0 = scene();
    let g_rot: Matrix3<f32> = {
        let v = g0.rotation;
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[3], v[0], v[1], v[2]));
        q.to_rotation_matrix().into_inner()
    };
    let qg1 =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(a_mat * g_rot));
    let g1 = Gaussian {
        position: Point3::from(a_mat * g0.position.coords),
        rotation: quat_vec(&qg1),
        ..g0.clone()
    };

    let mut cloud1 = GaussianCloud::new();
    cloud1.push(g1);

    let r0 = GaussianRasterizer::new(camera0, 16, 16);
    let r1 = GaussianRasterizer::new(camera1, 16, 16);

    let p0 = r0.project_gaussians(&cloud0)[0].clone();
    let p1 = r1.project_gaussians(&cloud1)[0].clone();

    assert!(p0.is_valid() && p1.is_valid(), "both splats must project");

    let center_err = (p1.center - p0.center).norm();
    let cov_err = (p1.covariance - p0.covariance)
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));

    println!("base centre      = {:?}", p0.center);
    println!("rotated centre   = {:?}", p1.center);
    println!("centre error     = {center_err:e}");
    println!("base cov         = {:?}", p0.covariance);
    println!("rotated cov      = {:?}", p1.covariance);
    println!("covariance error = {cov_err:e}");

    // The centre is already frame-independent (it goes through the full view
    // matrix) - this is the control that the world change really is a change of
    // frame, and the failure below is not the whole test being wrong.
    assert!(
        center_err < 1e-3,
        "the projected centre must be frame-independent, but moved by {center_err:e}"
    );

    assert!(
        cov_err < 1e-4 * 1e3,
        "the screen-space covariance must be frame-independent, but changed by \
         {cov_err:e} (base {:?} vs rotated {:?})",
        p0.covariance,
        p1.covariance
    );

    // And the end-to-end statement: the same scene, seen by the same camera in a
    // rotated world, must rasterize to the same image.
    let out0 = r0.rasterize(&cloud0);
    let out1 = r1.rasterize(&cloud1);
    let alpha_err = max_abs_diff(&out0.alpha, &out1.alpha);
    let colour_err = max_abs_diff(
        &out0
            .color
            .iter()
            .flat_map(|c| [c.x, c.y, c.z])
            .collect::<Vec<_>>(),
        &out1
            .color
            .iter()
            .flat_map(|c| [c.x, c.y, c.z])
            .collect::<Vec<_>>(),
    );
    let differing = out0
        .alpha
        .iter()
        .zip(out1.alpha.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-4)
        .count();
    println!("alpha error = {alpha_err:e}, colour error = {colour_err:e}, differing pixels = {differing}/{}", out0.alpha.len());
    assert!(
        alpha_err < 1e-3 && colour_err < 1e-3,
        "a world-frame change must not change the rendered image: alpha {alpha_err:e}, \
         colour {colour_err:e}, {differing} pixels differ"
    );
}
