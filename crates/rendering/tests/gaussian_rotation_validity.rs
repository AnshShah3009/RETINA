//! A `Gaussian`'s rotation must be a rotation, whatever the quaternion holds.
//!
//! `Gaussian::rotation` is a public `Vector4` and a deserialiser, a PLY reader, or
//! a direct field write can put a non-unit quaternion in it. `rotation_matrix`
//! applied the textbook formula to the quaternion as given, which for a
//! quaternion of norm `k` returns the rotation **scaled by `k²`**: determinant
//! `k⁶` instead of +1. The covariance is then `k⁴` too large - a well-formed
//! matrix, so nothing downstream complains, it just renders the splat the wrong
//! size. `Gaussian::new` also fed a zero quaternion straight into `normalize()`
//! (`0/0`), so a constructor call could return NaN.
//!
//! Control: a unit quaternion - what `Gaussian::new`, the PLY reader and the
//! optimizer all produce - is untouched by either guard.

use cv_rendering::gaussian_splatting::{Gaussian, SphericalHarmonics};
use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3, Vector4};

fn gaussian_with_rotation(rotation: Vector4<f32>) -> Gaussian {
    Gaussian {
        position: Point3::new(0.0, 0.0, 3.0),
        scale: Vector3::new(0.2, 0.1, 0.05),
        rotation,
        opacity: 0.5,
        spherical_harmonics: SphericalHarmonics::from_dc(Vector3::new(1.0, 0.0, 0.0)),
        features: Vector3::zeros(),
    }
}

fn all_finite(v: impl IntoIterator<Item = f32>) -> bool {
    v.into_iter().all(|x| x.is_finite())
}

/// Control: a unit quaternion gives determinant +1 and the covariance is
/// `R S S Rᵀ`, exactly as documented.
#[test]
fn a_unit_quaternion_gives_a_proper_rotation() {
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.9);
    let g = gaussian_with_rotation(Vector4::new(q.i, q.j, q.k, q.w));

    let r = g.rotation_matrix();
    assert!(
        (r.determinant() - 1.0).abs() < 1e-5,
        "determinant {} for a unit quaternion",
        r.determinant()
    );
    let expected = r * Matrix3_from_diag(&g.scale) * Matrix3_from_diag(&g.scale) * r.transpose();
    let err = (g.covariance() - expected)
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(err < 1e-6, "covariance does not match R S Sᵀ Rᵀ: {err:e}");
}

fn Matrix3_from_diag(v: &Vector3<f32>) -> nalgebra::Matrix3<f32> {
    nalgebra::Matrix3::from_diagonal(v)
}

/// A quaternion of norm 2 is a rotation of twice the angle only if it is
/// normalized first. The matrix built from it raw has determinant `2⁶ = 64` and
/// its covariance is `2⁴ = 16` times too large.
#[test]
fn a_non_unit_quaternion_still_gives_a_proper_rotation() {
    let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.9);
    let scaled = Vector4::new(q.i, q.j, q.k, q.w) * 2.0;

    let g = gaussian_with_rotation(scaled);
    let det = g.rotation_matrix().determinant();
    println!("determinant of rotation_matrix() for a norm-2 quaternion: {det}");
    assert!(
        (det - 1.0).abs() < 1e-4,
        "a stored quaternion of norm 2 must still describe a rotation, but its \
         matrix has determinant {det} (the raw formula scales it by norm^2, and \
         the covariance by norm^4)"
    );

    // And it must be the same rotation as the normalized quaternion - the same
    // ellipse, not one scaled by 16.
    let reference = gaussian_with_rotation(Vector4::new(q.i, q.j, q.k, q.w));
    let err = (g.covariance() - reference.covariance())
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "covariance difference vs the normalized quaternion: {err:e} (mine {:?} vs {:?})",
        g.covariance()[(0, 0)],
        reference.covariance()[(0, 0)]
    );
    assert!(err < 1e-6, "the covariance is norm^4 too large: {err:e}");
}

/// `Gaussian::new(.., Vector4::zeros(), ..)` ran `0.0/0.0`: the rotation, and
/// every matrix derived from it, came back NaN. The PLY reader already guards
/// this case; the constructor did not.
#[test]
fn a_zero_quaternion_does_not_produce_nan() {
    let g = Gaussian::new(
        Point3::new(0.0, 0.0, 3.0),
        Vector3::new(0.2, 0.1, 0.05),
        Vector4::zeros(),
        Vector3::new(1.0, 0.0, 0.0),
    );

    println!(
        "rotation after Gaussian::new(.., Vector4::zeros(), ..) = {:?}",
        g.rotation
    );
    assert!(
        all_finite(g.rotation.iter().copied()),
        "Gaussian::new normalised a zero quaternion into {:?}",
        g.rotation
    );
    assert!(
        all_finite(g.covariance().iter().copied()),
        "the covariance built from a zero quaternion is not finite"
    );
    assert!(
        (g.rotation.norm() - 1.0).abs() < 1e-6,
        "the stored quaternion must be a unit quaternion, got norm {}",
        g.rotation.norm()
    );
    let det = g.rotation_matrix().determinant();
    assert!((det - 1.0).abs() < 1e-5, "determinant {det}");
}

/// Control: the same constructor with a real quaternion is unaffected.
#[test]
fn gaussian_new_still_normalizes_a_real_quaternion() {
    let q = UnitQuaternion::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(1.0f32, 0.0, 1.0)),
        0.5,
    );
    let g = Gaussian::new(
        Point3::new(0.0, 0.0, 3.0),
        Vector3::new(0.2, 0.1, 0.05),
        Vector4::new(q.i, q.j, q.k, q.w) * 3.0,
        Vector3::new(1.0, 0.0, 0.0),
    );
    assert!(
        (g.rotation.norm() - 1.0).abs() < 1e-6,
        "norm {}",
        g.rotation.norm()
    );
    let stored = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        g.rotation[3],
        g.rotation[0],
        g.rotation[1],
        g.rotation[2],
    ));
    let dot = stored.coords.dot(&q.coords).abs();
    assert!(dot > 1.0 - 1e-5, "rotation changed direction: dot {dot}");
}
