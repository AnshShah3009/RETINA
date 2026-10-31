//! Degenerate-input tests for ray casting.
//!
//! Found by a scratch probe while auditing untested code: `Ray::new`
//! normalised its direction unconditionally, and `Vector3::normalize` returns
//! NaN for a zero vector. Every downstream `point_at`, dot product and
//! intersection then returned NaN with no error anywhere, and NaN propagates
//! through any transform that touches it.

use cv_3d::raycasting::Ray;
use nalgebra::{Point3, Vector3};

/// A zero direction has no normalisation, and inventing one is worse than
/// refusing.
#[test]
fn a_zero_direction_is_rejected_rather_than_producing_nan() {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let zero = Vector3::new(0.0, 0.0, 0.0);

    assert!(
        Ray::try_new(origin, zero).is_none(),
        "a zero-length direction must be rejected, not normalised into NaN"
    );
}

/// A non-finite direction is the same class of defect: it can only propagate.
#[test]
fn a_non_finite_direction_is_rejected() {
    let origin = Point3::new(0.0, 0.0, 0.0);
    for bad in [
        Vector3::new(f32::NAN, 0.0, 0.0),
        Vector3::new(0.0, f32::INFINITY, 0.0),
        Vector3::new(0.0, 0.0, f32::NEG_INFINITY),
    ] {
        assert!(
            Ray::try_new(origin, bad).is_none(),
            "{bad:?} must be rejected, not propagated"
        );
    }
}

/// `Ray::new` panics rather than returning a ray that cannot be used.
///
/// A panic is the right call here: silently returning NaN is what this defect
/// was, and a caller passing a zero direction has a bug that should surface at
/// the point it is made rather than as a mysteriously wrong result three layers
/// downstream.
#[test]
#[should_panic(expected = "degenerate direction")]
fn new_panics_on_a_zero_direction() {
    Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0));
}

/// A valid ray is normalised and remains finite, at any input length.
#[test]
fn a_valid_direction_is_normalised_and_stays_finite() {
    let origin = Point3::new(1.0, 2.0, 3.0);
    for d in [
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(0.0, 0.0, 1e-6),
        Vector3::new(3.0, 4.0, 12.0),
        Vector3::new(-1.0, -1.0, -1.0),
    ] {
        let ray = Ray::try_new(origin, d).expect("a valid direction");
        assert!(
            (ray.direction.norm() - 1.0).abs() < 1e-5,
            "{d:?} normalised to norm {}",
            ray.direction.norm()
        );
        let p = ray.point_at(2.0);
        assert!(
            p.coords.iter().all(|v| v.is_finite()),
            "point_at produced {p:?} from a valid ray"
        );
    }
}

/// A very small but non-zero direction is still a direction.
///
/// The guard is `> f32::EPSILON`, not `== 0.0`, so a legitimately tiny
/// direction is not rejected along with the degenerate ones - the test pins
/// that boundary so a future "fix" cannot quietly widen the rejection.
#[test]
fn a_tiny_but_real_direction_is_accepted() {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let tiny = Vector3::new(0.0, 0.0, 1e-7);
    let ray = Ray::try_new(origin, tiny).expect("1e-7 is a direction, just a small one");
    assert!(ray.direction.iter().all(|v| v.is_finite()));
    assert!((ray.direction.norm() - 1.0).abs() < 1e-5);
}
