//! `compute_point_normals` estimates a missing normal from the points.
//!
//! It used to answer `(0, 1, 0)` for every point of a cloud that carried no
//! normals: the right length, a unit vector, and independent of the cloud.
//! The estimate is PCA over the neighbours, so two different planes do not
//! share one normal, and a cloud that already has normals still gets those
//! back unchanged.

use cv_3d::mesh::reconstruction::compute_point_normals;
use cv_core::PointCloud;
use nalgebra::{Point3, Vector3};

/// Three corners of the xy plane estimate +Z, not one `(0, 1, 0)` each.
#[test]
fn a_cloud_without_normals_is_estimated_from_the_plane() {
    let cloud = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]);
    assert!(cloud.normals.is_none());

    let normals = compute_point_normals(&cloud, 10);
    assert_eq!(normals.len(), cloud.points.len());
    for n in &normals {
        assert!(
            (n - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-3,
            "xy plane should estimate +Z, got {n:?}"
        );
    }
}

/// A tilted plane does not get the same normal as the xy plane. The old
/// default was `(0, 1, 0)` for both.
#[test]
fn two_planes_do_not_share_one_fabricated_normal() {
    let xy = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]);
    let tilted = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 1.0),
    ]);
    let n_xy = compute_point_normals(&xy, 5);
    let n_tilt = compute_point_normals(&tilted, 5);
    assert_eq!(n_xy.len(), 3);
    assert_eq!(n_tilt.len(), 3);
    let expected = Vector3::new(0.0, -1.0, 1.0).normalize();
    for n in &n_tilt {
        assert!(
            n.dot(&expected) > 0.99,
            "tilted plane should follow (0, -1, 1), got {n:?}"
        );
        assert!(
            n.dot(&n_xy[0]) < 0.8,
            "tilted and xy normals must differ, got {n:?} vs {:?}",
            n_xy[0]
        );
    }
}

/// CONTROL: a cloud that *does* carry normals still gets them back, unchanged and
/// in order. A fix that returned `Vec::new()` unconditionally would pass the two
/// tests above and fail here.
#[test]
fn control_a_cloud_with_normals_still_reports_them_verbatim() {
    let wanted = vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    ];
    let cloud = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ])
    .with_normals(wanted.clone())
    .expect("the fixture's normals are valid");

    let normals = compute_point_normals(&cloud, 10);
    assert_eq!(
        normals, wanted,
        "CONTROL: a cloud that carries normals must get them back unchanged"
    );
    assert_eq!(normals.len(), 3, "CONTROL: and one per point");
}

/// CONTROL: the fixture generators used elsewhere in this module produce clouds
/// that *do* carry normals, so the common path is unaffected.
#[test]
fn control_the_sphere_generator_still_reports_one_normal_per_point() {
    let cloud =
        cv_3d::mesh::reconstruction::create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 50);
    let normals = compute_point_normals(&cloud, 5);
    assert_eq!(
        normals.len(),
        50,
        "CONTROL: the in-module test that pins this length must keep passing"
    );
}
