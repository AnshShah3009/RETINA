//! A normal that was not measured must not be handed to the caller in a
//! plausible-looking shape.
//!
//! # The defect
//!
//! `mesh::reconstruction::compute_point_normals` is documented as
//! "compute normals using PCA (simplified)". It is a placeholder: it returns the
//! cloud's own normals if it has them, and otherwise
//!
//! ```ignore
//! // Default: return upward normals
//! vec![Vector3::new(0.0, 1.0, 0.0); n]
//! ```
//!
//! for every point.
//!
//! That is a fabricated success in the exact shape this crate has been removing
//! elsewhere. The result has precisely the right **length**, so the obvious
//! check `assert_eq!(normals.len(), points.len())` passes; every element is a
//! unit vector, so nothing downstream — a shading term, a back-face test, a
//! surface-area weighting — can tell it apart from a measurement. A caller that
//! checks only `is_empty()` would be told the cloud was fully oriented along +Y.
//!
//! The sibling precedent is explicit: `gpu::point_cloud::voxel_to_point_normal_transfer`
//! used to return a zero vector per point — a valid-length result carrying no
//! information — and was changed to return `Vec::new()` for exactly this reason.
//!
//! # Measurement
//!
//! A three-point cloud with no normals:
//!
//! ```text
//! before: [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
//! after:  []
//! ```
//!
//! Note the deliberate consequence: in the absent case the returned length is
//! **not** `points.len()`. That is the only way a caller can detect the absence at
//! all, and it is the same trade `voxel_to_point_normal_transfer` made.

use cv_3d::mesh::reconstruction::compute_point_normals;
use cv_core::PointCloud;
use nalgebra::{Point3, Vector3};

/// A cloud with no normals has none to report.
#[test]
fn a_cloud_without_normals_reports_absence_rather_than_fabricating_them() {
    let cloud = PointCloud::new(vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]);
    assert!(
        cloud.normals.is_none(),
        "CONTROL of the fixture: this cloud really has no normals"
    );

    let normals = compute_point_normals(&cloud, 10);
    assert!(
        normals.is_empty(),
        "a cloud with no normals must produce no normals, not one (0, 1, 0) per \
         point. Got {normals:?} - a valid-length, plausible-valued answer that no \
         caller could distinguish from a measurement."
    );
}

/// The sharper statement of the same defect: no element of the result may be the
/// constant the old default produced.
#[test]
fn no_fabricated_upward_normal_is_returned() {
    let cloud = PointCloud::new((0..10).map(|i| Point3::new(i as f32, 0.0, 0.0)).collect());
    let normals = compute_point_normals(&cloud, 5);
    assert!(
        !normals.iter().any(|n| *n == Vector3::new(0.0, 1.0, 0.0)),
        "the default upward normal is a fabrication, not a result"
    );
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
    let cloud = cv_3d::mesh::reconstruction::create_sphere_point_cloud(
        Point3::new(0.0, 0.0, 0.0),
        1.0,
        50,
    );
    let normals = compute_point_normals(&cloud, 5);
    assert_eq!(
        normals.len(),
        50,
        "CONTROL: the in-module test that pins this length must keep passing"
    );
}