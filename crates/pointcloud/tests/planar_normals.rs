//! A planar cloud must get normals perpendicular to its plane.
//!
//! `estimate_normals` finds the smallest eigenvector of the neighbourhood
//! covariance with `cv_math::linalg::min_eigenvector_3x3`. That routine is
//! generic over the element type and, when instantiated at `f32`, computes the
//! determinant term `det((A - qI) / p)` of a three-point neighbourhood: the
//! out-of-plane eigenvalue is exactly zero and the in-plane spread is small
//! (the three neighbours are close to collinear), so the determinant saturates
//! at its `±1` clamp. The decomposition then treats the two small eigenvalues
//! as *repeated* and returns an arbitrary vector inside their shared span -
//! which here contains the in-plane direction. The result was a normal lying
//! *in* the plane.
//!
//! This cloud lies entirely in `z = 0`, so every normal must be `(0, 0, ±1)`.
//! For the third point the three remaining points are nearly collinear, and
//! before the fix the returned normal was `(0.013, 0.9999, 0)` (dot product
//! with the true normal was 0), i.e. horizontal. The same covariance solved in
//! `f64` yields `(0, 0, 1)`.

use cv_core::PointCloud;
use cv_pointcloud::estimate_normals;
use nalgebra::Point3;

#[test]
fn a_planar_cloud_gets_normals_perpendicular_to_the_plane() {
    let points = vec![
        Point3::new(-1.3555131f32, -0.45285225f32, 0.0),
        Point3::new(1.3084972f32, -0.4922645f32, 0.0),
        Point3::new(-1.4549813f32, 0.295022f32, 0.0),
        Point3::new(0.49400902f32, -0.46105266f32, 0.0),
    ];
    let mut pc = PointCloud::new(points);

    estimate_normals(&mut pc, 3);

    let normals = pc.normals.as_ref().expect("normals must be set");
    for (i, normal) in normals.iter().enumerate() {
        assert!(
            (normal.norm() - 1.0).abs() < 1e-4,
            "normal {i} is not unit length: {normal:?}"
        );
        assert!(
            normal.z.abs() > 0.99,
            "normal {i} = {normal:?} lies in the z = 0 plane the points span"
        );
    }
}
