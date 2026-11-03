use nalgebra::{Point3, Vector3};

/// Estimate point cloud normals using PCA over k nearest neighbours.
///
/// [`cv_pointcloud::normals::estimate_normals_knn`] (Open3D's
/// `estimate_normals(KDTreeSearchParamKNN(k))`), then oriented into the
/// `z >= 0` half-space. Open3D leaves the sign of a freshly estimated normal
/// to its eigensolver, so the two agree up to sign.
///
/// The neighbourhood of a point is its `k` nearest points *including the
/// point itself*; `k` larger than the cloud uses the whole cloud, and a
/// neighbourhood of fewer than three points gives the default `(0, 0, 1)`.
/// A point with a NaN or infinite coordinate gets `(0, 0, 1)` and is never in
/// another point's neighbourhood.
pub fn estimate_normals_knn(points: &[Point3<f64>], k: usize) -> Vec<Vector3<f64>> {
    let mut normals = cv_pointcloud::normals::estimate_normals_knn(points, k);
    for n in &mut normals {
        if n.z < 0.0 {
            *n = -*n;
        }
    }
    normals
}
