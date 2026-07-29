use nalgebra::Point3;

/// Crop point cloud to an axis-aligned bounding box.
///
/// # Returns
/// `(cropped_points, inlier_indices)`.
pub fn crop_aabb(
    points: &[Point3<f64>],
    min_bound: &Point3<f64>,
    max_bound: &Point3<f64>,
) -> (Vec<Point3<f64>>, Vec<usize>) {
    let mut out_points = Vec::new();
    let mut out_indices = Vec::new();
    for (i, p) in points.iter().enumerate() {
        if p.x >= min_bound.x
            && p.x <= max_bound.x
            && p.y >= min_bound.y
            && p.y <= max_bound.y
            && p.z >= min_bound.z
            && p.z <= max_bound.z
        {
            out_points.push(*p);
            out_indices.push(i);
        }
    }
    (out_points, out_indices)
}
