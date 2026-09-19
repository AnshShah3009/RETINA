use nalgebra::{Point3, Vector3};

/// Transform a point cloud in place by a 4x4 homogeneous matrix.
pub fn transform_points(points: &mut [Point3<f64>], transform: &nalgebra::Matrix4<f64>) {
    for p in points.iter_mut() {
        let h = transform * nalgebra::Vector4::new(p.x, p.y, p.z, 1.0);
        let w = if h.w.abs() > 1e-15 { h.w } else { 1.0 };
        *p = Point3::new(h.x / w, h.y / w, h.z / w);
    }
}

/// Create a uniform color array for `num_points` points.
pub fn paint_uniform(num_points: usize, color: &Vector3<f64>) -> Vec<Vector3<f64>> {
    vec![*color; num_points]
}
