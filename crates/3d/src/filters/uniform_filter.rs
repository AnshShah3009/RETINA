use nalgebra::Point3;

/// Uniform downsampling: keep every `every_k`-th point.
pub fn uniform_downsample(points: &[Point3<f64>], every_k: usize) -> Vec<Point3<f64>> {
    if every_k == 0 {
        return points.to_vec();
    }
    points.iter().step_by(every_k).copied().collect()
}
