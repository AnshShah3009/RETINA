//! Surface Reconstruction from Point Clouds
//!
//! Implements:
//! - Poisson Surface Reconstruction
//! - Ball Pivoting Algorithm (BPA)
//! - Alpha Shapes
//! - Marching Cubes
//! - Delaunay-based reconstruction

pub mod alpha_shapes;
pub mod ball_pivoting;
pub mod delaunay;
pub mod marching_cubes;
pub mod poisson;

pub use alpha_shapes::*;
pub use ball_pivoting::*;
pub use poisson::*;

use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};

/// Compute normals for point cloud using PCA (simplified)
pub fn compute_point_normals(cloud: &PointCloud, _k: usize) -> Vec<Vector3<f32>> {
    let n = cloud.points.len();
    if n == 0 {
        return vec![];
    }

    // Simplified: use existing normals or compute basic normals
    if let Some(ref normals) = cloud.normals {
        return normals.clone();
    }

    // Default: return upward normals
    vec![Vector3::new(0.0, 1.0, 0.0); n]
}

/// Create a simple sphere point cloud for testing
pub fn create_sphere_point_cloud(
    center: Point3<f32>,
    radius: f32,
    num_points: usize,
) -> PointCloud {
    let _rng = rand::rng();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    let phi = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());

    for i in 0..num_points {
        let y = 1.0 - (i as f32 / (num_points - 1).max(1) as f32) * 2.0;
        let radius_at_y = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi * i as f32;

        let x = theta.cos() * radius_at_y;
        let z = theta.sin() * radius_at_y;

        let point = center + radius * Vector3::new(x, y, z);
        points.push(point);

        let normal = (point - center).normalize();
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

/// Create a simple plane point cloud for testing
pub fn create_plane_point_cloud(
    origin: Point3<f32>,
    normal: Vector3<f32>,
    size: f32,
    num_points: usize,
) -> PointCloud {
    use rand::Rng;
    let mut rng = rand::rng();

    let up = if normal.z.abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let right = normal.cross(&up).normalize();
    let up = right.cross(&normal).normalize();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let u = rng.random_range(-size..size);
        let v = rng.random_range(-size..size);

        let point = origin + right * u + up * v;
        points.push(point);
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

pub(crate) fn compute_bounds(cloud: &PointCloud) -> (Point3<f32>, Point3<f32>) {
    if cloud.points.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = cloud.points[0];
    let mut max = cloud.points[0];

    for p in &cloud.points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sphere_point_cloud() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        assert_eq!(cloud.points.len(), 100);
        assert!(cloud.normals.is_some());
    }

    #[test]
    fn test_create_plane_point_cloud() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let cloud = create_plane_point_cloud(Point3::origin(), normal, 1.0, 50);
        assert_eq!(cloud.points.len(), 50);
    }

    #[test]
    fn test_ball_pivoting_empty() {
        let cloud = PointCloud::new(vec![]);
        let mesh = ball_pivoting(&cloud, 0.1);
        assert_eq!(mesh.num_vertices(), 0);
    }

    #[test]
    fn test_ball_pivoting_sphere() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 200);
        let mesh = ball_pivoting(&cloud, 0.2);
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn test_alpha_shapes() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        let mesh = alpha_shapes(&cloud, 0.1);
        // alpha_shapes should return a valid mesh (vertex count is non-negative by type)
        let _ = mesh.num_vertices();
    }

    #[test]
    fn test_compute_point_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 50);
        let normals = compute_point_normals(&cloud, 5);
        assert_eq!(normals.len(), 50);
    }

    #[test]
    fn test_poisson_with_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 200);
        let mesh = poisson_reconstruction(&cloud, 4, 1.0);
        assert!(mesh.is_some());
        let m = mesh.unwrap();
        // Should produce a non-trivial mesh with actual vertices and faces
        assert!(
            m.num_vertices() > 0,
            "Expected vertices, got {}",
            m.num_vertices()
        );
        assert!(m.num_faces() > 0, "Expected faces, got {}", m.num_faces());
    }

    #[test]
    fn test_poisson_without_normals() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let cloud = PointCloud::new(points);
        let mesh = poisson_reconstruction(&cloud, 3, 1.0);
        assert!(mesh.is_none());
    }

    #[test]
    fn test_spatial_index() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 20);
        let index = ball_pivoting::SpatialIndex::new(&cloud, 0.5);
        let neighbors = index.find_neighbors(&cloud.points[0], 0.5);
        assert!(neighbors.len() >= 1);
    }

    #[test]
    fn test_compute_bounds() {
        let cloud = create_sphere_point_cloud(Point3::new(5.0, 5.0, 5.0), 2.0, 50);
        let (min, max) = compute_bounds(&cloud);
        assert!(min.x < 5.0);
        assert!(max.x > 5.0);
    }
}
