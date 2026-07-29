//! Open3D-equivalent point cloud filtering operations.
//!
//! Standalone functions operating on `&[Point3<f64>]` slices — no dependency on
//! HAL, GPU, or the `PointCloud` struct from cv-scientific.
//!
//! All heavy loops use **rayon** for automatic parallelism.

use nalgebra::{Point3, Vector3};

pub mod bounding_box;
pub mod crop_filter;
pub mod normal_estimation;
pub mod radius_filter;
pub mod statistical_filter;
pub mod transform;
pub mod uniform_filter;
pub mod voxel_filter;

pub use bounding_box::{compute_aabb, compute_obb, OrientedBoundingBox};
pub use crop_filter::crop_aabb;
pub use normal_estimation::estimate_normals_knn;
pub use radius_filter::radius_outlier_removal;
pub use statistical_filter::statistical_outlier_removal;
pub use transform::{paint_uniform, transform_points};
pub use uniform_filter::uniform_downsample;
pub use voxel_filter::voxel_downsample;

/// Result of voxel downsampling, optionally including normals and colors.
#[derive(Debug, Clone)]
pub struct VoxelDownsampleResult {
    pub points: Vec<Point3<f64>>,
    pub normals: Option<Vec<Vector3<f64>>>,
    pub colors: Option<Vec<Vector3<f64>>>,
}

#[inline]
fn dist_sq(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a small cluster around the origin.
    fn cluster(n: usize, spacing: f64) -> Vec<Point3<f64>> {
        let mut pts = Vec::with_capacity(n * n * n);
        let size = (n as f64 - 1.0) * spacing;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(Point3::new(
                        i as f64 * spacing - size / 2.0,
                        j as f64 * spacing - size / 2.0,
                        k as f64 * spacing - size / 2.0,
                    ));
                }
            }
        }
        pts
    }

    #[test]
    fn test_statistical_outlier_removal() {
        let mut points = cluster(3, 0.1);
        points.push(Point3::new(100.0, 0.0, 0.0));
        points.push(Point3::new(0.0, 100.0, 0.0));
        points.push(Point3::new(0.0, 0.0, 100.0));

        let (filtered, indices) = statistical_outlier_removal(&points, 5, 2.0);

        assert_eq!(filtered.len(), 27);
        assert_eq!(indices.len(), 27);

        for &idx in &indices {
            assert!(idx < 27);
        }
    }

    #[test]
    fn test_radius_outlier_removal() {
        let mut points = cluster(3, 0.1);
        points.push(Point3::new(50.0, 0.0, 0.0));
        points.push(Point3::new(-50.0, 0.0, 0.0));

        let (filtered, indices) = radius_outlier_removal(&points, 0.5, 2);

        assert_eq!(filtered.len(), 27);
        for &idx in &indices {
            assert!(idx < 27);
        }
    }

    #[test]
    fn test_voxel_downsample_cube() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];

        let result = voxel_downsample(&points, None, None, 2.0);
        assert_eq!(result.points.len(), 1);

        let p = &result.points[0];
        assert!((p.x - 0.5).abs() < 1e-10);
        assert!((p.y - 0.5).abs() < 1e-10);
        assert!((p.z - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_voxel_downsample_with_normals() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.1, 0.0, 0.0)];
        let normals = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0)];

        let result = voxel_downsample(&points, Some(&normals), None, 1.0);
        assert_eq!(result.points.len(), 1);

        let n = &result.normals.as_ref().unwrap()[0];
        let expected = Vector3::new(0.0, 1.0, 1.0).normalize();
        assert!((n - expected).norm() < 1e-10);
    }

    #[test]
    fn test_uniform_downsample() {
        let points: Vec<Point3<f64>> = (0..10).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();

        let down = uniform_downsample(&points, 2);
        assert_eq!(down.len(), 5);
        assert!((down[0].x - 0.0).abs() < 1e-15);
        assert!((down[1].x - 2.0).abs() < 1e-15);
        assert!((down[4].x - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_crop_aabb() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];
        let min_b = Point3::new(-0.5, -0.5, -0.5);
        let max_b = Point3::new(1.5, 1.5, 1.5);

        let (cropped, indices) = crop_aabb(&points, &min_b, &max_b);
        assert_eq!(cropped.len(), 2);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_compute_aabb() {
        let points = vec![
            Point3::new(-1.0, 2.0, 3.0),
            Point3::new(4.0, -5.0, 6.0),
            Point3::new(0.0, 0.0, 0.0),
        ];
        let (min_b, max_b) = compute_aabb(&points);
        assert!((min_b.x - (-1.0)).abs() < 1e-15);
        assert!((min_b.y - (-5.0)).abs() < 1e-15);
        assert!((min_b.z - 0.0).abs() < 1e-15);
        assert!((max_b.x - 4.0).abs() < 1e-15);
        assert!((max_b.y - 2.0).abs() < 1e-15);
        assert!((max_b.z - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_estimate_normals_knn_plane() {
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
            }
        }

        let normals = estimate_normals_knn(&points, 8);
        assert_eq!(normals.len(), points.len());

        for (i, n) in normals.iter().enumerate() {
            assert!(
                n.z.abs() > 0.9,
                "Point {} normal z-component should be ~1.0, got {:?}",
                i,
                n
            );
            assert!(
                (n.norm() - 1.0).abs() < 1e-10,
                "Normal {} should be unit length, norm = {}",
                i,
                n.norm()
            );
        }
    }

    #[test]
    fn test_transform_points() {
        let mut points = vec![Point3::new(1.0, 0.0, 0.0)];
        let mut t = nalgebra::Matrix4::identity();
        t[(0, 3)] = 10.0;
        t[(1, 3)] = 20.0;
        t[(2, 3)] = 30.0;

        transform_points(&mut points, &t);
        assert!((points[0].x - 11.0).abs() < 1e-10);
        assert!((points[0].y - 20.0).abs() < 1e-10);
        assert!((points[0].z - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_paint_uniform() {
        let colors = paint_uniform(5, &Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(colors.len(), 5);
        for c in &colors {
            assert!((c.x - 1.0).abs() < 1e-15);
            assert!(c.y.abs() < 1e-15);
            assert!(c.z.abs() < 1e-15);
        }
    }

    #[test]
    fn test_compute_obb() {
        let points: Vec<Point3<f64>> = (0..20).map(|i| Point3::new(i as f64, 0.0, 0.0)).collect();

        let obb = compute_obb(&points);
        let ax = obb.axes[0];
        assert!(ax.x.abs() > 0.9);
        assert!(obb.extents.x > 5.0);
    }
}
