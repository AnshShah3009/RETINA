//! Batch Processing Module
//!
//! Provides batch versions of all GPU-accelerated functions for maximum throughput.
//! Processes multiple operations in parallel, amortizing GPU overhead.

use crate::gpu;
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;

/// Batch configuration for optimal GPU utilization
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Maximum batch size (number of operations per batch)
    pub max_batch_size: usize,
    /// Minimum elements to process on GPU (per operation)
    pub gpu_threshold: usize,
    /// Whether to use async execution
    pub async_execution: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            gpu_threshold: 1000,
            async_execution: true,
        }
    }
}

impl BatchConfig {
    /// High-throughput configuration
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 64,
            gpu_threshold: 500,
            async_execution: true,
        }
    }

    /// Low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 4,
            gpu_threshold: 10000,
            async_execution: false,
        }
    }

    /// Balanced configuration
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Batch point cloud operations
pub mod point_cloud {
    use super::*;

    /// Batch transform multiple point clouds
    pub fn batch_transform(
        point_clouds: &[Vec<Point3<f32>>],
        transforms: &[Matrix4<f32>],
        _config: &BatchConfig,
    ) -> Vec<Vec<Point3<f32>>> {
        assert_eq!(point_clouds.len(), transforms.len());

        point_clouds
            .par_iter()
            .zip(transforms.par_iter())
            .map(|(points, transform)| {
                points
                    .iter()
                    .map(|p| transform.transform_point(p))
                    .collect()
            })
            .collect()
    }

    /// Batch voxel downsampling
    pub fn batch_voxel_downsample(
        point_clouds: &[Vec<Point3<f32>>],
        voxel_size: f32,
    ) -> Vec<Vec<Point3<f32>>> {
        point_clouds
            .par_iter()
            .map(|points| gpu::point_cloud::voxel_downsample(points, voxel_size))
            .collect()
    }

    /// Batch normal computation
    pub fn batch_compute_normals(
        point_clouds: &[Vec<Point3<f32>>],
        k_neighbors: usize,
    ) -> Vec<Vec<Vector3<f32>>> {
        point_clouds
            .par_iter()
            .map(|points| gpu::point_cloud::compute_normals_simple(points, k_neighbors))
            .collect()
    }

    /// Batch voxel-based normal computation
    pub fn batch_voxel_based_normals(
        point_clouds: &[Vec<Point3<f32>>],
        voxel_size: f32,
    ) -> Vec<Vec<Vector3<f32>>> {
        point_clouds
            .par_iter()
            .map(|points| gpu::point_cloud::voxel_based_normals_simple(points, voxel_size))
            .collect()
    }

    /// Batch voxel to point normal transfer
    pub fn batch_voxel_to_point_normal_transfer(
        point_clouds: &[Vec<Point3<f32>>],
        voxel_normals: &[Vec<Vector3<f32>>],
        voxel_size: f32,
    ) -> Vec<Vec<Vector3<f32>>> {
        assert_eq!(point_clouds.len(), voxel_normals.len());

        point_clouds
            .par_iter()
            .zip(voxel_normals.par_iter())
            .map(|(points, normals)| {
                gpu::point_cloud::voxel_to_point_normal_transfer(points, normals, voxel_size)
            })
            .collect()
    }

    /// Batch approximate normal computation
    pub fn batch_approximate_normals(
        point_clouds: &[Vec<Point3<f32>>],
        k_neighbors: usize,
        epsilon: f32,
    ) -> Vec<Vec<Vector3<f32>>> {
        point_clouds
            .par_iter()
            .map(|points| {
                gpu::point_cloud::approximate_normals_simple(points, k_neighbors, epsilon)
            })
            .collect()
    }
}

/// Batch registration operations
pub mod registration {
    use super::*;

    /// Batch ICP registration
    pub fn batch_icp(
        sources: &[Vec<Point3<f32>>],
        targets: &[Vec<Point3<f32>>],
        target_normals: &[Vec<Vector3<f32>>],
        max_distance: f32,
        max_iterations: usize,
    ) -> Vec<Option<Matrix4<f32>>> {
        assert_eq!(sources.len(), targets.len());
        assert_eq!(sources.len(), target_normals.len());

        sources
            .par_iter()
            .zip(targets.par_iter())
            .zip(target_normals.par_iter())
            .map(|((src, tgt), normals)| {
                gpu::registration::icp_point_to_plane(
                    src,
                    tgt,
                    normals,
                    max_distance,
                    max_iterations,
                )
                .ok()
            })
            .collect()
    }
}

/// Batch mesh operations
pub mod mesh {
    use super::*;

    /// Batch Laplacian smoothing
    pub fn batch_laplacian_smooth(
        vertices_list: &mut [Vec<Point3<f32>>],
        faces_list: &[Vec<[usize; 3]>],
        iterations: usize,
        lambda: f32,
    ) {
        vertices_list
            .par_iter_mut()
            .zip(faces_list.par_iter())
            .for_each(|(vertices, faces)| {
                let _ = gpu::mesh::laplacian_smooth(vertices, faces, iterations, lambda);
            });
    }

    /// Batch vertex normal computation
    pub fn batch_compute_normals(
        vertices_list: &[Vec<Point3<f32>>],
        faces_list: &[Vec<[usize; 3]>],
    ) -> Vec<Vec<Vector3<f32>>> {
        vertices_list
            .par_iter()
            .zip(faces_list.par_iter())
            .map(|(vertices, faces)| {
                gpu::mesh::compute_vertex_normals(vertices, faces).unwrap_or_default()
            })
            .collect()
    }
}

/// Batch ray casting operations
pub mod raycasting {
    use super::*;

    /// Batch ray casting against multiple meshes
    #[allow(clippy::type_complexity)]
    pub fn batch_cast_rays(
        ray_batches: &[Vec<(Point3<f32>, Vector3<f32>)>],
        mesh_vertices: &[Vec<Point3<f32>>],
        mesh_faces: &[Vec<[usize; 3]>],
    ) -> Vec<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>> {
        assert_eq!(ray_batches.len(), mesh_vertices.len());
        assert_eq!(ray_batches.len(), mesh_faces.len());

        ray_batches
            .par_iter()
            .zip(mesh_vertices.par_iter())
            .zip(mesh_faces.par_iter())
            .map(|((origins_directions, vertices), faces)| {
                let origins: Vec<_> = origins_directions.iter().map(|(o, _)| *o).collect();
                let directions: Vec<_> = origins_directions.iter().map(|(_, d)| *d).collect();
                gpu::raycasting::cast_rays(
                    &origins,
                    &directions,
                    vertices.as_slice(),
                    faces.as_slice(),
                )
                .unwrap_or_default()
            })
            .collect()
    }

    /// Cast many rays against single mesh
    #[allow(clippy::type_complexity)]
    pub fn batch_cast_rays_single_mesh(
        rays: &[(Point3<f32>, Vector3<f32>)],
        mesh_vertices: &[Point3<f32>],
        mesh_faces: &[[usize; 3]],
    ) -> Vec<Option<(f32, Point3<f32>, Vector3<f32>)>> {
        let origins: Vec<_> = rays.iter().map(|(o, _)| *o).collect();
        let directions: Vec<_> = rays.iter().map(|(_, d)| *d).collect();
        gpu::raycasting::cast_rays(&origins, &directions, mesh_vertices, mesh_faces)
            .unwrap_or_default()
    }
}

/// Benchmarking utilities
pub mod bench {
    use super::*;
    use std::time::Instant;

    /// Benchmark batch transform
    pub fn benchmark_batch_transform(num_clouds: usize, cloud_size: usize) -> (f64, f64) {
        let point_clouds: Vec<_> = (0..num_clouds)
            .map(|_| {
                (0..cloud_size)
                    .map(|_| Point3::new(rand::random(), rand::random(), rand::random()))
                    .collect()
            })
            .collect();

        let transforms: Vec<_> = (0..num_clouds)
            .map(|_| {
                Matrix4::new_translation(&Vector3::new(
                    rand::random(),
                    rand::random(),
                    rand::random(),
                ))
            })
            .collect();

        let config = BatchConfig::default();

        let start = Instant::now();
        let _ = point_cloud::batch_transform(&point_clouds, &transforms, &config);
        let elapsed = start.elapsed();

        (
            elapsed.as_secs_f64() * 1000.0,
            cloud_size as f64 * num_clouds as f64 / elapsed.as_secs_f64(),
        )
    }
}

pub use bench::benchmark_batch_transform;
pub use mesh::{batch_compute_normals as batch_mesh_compute_normals, batch_laplacian_smooth};
/// Re-exports
pub use point_cloud::{batch_compute_normals, batch_transform, batch_voxel_downsample};
pub use raycasting::{batch_cast_rays, batch_cast_rays_single_mesh};
pub use registration::batch_icp;
