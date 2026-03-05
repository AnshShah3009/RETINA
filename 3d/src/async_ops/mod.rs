//! Async CPU Operations
//!
//! Provides async/await support for CPU-based operations using tokio.
//! All operations are non-blocking and can run concurrently.

use crate::gpu;
use nalgebra::{Matrix4, Point3, Vector3};
use std::future::Future;
use tokio::task;

/// Async point cloud operations
pub mod point_cloud {
    use super::*;

    /// Async transform point cloud
    pub async fn transform_async(
        points: Vec<Point3<f32>>,
        transform: Matrix4<f32>,
    ) -> crate::Result<Vec<Point3<f32>>> {
        Ok(task::spawn_blocking(move || {
            points
                .iter()
                .map(|p| transform.transform_point(p))
                .collect()
        })
        .await?)
    }

    /// Async batch transform
    pub async fn batch_transform_async(
        point_clouds: Vec<Vec<Point3<f32>>>,
        transforms: Vec<Matrix4<f32>>,
    ) -> crate::Result<Vec<Vec<Point3<f32>>>> {
        if point_clouds.len() != transforms.len() {
            return Err(crate::Error::RuntimeError(format!(
                "Point cloud count ({}) does not match transform count ({})",
                point_clouds.len(),
                transforms.len()
            )));
        }

        let futures: Vec<_> = point_clouds
            .into_iter()
            .zip(transforms.into_iter())
            .map(|(points, transform)| transform_async(points, transform))
            .collect();

        futures::future::try_join_all(futures).await
    }

    /// Async voxel downsampling
    pub async fn voxel_downsample_async(
        points: Vec<Point3<f32>>,
        voxel_size: f32,
    ) -> crate::Result<Vec<Point3<f32>>> {
        Ok(
            task::spawn_blocking(move || gpu::point_cloud::voxel_downsample(&points, voxel_size))
                .await?,
        )
    }

    /// Async normal computation
    pub async fn compute_normals_async(
        points: Vec<Point3<f32>>,
        k: usize,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Ok(
            task::spawn_blocking(move || gpu::point_cloud::compute_normals_simple(&points, k))
                .await?,
        )
    }

    /// Async voxel-based normal computation
    pub async fn voxel_based_normals_async(
        points: Vec<Point3<f32>>,
        voxel_size: f32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Ok(task::spawn_blocking(move || {
            gpu::point_cloud::voxel_based_normals_simple(&points, voxel_size)
        })
        .await?)
    }

    /// Async voxel to point normal transfer
    pub async fn voxel_to_point_normal_transfer_async(
        points: Vec<Point3<f32>>,
        voxel_normals: Vec<Vector3<f32>>,
        voxel_size: f32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Ok(task::spawn_blocking(move || {
            gpu::point_cloud::voxel_to_point_normal_transfer(&points, &voxel_normals, voxel_size)
        })
        .await?)
    }

    /// Async approximate normal computation
    pub async fn approximate_normals_async(
        points: Vec<Point3<f32>>,
        k: usize,
        epsilon: f32,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        Ok(task::spawn_blocking(move || {
            gpu::point_cloud::approximate_normals_simple(&points, k, epsilon)
        })
        .await?)
    }

    /// Async batch voxel-based normal computation
    pub async fn batch_voxel_based_normals_async(
        point_clouds: Vec<Vec<Point3<f32>>>,
        voxel_size: f32,
    ) -> crate::Result<Vec<Vec<Vector3<f32>>>> {
        let futures: Vec<_> = point_clouds
            .into_iter()
            .map(|points| voxel_based_normals_async(points, voxel_size))
            .collect();
        futures::future::try_join_all(futures).await
    }

    /// Async batch approximate normal computation
    pub async fn batch_approximate_normals_async(
        point_clouds: Vec<Vec<Point3<f32>>>,
        k: usize,
        epsilon: f32,
    ) -> crate::Result<Vec<Vec<Vector3<f32>>>> {
        let futures: Vec<_> = point_clouds
            .into_iter()
            .map(|points| approximate_normals_async(points, k, epsilon))
            .collect();
        futures::future::try_join_all(futures).await
    }
}

/// Async registration operations
pub mod registration {
    use super::*;

    /// Async ICP registration
    pub async fn icp_async(
        source: Vec<Point3<f32>>,
        target: Vec<Point3<f32>>,
        target_normals: Vec<Vector3<f32>>,
        max_distance: f32,
        max_iterations: usize,
    ) -> crate::Result<Option<Matrix4<f32>>> {
        let result = task::spawn_blocking(move || {
            gpu::registration::icp_point_to_plane(
                &source,
                &target,
                &target_normals,
                max_distance,
                max_iterations,
            )
        })
        .await;

        match result {
            Ok(Ok(matrix)) => Ok(Some(matrix)),
            Ok(Err(_)) => Ok(None),
            Err(e) => Err(crate::Error::RuntimeError(format!(
                "Async task error: {}",
                e
            ))),
        }
    }

    /// Async batch ICP
    pub async fn batch_icp_async(
        sources: Vec<Vec<Point3<f32>>>,
        targets: Vec<Vec<Point3<f32>>>,
        target_normals: Vec<Vec<Vector3<f32>>>,
        max_distance: f32,
        max_iterations: usize,
    ) -> crate::Result<Vec<Option<Matrix4<f32>>>> {
        let futures: Vec<_> = sources
            .into_iter()
            .zip(targets.into_iter())
            .zip(target_normals.into_iter())
            .map(|((src, tgt), normals)| icp_async(src, tgt, normals, max_distance, max_iterations))
            .collect();

        futures::future::try_join_all(futures).await
    }
}

/// Async mesh operations
pub mod mesh {
    use super::*;

    /// Async Laplacian smoothing
    pub async fn laplacian_smooth_async(
        vertices: Vec<Point3<f32>>,
        faces: Vec<[usize; 3]>,
        iterations: usize,
        lambda: f32,
    ) -> crate::Result<Vec<Point3<f32>>> {
        Ok(task::spawn_blocking(move || {
            let mut verts = vertices;
            let _ = gpu::mesh::laplacian_smooth(&mut verts, &faces, iterations, lambda);
            verts
        })
        .await?)
    }

    /// Async vertex normal computation
    pub async fn compute_normals_async(
        vertices: Vec<Point3<f32>>,
        faces: Vec<[usize; 3]>,
    ) -> crate::Result<Vec<Vector3<f32>>> {
        match task::spawn_blocking(move || gpu::mesh::compute_vertex_normals(&vertices, &faces))
            .await
        {
            Ok(result) => {
                result.map_err(|e| crate::Error::RuntimeError(format!("GPU compute failed: {}", e)))
            }
            Err(e) => Err(crate::Error::RuntimeError(format!(
                "Async task error: {}",
                e
            ))),
        }
    }
}

/// Async ray casting operations
pub mod raycasting {
    use super::*;

    /// Async batch ray casting
    pub async fn cast_rays_async(
        ray_origins: Vec<Point3<f32>>,
        ray_directions: Vec<Vector3<f32>>,
        mesh_vertices: Vec<Point3<f32>>,
        mesh_faces: Vec<[usize; 3]>,
    ) -> crate::Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>> {
        match task::spawn_blocking(move || {
            gpu::raycasting::cast_rays(&ray_origins, &ray_directions, &mesh_vertices, &mesh_faces)
        })
        .await
        {
            Ok(result) => {
                result.map_err(|e| crate::Error::RuntimeError(format!("GPU compute failed: {}", e)))
            }
            Err(e) => Err(crate::Error::RuntimeError(format!(
                "Async task error: {}",
                e
            ))),
        }
    }
}

/// Async TSDF operations
pub mod tsdf {
    use super::*;

    /// Async depth frame integration
    pub async fn integrate_depth_async(
        depth_image: Vec<f32>,
        width: u32,
        height: u32,
        camera_pose: Matrix4<f32>,
        intrinsics: [f32; 4],
        mut tsdf_volume: Vec<f32>,
        mut weights: Vec<f32>,
        voxel_size: f32,
        truncation: f32,
    ) -> crate::Result<(Vec<f32>, Vec<f32>)> {
        Ok(task::spawn_blocking(move || {
            let _ = gpu::tsdf::integrate_depth(
                &depth_image,
                width,
                height,
                &camera_pose,
                &intrinsics,
                &mut tsdf_volume,
                &mut weights,
                voxel_size,
                truncation,
            );
            (tsdf_volume, weights)
        })
        .await?)
    }
}

/// Pipeline combinators
pub mod pipeline {
    use super::*;

    /// Chain async operations
    pub async fn chain<T, F, Fut>(initial: T, ops: Vec<F>) -> T
    where
        F: Fn(T) -> Fut,
        Fut: Future<Output = T>,
    {
        let mut result = initial;
        for op in ops {
            result = op(result).await;
        }
        result
    }

    /// Parallel map with concurrency limit
    pub async fn parallel_map<T, R, F, Fut>(items: Vec<T>, limit: usize, f: F) -> Vec<R>
    where
        T: Send + 'static,
        R: Send + 'static,
        F: Fn(T) -> Fut + Send + Clone + Sync,
        Fut: Future<Output = R> + Send,
    {
        use futures::stream::{self, StreamExt};

        stream::iter(items)
            .map(|item| {
                let f = f.clone();
                async move { f(item).await }
            })
            .buffer_unordered(limit)
            .collect()
            .await
    }

    /// Run with timeout
    pub async fn with_timeout<T>(
        future: impl Future<Output = T>,
        dur: std::time::Duration,
    ) -> Option<T> {
        tokio::time::timeout(dur, future).await.ok()
    }
}

pub use mesh::{compute_normals_async as mesh_compute_normals_async, laplacian_smooth_async};
pub use pipeline::{chain, parallel_map, with_timeout};
/// Re-exports
pub use point_cloud::{
    batch_transform_async, compute_normals_async, transform_async, voxel_downsample_async,
};
pub use raycasting::cast_rays_async;
pub use registration::{batch_icp_async, icp_async};
pub use tsdf::integrate_depth_async;
