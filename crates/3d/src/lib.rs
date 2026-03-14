#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
//! 3D Computer Vision Algorithms
//!
//! This crate provides core 3D computer vision algorithms including:
//! - Mesh processing and reconstruction
//! - TSDF volume integration
//! - Ray casting and intersection
//! - Spatial data structures
//! - Odometry computation
//! - GPU acceleration
//!
//! ## Modules
//!
//! - [`mesh`]: Triangle mesh operations, processing, and reconstruction
//! - [`tsdf`]: Truncated Signed Distance Field volume integration
//! - [`raycasting`]: Ray intersection with meshes
//! - [`spatial`]: Spatial data structures (KDTree, Octree, VoxelGrid)
//! - [`odometry`]: RGB-D odometry computation
//! - [`gpu`]: GPU-accelerated operations
//! - [`async_ops`]: Asynchronous operation pipelines
//! - [`batch`]: Batch processing utilities
//!
//! ## Key Types
//!
//! - [`TriangleMesh`]: Triangle mesh representation
//! - [`TSDFVolume`]: Truncated Signed Distance Field volume
//! - [`KDTree`]: K-dimensional tree for nearest neighbor search
//! - [`Octree`]: Hierarchical spatial data structure
//! - [`VoxelGrid`]: Voxelized representation
//! - [`Ray`]: 3D ray for raycasting
//! - [`RayHit`]: Ray intersection result
//!
//! ## Example: Ray Casting
//!
//! ```rust
//! use cv_3d::{raycasting::{Ray, cast_ray_mesh}, mesh::TriangleMesh};
//! use nalgebra::{Point3, Vector3};
//!
//! // Create a simple triangle mesh
//! let mesh = TriangleMesh::with_vertices_and_faces(
//!     vec![
//!         Point3::new(0.0, 0.0, 0.0),
//!         Point3::new(1.0, 0.0, 0.0),
//!         Point3::new(0.0, 1.0, 0.0),
//!     ],
//!     vec![[0, 1, 2]],
//! );
//!
//! // Cast a ray
//! let ray = Ray::new(
//!     Point3::new(0.5, 0.5, 1.0),
//!     Vector3::new(0.0, 0.0, -1.0),
//! );
//!
//! if let Some(hit) = cast_ray_mesh(&ray, &mesh) {
//!     println!("Hit at distance: {}", hit.distance);
//! }
//! ```
//!
//! ## Example: KDTree Nearest Neighbor
//!
//! ```rust
//! use cv_3d::spatial::KDTree;
//! use nalgebra::Point3;
//!
//! let mut tree = KDTree::new();
//! tree.insert(Point3::new(0.0, 0.0, 0.0), 0);
//! tree.insert(Point3::new(1.0, 0.0, 0.0), 1);
//! tree.insert(Point3::new(0.0, 1.0, 0.0), 2);
//!
//! if let Some((point, idx, dist)) = tree.nearest_neighbor(&Point3::new(0.5, 0.1, 0.0)) {
//!     println!("Nearest: point={:?}, idx={}, dist={}", point, idx, dist);
//! }
//! ```

pub mod async_ops;
pub mod batch;
pub mod gpu;
pub mod mesh;
pub mod odometry;
pub mod raycasting;
pub mod spatial;
pub mod tsdf;
pub mod visibility;

pub use async_ops::{
    mesh as async_mesh, pipeline as async_pipeline, point_cloud as async_point_cloud,
    raycasting as async_raycasting, registration as async_registration, tsdf as async_tsdf,
};

pub use batch::{
    bench as batch_bench, mesh as batch_mesh, point_cloud as batch_point_cloud,
    raycasting as batch_raycasting, registration as batch_registration, BatchConfig,
};

pub use cv_core::PointCloud;
pub use gpu::{
    force_cpu_mode, force_gpu_mode, gpu_info, is_gpu_available, mesh as gpu_mesh,
    point_cloud as gpu_point_cloud, raycasting as gpu_raycasting, registration as gpu_registration,
    tsdf as gpu_tsdf,
};
pub use mesh::TriangleMesh;
pub use odometry::{compute_rgbd_odometry, OdometryMethod, OdometryResult};
pub use raycasting::{
    cast_ray_mesh, cast_rays_mesh, closest_point_on_mesh, closest_points_on_mesh,
    mesh_to_mesh_distance, point_inside_mesh, Ray, RayHit,
};
pub use spatial::{KDTree, Octree, VoxelGrid};
pub use tsdf::{CameraIntrinsics, TSDFVolume, Triangle, VoxelBlock};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Async task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Core error: {0}")]
    CoreError(#[from] cv_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
