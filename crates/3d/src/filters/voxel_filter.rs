//! Voxel downsampling.
//!
//! The implementation lives in `cv-pointcloud`, shared with its `f32`
//! `voxel_down_sample` and with `cv_3d::gpu::voxel_downsample`; see
//! [`cv_pointcloud::filters::voxel_downsample`] for the contract, including
//! how invalid sizes and non-finite points are treated.

pub use cv_pointcloud::filters::{VoxelDownsampleResult, voxel_downsample};
