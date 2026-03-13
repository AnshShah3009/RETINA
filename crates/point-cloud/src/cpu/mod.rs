//! CPU-based point cloud operations
//!
//! This module re-exports point cloud functionality from cv-scientific,
//! providing a unified interface for CPU-based operations including:
//! - Filtering (voxel downsampling, outlier removal)
//! - Normal estimation and orientation
//! - Segmentation and clustering

pub mod filtering;
pub mod normals;
pub mod segmentation;

// Re-export main functionality
pub use filtering::*;
pub use normals::*;
pub use segmentation::*;
