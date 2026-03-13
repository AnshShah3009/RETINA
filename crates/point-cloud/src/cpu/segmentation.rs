//! CPU-based point cloud segmentation and clustering
//!
//! Segmentation and clustering operations include:
//! - Plane segmentation using RANSAC
//! - DBSCAN clustering
//! - FPFH feature computation

pub use cv_scientific::point_cloud::{cluster_dbscan, compute_fpfh_feature, segment_plane};
