//! CPU-based point cloud filtering operations
//!
//! Filtering operations include:
//! - Voxel downsampling
//! - Statistical outlier removal
//! - Radius outlier removal

pub use cv_scientific::point_cloud::{
    remove_radius_outliers, remove_statistical_outliers, voxel_down_sample,
};
