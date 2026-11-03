//! Statistical outlier removal.
//!
//! The implementation lives in `cv-pointcloud`, shared with its `f32`
//! `remove_statistical_outliers`; see
//! [`cv_pointcloud::filters::statistical_outlier_removal`] for the contract,
//! including how non-finite points are treated.

pub use cv_pointcloud::filters::statistical_outlier_removal;
