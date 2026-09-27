#![forbid(unsafe_code)]
//! Structure from Motion (SfM)
//!
//! This crate provides algorithms for 3D reconstruction from multiple 2D images.
//! It implements complete photogrammetry pipelines including:
//!
//! ## Core Components
//!
//! - [`bundle_adjustment`]: Non-linear optimization for refining camera poses and 3D points
//! - [`triangulation`]: Triangulating 3D points from multiple views
//! - [`mapper`]: Incremental Structure-from-Motion — builds cameras and 3D points
//!   from images and intrinsics alone, with no ground-truth poses
//!
//! ## Bundle Adjustment
//!
//! The bundle adjustment module provides:
//! - Levenberg-Marquardt optimization
//! - Robust kernel support
//! - Sparse Jacobian computation for efficiency
//! - Configurable convergence criteria
//!
//! ## Example
//!
//! ```rust,ignore
//! use cv_sfm::bundle_adjustment;
//!
//! // See bundle_adjustment module for full API
//! ```

pub mod bundle_adjustment;
pub mod mapper;
pub mod triangulation;

pub use mapper::{
    map_views, MapperConfig, Mapping, MappingReport, PairSelection, Reconstruction,
    RegistrationFailure, View, ViewOutcome,
};
pub use triangulation::*;

pub trait BundleAdjustment {
    fn optimize(&mut self);
}
