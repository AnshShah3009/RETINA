//! Point cloud registration algorithms
//!
//! This crate provides comprehensive 3D point cloud registration algorithms
//! for aligning and aligning multiple point clouds.
//!
//! ## Registration Methods
//!
//! ### ICP (Iterative Closest Point)
//! - [`registration_icp_point_to_plane`]: Standard point-to-plane ICP
//! - [`registration_multi_scale_icp`]: Multi-resolution ICP for large misalignments
//!
//! ### Global Registration
//! - [`registration_fgr_based_on_feature_matching`]: Fast Global Registration using FPFH features
//! - [`registration_ransac_based_on_feature_matching`]: RANSAC-based registration
//!
//! ### Color-based Registration
//! - [`registration_colored_icp`]: ICP variant using both geometry and color
//!
//! ### Robust Registration
//! - [`registration_gnc`]: Graduated Non-Convexity for handling outliers
//!
//! ## Example
//!
//! ```rust
//! use cv_registration::{registration_icp_point_to_plane, ICPResult};
//! use cv_core::PointCloud;
//! use nalgebra::{Matrix4, Point3};
//!
//! // Create source and target point clouds
//! let source = PointCloud::new(vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(1.0, 0.0, 0.0),
//!     Point3::new(0.0, 1.0, 0.0),
//! ]);
//!
//! let target = PointCloud::new(vec![
//!     Point3::new(0.1, 0.1, 0.0),
//!     Point3::new(1.1, 0.1, 0.0),
//!     Point3::new(0.1, 1.1, 0.0),
//! ]);
//!
//! // Run ICP registration
//! let result: Option<ICPResult> = registration_icp_point_to_plane(
//!     &source,
//!     &target,
//!     0.5,           // max correspondence distance
//!     &Matrix4::identity(), // initial transformation
//!     50,            // max iterations
//! );
//!
//! if let Some(icp) = result {
//!     println!("Fitness: {}", icp.fitness);
//!     println!("RMSE: {}", icp.inlier_rmse);
//! }
//! ```
//!
//! ## Output Types
//!
//! - [`ICPResult`]: Result of ICP registration
//! - [`ColoredICPResult`]: Result of colored ICP
//! - [`GlobalRegistrationResult`]: Result of global registration
//! - [`GNCResult`]: Result of GNC-based robust registration
//!
//! ## Evaluation
//!
//! - [`evaluate_registration`]: Evaluate registration quality
//! - [`get_information_matrix_from_point_clouds`]: Compute information matrix for uncertainty

/// Use cv_core::Error as the standard error type
pub use cv_core::Error as RegistrationError;
pub use cv_core::Result;

pub mod registration;

pub use registration::{
    evaluate_registration, get_information_matrix_from_point_clouds,
    registration_icp_point_to_plane, registration_multi_scale_icp, ICPResult,
};

pub use cv_core::RobustLoss;
pub use registration::colored::{registration_colored_icp, ColoredICPResult};
pub use registration::global::{
    registration_fgr_based_on_feature_matching, registration_ransac_based_on_feature_matching,
    FPFHFeature, FastGlobalRegistrationOption, GlobalRegistrationResult,
};
pub use registration::gnc::{registration_gnc, GNCOptimizer, GNCResult};
