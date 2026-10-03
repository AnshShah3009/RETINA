//! # cv-eval
//!
//! Evaluation metrics for visual localization, SfM and SLAM.
//!
//! This crate provides the measurement layer for reconstruction-style
//! pipelines: trajectory error (ATE/RPE), reconstruction quality and image
//! retrieval metrics. Everything is pure Rust, works on synthetic data, and is
//! free of panics on empty input.
//!
//! No metric reports a *perfect* score for input it never measured. Metrics
//! where smaller is better (ATE, RPE, camera-centre RMSE, reprojection RMSE,
//! Chamfer distance) return `NaN` when there is nothing to measure, and metrics
//! where larger is better (registration rate, F-score, recall/precision/mAP)
//! return `0.0`, their own worst value. Both directions keep "measured nothing"
//! out of the top of a ranking; see `ErrorStats::from_errors` and the module
//! documentation of [`reconstruction`].
//!
//! [`Trajectory::ate`] additionally carries [`AteResult::is_valid`], which is
//! `false` when the input itself was malformed (a non-finite camera centre).
//!
//! ## Modules
//!
//! - [`trajectory`]: [`Trajectory`] with absolute trajectory error
//!   ([`Trajectory::ate`], with optional SE(3)/Sim(3) Umeyama alignment) and
//!   relative pose error ([`Trajectory::rpe`]).
//! - [`reconstruction`]: registration rate, reprojection RMSE, scale-free RMSE,
//!   symmetric Chamfer distance and F-score.
//! - [`retrieval`]: recall@K, precision@K and mean average precision.
//!
//! # Example
//!
//! ```
//! use cv_eval::{Alignment, Trajectory};
//! use nalgebra::{UnitQuaternion, Vector3};
//!
//! let positions = [
//!     Vector3::new(0.0, 0.0, 0.0),
//!     Vector3::new(1.0, 0.0, 0.0),
//!     Vector3::new(0.0, 1.0, 0.0),
//!     Vector3::new(0.0, 0.0, 1.0),
//! ];
//! let quaternions = [UnitQuaternion::identity(); 4];
//! let trajectory = Trajectory::from_positions_and_quaternions(&positions, &quaternions);
//!
//! // A trajectory measured against itself is exact.
//! let ate = trajectory.ate(&trajectory, Alignment::Sim3);
//! assert!(ate.rmse < 1e-9);
//! ```

#![forbid(unsafe_code)]

pub mod reconstruction;
pub mod retrieval;
pub mod trajectory;

pub use reconstruction::{
    chamfer_distance, f_score, registration_rate, reprojection_rmse, rmse_over_extent,
};
pub use retrieval::{mean_average_precision, precision_at_k, recall_at_k};
pub use trajectory::{
    Alignment, AteResult, ErrorStats, RpeResult, SimilarityTransform, Trajectory,
};
