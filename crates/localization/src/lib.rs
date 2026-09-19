//! End-to-end visual localization.
//!
//! This crate answers the question a localization user actually asks — *given
//! this query image, where was it taken?* — by composing the primitives that
//! already live in the workspace:
//!
//! * **Retrieval** ([`Database`]) keeps a set of database images with their
//!   keypoints, descriptors and (world-frame) camera poses, plus the 3D
//!   [`Landmark`]s those images observe. When a BoW [`Vocabulary`] is supplied
//!   the images are indexed in a [`cv_features::retrieval::BowDatabase`];
//!   otherwise candidate images are ranked by raw descriptor-match counts.
//! * **Matching + PnP** ([`Localizer`]) retrieves the top candidates, matches
//!   the query descriptors to the best candidate with a ratio test, lifts the
//!   surviving matches to 2D–3D correspondences through the candidate's
//!   landmark indices, and runs a PnP + RANSAC solver
//!   ([`cv_calib3d::pnp::solve_pnp_ransac`]) followed by a refinement pass.
//! * **Evaluation** ([`evaluate_localization`]) aggregates a batch of
//!   [`LocalizationResult`]s against known ground-truth poses into translation
//!   and rotation error statistics plus a success rate.
//!
//! # Example
//!
//! ```
//! use cv_localization::{Database, Localizer, LocalizerConfig};
//! use cv_core::{CameraIntrinsics, Descriptors};
//!
//! // An empty database is a valid (if useless) one: nothing panics.
//! let database = Database::new(None);
//! let localizer = Localizer::new(&database, LocalizerConfig::default());
//! let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
//!
//! let kps: Vec<cv_core::KeyPoint> = Vec::new();
//! assert!(localizer.localize(&kps, &Descriptors::new(), &intrinsics).is_none());
//! ```
//!
//! See the [`synthetic`] module (feature `synthetic`) and the
//! `synthetic_localization` example for a runnable end-to-end demonstration.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod database;
mod evaluate;
mod localizer;

#[cfg(any(test, feature = "synthetic"))]
pub mod synthetic;

#[cfg(feature = "synthetic")]
pub mod benchmark;

pub use database::{database_image_from_colmap, Database, DatabaseImage, DescriptorRef, Landmark};
pub use evaluate::{
    evaluate_localization, rotation_error_degrees, translation_error, LocalizationStats,
};
pub use localizer::{LocalizationResult, Localizer, LocalizerConfig};

// Re-exported so users can name the retrieval types without depending on
// `cv-features` directly.
pub use cv_features::retrieval::Vocabulary;

#[cfg(test)]
mod tests;
