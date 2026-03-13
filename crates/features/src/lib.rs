#![warn(missing_docs)]

//! Feature detection, extraction, and matching algorithms.
//!
//! This crate provides implementations of common computer vision feature detection and
//! descriptor extraction algorithms, including:
//!
//! - **AKAZE**: Accelerated-KAZE multi-scale feature detector with M-SURF descriptors
//! - **BRIEF**: Binary Robust Independent Elementary Features descriptor
//! - **FAST**: Features from Accelerated Segment Test corner detection
//! - **GFTT**: Good Features to Track corner detection (Harris corner detector)
//! - **ORB**: Oriented FAST and Rotated BRIEF features
//! - **SIFT**: Scale-Invariant Feature Transform detector and descriptor
//! - **FLANN**: Fast Library for Approximate Nearest Neighbors matching
//! - **Matcher**: Feature descriptor matching with various strategies
//!
//! # Basic Usage
//!
//! ```no_run
//! # use cv_features::*;
//! # use image::GrayImage;
//! let image = GrayImage::new(640, 480);
//!
//! // Detect ORB features
//! let orb = orb::Orb::new().with_n_features(500);
//! let keypoints = orb.detect(&image);
//! let descriptors = orb.extract(&image, &keypoints);
//!
//! // Match features
//! let matcher = matcher::Matcher::new(matcher::MatchType::BruteForceHamming);
//! let matches = matcher.match_descriptors(&descriptors, &descriptors);
//! ```

/// AKAZE (Accelerated-KAZE) feature detector and descriptor
pub mod akaze;
/// ArUco marker detection, generation, and pose estimation (tensor-based API)
pub mod aruco;
/// BRIEF (Binary Robust Independent Elementary Features) descriptor
pub mod brief;
/// Feature descriptor types and trait implementations
pub mod descriptor;
/// FAST (Features from Accelerated Segment Test) corner detection
pub mod fast;
/// FLANN (Fast Library for Approximate Nearest Neighbors) matching
pub mod flann;
/// GFTT (Good Features to Track) corner detection
pub mod gftt;
/// Harris corner detection algorithm
pub mod harris;
/// Histogram of Oriented Gradients (HOG) feature extraction
pub mod hog;
/// LBD (Line Band Descriptor) for line feature matching
pub mod lbd;
/// Line feature matching and tracking
pub mod line_matcher;
/// Marker detection and tracking
pub mod markers;
/// Feature descriptor matching with various strategies
pub mod matcher;
/// ORB (Oriented FAST and Rotated BRIEF) detector and descriptor
pub mod orb;
/// RANSAC-based model estimation and outlier rejection
pub mod ransac;
/// SIFT (Scale-Invariant Feature Transform) detector and descriptor
pub mod sift;

pub use akaze::*;
pub use brief::*;
pub use cv_imgproc::hough;
pub use descriptor::*;
pub use fast::*;
pub use flann::*;
pub use gftt::*;
pub use harris::*;
pub use hog::*;
pub use lbd::*;
pub use line_matcher::*;
pub use markers::*;
pub use matcher::*;
pub use orb::*;
pub use ransac::*;
pub use sift::*;

pub use cv_core::{Error, KeyPoint, KeyPoints, Result};

/// Backward compatibility alias for deprecated custom error type
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Error instead. This type exists only for backward compatibility."
)]
pub type FeatureError = cv_core::Error;

/// Deprecated Result type alias - use cv_core::Result instead
#[deprecated(
    since = "0.1.0",
    note = "Use cv_core::Result instead. This type alias exists only for backward compatibility."
)]
pub type FeatureResult<T> = cv_core::Result<T>;
