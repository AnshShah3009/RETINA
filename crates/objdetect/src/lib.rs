#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Object and marker detection, after OpenCV's `objdetect` module.
//!
//! - **ArUco** markers: dictionaries, detection (OpenCV's `detectMarkers`
//!   pipeline), drawing and pose estimation
//! - **ChArUco** boards: interpolated chessboard corners and board pose
//! - **AprilTag** and image-based board helpers (`markers`)
//! - **HOG** descriptors and **Haar cascades** (Viola-Jones)
//!
//! These lived in `cv-features` before; they moved here because marker pose
//! needs `cv-calib3d`, and feature detection should not.

/// ArUco marker detection, generation, and pose estimation (tensor-based API)
pub mod aruco;
/// OpenCV's predefined ArUco dictionaries as packed bytes
pub mod aruco_tables;
/// ChArUco board detection — fiducial marker board with chessboard corners
pub mod charuco;
/// Haar cascade classifier for object detection (Viola-Jones)
pub mod haar_cascade;
/// Histogram of Oriented Gradients (HOG) feature extraction
pub mod hog;
/// Marker detection and tracking
pub mod markers;

pub use charuco::*;
pub use haar_cascade::*;
pub use hog::*;
pub use markers::*;

// `charuco` and `markers` both define a `CharucoBoard`. The crate-root name is
// the tensor-API board that `CharucoDetector::new` takes; the image-API board
// from `create_charuco_board` stays reachable as `markers::CharucoBoard`.
pub use charuco::CharucoBoard;

pub use cv_core::{Error, Result};
