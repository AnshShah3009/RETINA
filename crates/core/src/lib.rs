#![allow(missing_docs)]
//! Core computer vision types and traits
//!
//! This crate provides fundamental types and abstractions for computer vision operations,
//! including point clouds, tensors, keypoints, and descriptors.

pub mod buffer_handle;
pub mod descriptor;
pub mod float;
pub mod frames;
pub mod geometry;
pub mod image;
pub mod keypoint;
pub mod nalgebra_adapters;
pub mod nalgebra_wrapper;
pub mod point_cloud;
pub mod robust;
pub mod robust_loss;
pub mod runtime;
pub mod storage;
pub mod tensor;
pub mod vector;

pub use buffer_handle::*;
pub use descriptor::*;
pub use float::*;
pub use frames::*;
pub use geometry::*;
pub use image::*;
pub use keypoint::*;
pub use nalgebra_adapters::*;
pub use nalgebra_wrapper::*;
pub use point_cloud::*;
pub use robust::*;
pub use robust_loss::*;
pub use runtime::*;
pub use storage::*;
pub use tensor::*;
pub use vector::*;

/// Error types for core operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Runtime error with message
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Invalid input with message
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Memory allocation or access error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Tensor or array dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// I/O error from file operations
    #[error("I/O error: {0}")]
    IoError(String),

    /// Hardware/Device error
    #[error("Device error: {0}")]
    DeviceError(String),

    /// GPU computation error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Parse/Format error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Algorithm failure (convergence, optimization, etc.)
    #[error("Algorithm failed: {0}")]
    AlgorithmError(String),

    /// Generic error with custom message
    #[error("{0}")]
    Other(String),
}

/// Result type for core operations
pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err.to_string())
    }
}
