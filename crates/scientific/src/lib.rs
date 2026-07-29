//! Scientific Computing — meta-crate re-exporting from cv-math, cv-geometry2d, cv-signal, cv-pointcloud
//!
//! This crate is a backward-compatible re-export layer. New code should depend on
//! the individual sub-crates directly:
//! - [`cv-math`]: Numerical math (stats, special, linalg, sparse, interpolate, integrate, spatial, geometry, jit)
//! - [`cv-geometry2d`]: 2D computational geometry
//! - [`cv-signal`]: Signal processing (FFT, filters, windows, spectral analysis, wavelets)
//! - [`cv-pointcloud`]: Point cloud processing (normals, downsampling, RANSAC, FPFH)

pub mod fft { pub use cv_signal::fft::*; }
pub mod geometry { pub use cv_math::geometry::*; }
pub mod geometry2d { pub use cv_geometry2d::geometry2d::*; }
pub mod integrate { pub use cv_math::integrate::*; }
pub mod interpolate { pub use cv_math::interpolate::*; }
pub mod jit { pub use cv_math::jit::*; }
pub mod linalg { pub use cv_math::linalg::*; }
pub mod point_cloud { pub use cv_pointcloud::point_cloud::*; }
pub mod signal { pub use cv_signal::signal::*; }
pub mod sparse { pub use cv_math::sparse::*; }
pub mod spatial { pub use cv_math::spatial::*; }
pub mod special { pub use cv_math::special::*; }
pub mod stats { pub use cv_math::stats::*; }

pub use cv_math::Error;
pub use cv_math::Result;

pub use cv_math::{mean, std, lerp, Interp1d};

pub use geometry::*;
pub use integrate::*;
pub use jit::*;
pub use point_cloud::*;
pub use special::*;
