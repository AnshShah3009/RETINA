//! Gaussian Splatting for Novel View Synthesis
//!
//! This module provides a complete implementation of 3D Gaussian Splatting,
//! a state-of-the-art technique for real-time novel view synthesis.
//!
//! ## Overview
//!
//! Gaussian Splatting represents scenes as a collection of 3D Gaussians
//! (ellipsoids) instead of traditional meshes or point clouds. Each Gaussian
//! has:
//! - Position (3D center)
//! - Scale (covariance matrix)
//! - Rotation (orientation)
//! - Opacity (alpha value)
//! - Color (via Spherical Harmonics for view-dependent effects)
//!
//! ## Key Components
//!
//! - [`types::Gaussian`]: A single 3D Gaussian primitive
//! - [`types::GaussianCloud`]: Collection of Gaussians representing a scene
//! - [`types::SphericalHarmonics`]: View-dependent color representation
//! - [`rasterize::GaussianRasterizer`]: Forward rendering pipeline
//! - [`differentiable::DifferentiableRasterizer`]: Differentiable rendering for training
//! - [`optimize::GaussianOptimizer`]: Training optimizer with densification/pruning
//!
//! ## Example
//!
//! ```rust
//! use cv_rendering::gaussian_splatting::{Gaussian, GaussianCloud, GaussianRasterizer, Camera};
//! use nalgebra::{Point3, Vector3, Vector4};
//!
//! // Create a scene with Gaussians
//! let mut cloud = GaussianCloud::new();
//! cloud.push(Gaussian::new(
//!     Point3::new(0.0, 0.0, 0.0),
//!     Vector3::new(0.1, 0.1, 0.1),
//!     Vector4::new(0.0, 0.0, 0.0, 1.0),
//!     Vector3::new(0.8, 0.5, 0.3),
//! ));
//!
//! // Create camera and render
//! let camera = Camera::new(
//!     Point3::new(0.0, 0.0, 5.0),
//!     Vector4::new(0.0, 0.0, 0.0, 1.0),
//!     1000.0, 800, 600,
//! );
//! let rasterizer = GaussianRasterizer::new(camera, 16, 16);
//! let result = rasterizer.rasterize(&cloud);
//! ```
//!
//! ## References
//!
//! - [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
//! - [Original paper](https://arxiv.org/abs/2308.04079)

pub mod differentiable;
pub mod io;
pub mod optimize;
pub mod rasterize;
pub mod types;

pub use differentiable::DifferentiableRasterizer;
pub use io::{read_ply_gaussian_cloud, write_ply_gaussian_cloud};
pub use optimize::{DensificationConfig, GaussianOptimizer, GaussianTrainer, TrainingConfig};
pub use rasterize::{Camera, GaussianRasterizer, RasterizationResult};
pub use types::{Gaussian, GaussianCloud, ProjectedGaussian, SphericalHarmonics};
