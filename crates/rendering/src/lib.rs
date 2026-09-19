//! Rendering algorithms for computer vision
//!
//! This crate provides rendering and novel view synthesis algorithms:
//! - 3D Gaussian Splatting
//! - Differentiable rasterization
//! - Future: NeRF, neural rendering methods

pub mod gaussian_splatting;

pub use gaussian_splatting::{
    read_ply_gaussian_cloud, write_ply_gaussian_cloud, Camera, DensificationConfig,
    DifferentiableRasterizer, Gaussian, GaussianCloud, GaussianOptimizer, GaussianRasterizer,
    GaussianTrainer, ProjectedGaussian, RasterizationResult, SphericalHarmonics, TrainingConfig,
};
