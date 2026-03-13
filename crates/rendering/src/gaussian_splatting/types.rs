//! Core types for Gaussian Splatting
//!
//! This module provides the fundamental data structures for 3D Gaussian Splatting,
//! a novel approach to novel view synthesis that represents scenes as a collection
//! of 3D Gaussians instead of traditional meshes or point clouds.
//!
//! ## Overview
//!
//! Gaussian Splatting works by:
//! 1. Representing the scene as a set of 3D Gaussians (ellipsoids)
//! 2. Each Gaussian has position, scale, rotation, opacity, and color (via Spherical Harmonics)
//! 3. Rendering is done by projecting Gaussians to 2D and blending them
//! 4. The system is differentiable, allowing optimization from image observations
//!
//! ## Key Types
//!
//! - [`Gaussian`]: A single 3D Gaussian with position, scale, rotation, opacity, and SH coefficients
//! - [`GaussianCloud`]: A collection of Gaussians representing a scene
//! - [`SphericalHarmonics`]: View-dependent color representation using SH basis functions
//! - [`ProjectedGaussian`]: A Gaussian projected into 2D screen space for rendering
//!
//! ## Example
//!
//! ```rust
//! use cv_rendering::gaussian_splatting::{Gaussian, GaussianCloud};
//! use nalgebra::{Point3, Vector3, Vector4};
//!
//! // Create a single Gaussian
//! let gaussian = Gaussian::new(
//!     Point3::new(0.0, 0.0, 0.0),  // Position
//!     Vector3::new(0.1, 0.1, 0.1), // Scale
//!     Vector4::new(0.0, 0.0, 0.0, 1.0), // Rotation (quaternion)
//!     Vector3::new(0.8, 0.5, 0.3),  // Color (RGB)
//! );
//!
//! // Create a cloud and add the Gaussian
//! let mut cloud = GaussianCloud::new();
//! cloud.push(gaussian);
//!
//! println!("Cloud has {} gaussians", cloud.num_gaussians());
//! ```

use nalgebra::{Matrix3, Matrix3x4, Point3, Vector2, Vector3, Vector4};

/// Spherical Harmonics (SH) coefficients for view-dependent color
///
/// SH provides a compact representation of view-dependent lighting effects.
/// The color of a Gaussian depends on the viewing direction through these coefficients.
///
/// Degree 0 (DC): 1 coefficient per channel (constant color)
/// Degree 1: 4 coefficients per channel
/// Degree 2: 9 coefficients per channel
/// Degree n: (n+1)² coefficients per channel
///
/// Higher degrees allow more complex view-dependent effects like specular reflections.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalHarmonics {
    /// The SH coefficients stored as [R0, G0, B0, R1, G1, B1, ...]
    pub coeffs: Vec<f32>,
    /// The SH degree (0, 1, 2, or 3 typically)
    pub degree: usize,
}

impl SphericalHarmonics {
    pub fn new(degree: usize) -> Self {
        let num_coeffs = (degree + 1) * (degree + 1);
        Self {
            coeffs: vec![0.0; num_coeffs],
            degree,
        }
    }

    pub fn from_dc(dc: Vector3<f32>) -> Self {
        // For RGB DC coefficients, we need at least 3 values
        // Using degree 0 gives us (0+1)^2 = 1 coefficient which is not enough
        // So we store DC as first 3 coefficients manually
        Self {
            coeffs: vec![dc.x, dc.y, dc.z],
            degree: 0,
        }
    }

    pub fn dc(&self) -> Vector3<f32> {
        Vector3::new(self.coeffs[0], self.coeffs[1], self.coeffs[2])
    }

    pub fn eval(&self, view_dir: Vector3<f32>) -> Vector3<f32> {
        if self.degree == 0 {
            return self.dc();
        }

        let x = view_dir.x;
        let y = view_dir.y;
        let z = view_dir.z;

        let mut result = Vector3::zeros();

        result.x += self.coeffs[0];
        result.y += self.coeffs[1];
        result.z += self.coeffs[2];

        if self.degree >= 1 {
            result.x += self.coeffs[3] * y;
            result.y += self.coeffs[4] * z;
            result.z += self.coeffs[5] * x;
            result.x += self.coeffs[6] * x * y;
            result.y += self.coeffs[7] * y * z;
            result.z += self.coeffs[8] * z * x;
            result.x += self.coeffs[9] * (3.0 * z * z - 1.0);
            result.y += self.coeffs[10] * x * z;
            result.z += self.coeffs[11] * (x * x - y * y);
        }

        result
    }
}

#[derive(Clone, Debug)]
pub struct Gaussian {
    pub position: Point3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: Vector4<f32>,
    pub opacity: f32,
    pub spherical_harmonics: SphericalHarmonics,
    pub features: Vector3<f32>,
}

impl Gaussian {
    pub fn new(
        position: Point3<f32>,
        scale: Vector3<f32>,
        rotation: Vector4<f32>,
        color: Vector3<f32>,
    ) -> Self {
        let opacity = 0.5;
        Self {
            position,
            scale: scale.map(|s| s.max(0.0001)),
            rotation: rotation.normalize(),
            opacity,
            spherical_harmonics: SphericalHarmonics::from_dc(color),
            features: Vector3::zeros(),
        }
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn with_features(mut self, features: Vector3<f32>) -> Self {
        self.features = features;
        self
    }

    pub fn scale_matrix(&self) -> Matrix3<f32> {
        Matrix3::from_diagonal(&self.scale)
    }

    pub fn rotation_matrix(&self) -> Matrix3<f32> {
        let q = &self.rotation;
        let x = q[0];
        let y = q[1];
        let z = q[2];
        let w = q[3];

        Matrix3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    }

    pub fn covariance(&self) -> Matrix3<f32> {
        let r = self.rotation_matrix();
        let s = self.scale_matrix();
        r * s * s * r.transpose()
    }

    pub fn inverse_covariance(&self) -> Matrix3<f32> {
        self.covariance().try_inverse().unwrap_or(Matrix3::zeros())
    }

    pub fn project(&self, view_matrix: &Matrix3x4<f32>, focal_length: f32) -> ProjectedGaussian {
        let pos = self.position.coords;
        let view_pos = Vector3::new(
            view_matrix[(0, 0)] * pos.x
                + view_matrix[(0, 1)] * pos.y
                + view_matrix[(0, 2)] * pos.z
                + view_matrix[(0, 3)],
            view_matrix[(1, 0)] * pos.x
                + view_matrix[(1, 1)] * pos.y
                + view_matrix[(1, 2)] * pos.z
                + view_matrix[(1, 3)],
            view_matrix[(2, 0)] * pos.x
                + view_matrix[(2, 1)] * pos.y
                + view_matrix[(2, 2)] * pos.z
                + view_matrix[(2, 3)],
        );
        let depth = view_pos[2];

        if depth <= 0.0 {
            return ProjectedGaussian::invalid();
        }

        let px = view_pos[0] / depth * focal_length;
        let py = view_pos[1] / depth * focal_length;

        let cov = self.covariance();

        let jw = Matrix3::new(
            focal_length / depth,
            0.0,
            -focal_length * view_pos[0] / (depth * depth),
            0.0,
            focal_length / depth,
            -focal_length * view_pos[1] / (depth * depth),
            0.0,
            0.0,
            1.0 / depth,
        );

        let screen_cov = jw * cov * jw.transpose();

        ProjectedGaussian {
            center: Vector2::new(px, py),
            covariance: screen_cov,
            depth,
            opacity: self.opacity,
            color: self.spherical_harmonics.dc(),
            rotation: self.rotation,
            scale: self.scale,
            features: self.features,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProjectedGaussian {
    pub center: Vector2<f32>,
    pub covariance: Matrix3<f32>,
    pub depth: f32,
    pub opacity: f32,
    pub color: Vector3<f32>,
    pub rotation: Vector4<f32>,
    pub scale: Vector3<f32>,
    pub features: Vector3<f32>,
}

impl ProjectedGaussian {
    pub fn invalid() -> Self {
        Self {
            center: Vector2::new(f32::MAX, f32::MAX),
            covariance: Matrix3::zeros(),
            depth: f32::MAX,
            opacity: 0.0,
            color: Vector3::zeros(),
            rotation: Vector4::new(0.0, 0.0, 0.0, 1.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            features: Vector3::zeros(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.depth > 0.0 && self.depth < f32::MAX && self.opacity > 0.001
    }

    pub fn inv_cov_2d(&self) -> Matrix3<f32> {
        let cov_2d = Matrix3::new(
            self.covariance[(0, 0)],
            self.covariance[(0, 1)],
            0.0,
            self.covariance[(1, 0)],
            self.covariance[(1, 1)],
            0.0,
            0.0,
            0.0,
            1.0,
        );
        cov_2d.try_inverse().unwrap_or(Matrix3::zeros())
    }
}

pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian>,
    pub active_indices: Vec<usize>,
}

impl GaussianCloud {
    pub fn new() -> Self {
        Self {
            gaussians: Vec::new(),
            active_indices: Vec::new(),
        }
    }

    pub fn from_gaussians(gaussians: Vec<Gaussian>) -> Self {
        let active_indices = (0..gaussians.len()).collect();
        Self {
            gaussians,
            active_indices,
        }
    }

    pub fn push(&mut self, gaussian: Gaussian) -> usize {
        let idx = self.gaussians.len();
        self.gaussians.push(gaussian);
        self.active_indices.push(idx);
        idx
    }

    pub fn remove(&mut self, idx: usize) {
        self.gaussians.swap_remove(idx);
        if let Some(pos) = self.active_indices.iter().position(|&i| i == idx) {
            self.active_indices.swap_remove(pos);
        }
        for i in &mut self.active_indices {
            if *i == idx {
                *i = self.gaussians.len();
            }
        }
    }

    pub fn active(&self) -> &[Gaussian] {
        &self.gaussians
    }

    pub fn num_gaussians(&self) -> usize {
        self.gaussians.len()
    }

    pub fn num_active(&self) -> usize {
        self.active_indices.len()
    }

    pub fn clear_inactive(&mut self) {
        self.gaussians.retain(|g| g.opacity > 0.001);
        self.active_indices = (0..self.gaussians.len()).collect();
    }

    pub fn filter_by_opacity(&mut self, threshold: f32) {
        self.gaussians.retain(|g| g.opacity > threshold);
        self.active_indices = (0..self.gaussians.len()).collect();
    }

    pub fn normalize_opacity(&mut self) {
        for g in &mut self.gaussians {
            g.opacity = 1.0 - (-g.opacity).exp();
        }
    }
}

impl Default for GaussianCloud {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_harmonics_new() {
        let sh = SphericalHarmonics::new(0);
        assert_eq!(sh.degree, 0);
        assert_eq!(sh.coeffs.len(), 1);
        assert_eq!(sh.coeffs[0], 0.0);

        let sh = SphericalHarmonics::new(1);
        assert_eq!(sh.degree, 1);
        assert_eq!(sh.coeffs.len(), 4);
    }

    #[test]
    fn test_spherical_harmonics_from_dc() {
        let dc = Vector3::new(0.5, 0.6, 0.7);
        let sh = SphericalHarmonics::from_dc(dc);
        assert_eq!(sh.degree, 0);
        assert_eq!(sh.dc(), dc);
    }

    #[test]
    fn test_gaussian_new() {
        let position = Point3::new(1.0, 2.0, 3.0);
        let scale = Vector3::new(0.1, 0.2, 0.3);
        let rotation = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let color = Vector3::new(0.8, 0.5, 0.3);

        let gaussian = Gaussian::new(position, scale, rotation, color);

        assert_eq!(gaussian.position, position);
        assert_eq!(gaussian.scale, scale);
        assert!((gaussian.rotation.norm() - 1.0).abs() < 1e-6);
        assert_eq!(gaussian.opacity, 0.5);
        assert_eq!(gaussian.spherical_harmonics.dc(), color);
    }

    #[test]
    fn test_gaussian_with_opacity() {
        let gaussian = Gaussian::new(
            Point3::origin(),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.8);

        assert!((gaussian.opacity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_scale_matrix() {
        let scale = Vector3::new(1.0, 2.0, 3.0);
        let gaussian = Gaussian::new(
            Point3::origin(),
            scale,
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );

        let scale_matrix = gaussian.scale_matrix();
        assert_eq!(scale_matrix[(0, 0)], 1.0);
        assert_eq!(scale_matrix[(1, 1)], 2.0);
        assert_eq!(scale_matrix[(2, 2)], 3.0);
    }

    #[test]
    fn test_gaussian_rotation_matrix() {
        let gaussian = Gaussian::new(
            Point3::origin(),
            Vector3::new(1.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );

        let rot_mat = gaussian.rotation_matrix();
        // Identity rotation should give identity matrix
        assert!((rot_mat[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((rot_mat[(1, 1)] - 1.0).abs() < 1e-6);
        assert!((rot_mat[(2, 2)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_covariance() {
        let gaussian = Gaussian::new(
            Point3::origin(),
            Vector3::new(1.0, 2.0, 3.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );

        let cov = gaussian.covariance();
        // For identity rotation, covariance should be scale^2 on diagonal
        assert!((cov[(0, 0)] - 1.0).abs() < 1e-5);
        assert!((cov[(1, 1)] - 4.0).abs() < 1e-5);
        assert!((cov[(2, 2)] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_projected_gaussian_invalid() {
        let pg = ProjectedGaussian::invalid();
        assert!(!pg.is_valid());
        assert_eq!(pg.depth, f32::MAX);
        assert_eq!(pg.opacity, 0.0);
    }

    #[test]
    fn test_gaussian_cloud_new() {
        let cloud = GaussianCloud::new();
        assert_eq!(cloud.num_gaussians(), 0);
        assert_eq!(cloud.num_active(), 0);
    }

    #[test]
    fn test_gaussian_cloud_push() {
        let mut cloud = GaussianCloud::new();
        let gaussian = Gaussian::new(
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );

        let idx = cloud.push(gaussian);
        assert_eq!(idx, 0);
        assert_eq!(cloud.num_gaussians(), 1);
        assert_eq!(cloud.num_active(), 1);
    }

    #[test]
    fn test_gaussian_cloud_remove() {
        let mut cloud = GaussianCloud::new();

        let g1 = Gaussian::new(
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );
        let g2 = Gaussian::new(
            Point3::new(2.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        );

        cloud.push(g1);
        cloud.push(g2);
        assert_eq!(cloud.num_gaussians(), 2);

        cloud.remove(0);
        assert_eq!(cloud.num_gaussians(), 1);
    }

    #[test]
    fn test_gaussian_cloud_filter_by_opacity() {
        let mut cloud = GaussianCloud::new();

        let g1 = Gaussian::new(
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.1);

        let g2 = Gaussian::new(
            Point3::new(2.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.9);

        cloud.push(g1);
        cloud.push(g2);

        cloud.filter_by_opacity(0.5);
        assert_eq!(cloud.num_gaussians(), 1);
    }
}
