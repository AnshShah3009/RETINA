//! Frame and Coordinate Convention Definitions
//!
//! This module defines coordinate system conventions for 3D computer vision.
//! Different systems use different conventions:
//!
//! ## Coordinate Systems
//!
//! - **Right-handed (RH)**: +X right, +Y down, +Z forward (OpenCV default)
//! - **Left-handed (LH)**: +X right, +Y up, +Z into screen (WebGPU)
//!
//! ## Camera Conventions
//!
//! | Convention | Origin | +Z direction | +Y direction | Used by |
//! |------------|--------|--------------|--------------|---------|
//! | OpenCV     | at camera center | forward (+Z) | down (-Y) | OpenCV, ROS |
//! | OpenGL     | at camera center | backward (-Z) | up (+Y) | OpenGL, Unity |
//! | COLMAP     | at camera center | forward (+Z) | down (-Y) | COLMAP, OpenMVG |
//! | WebGPU     | at camera center | forward (+Z) | up (+Y) | WebGPU, Three.js |
//!
//! ## Rig Orientation
//!
//! For multi-camera rigs, the rig origin can be:
//! - **World origin**: First camera position
//! - **Body center**: IMU/body center
//! - **Arbitrary**: User-defined point

use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// Coordinate system handedness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    RightHanded,
    LeftHanded,
}

/// Camera projection convention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraConvention {
    /// +Z forward, +Y down - OpenCV standard
    OpenCV,
    /// +Z backward, +Y up - OpenGL standard
    OpenGL,
    /// +Z forward (out of screen), +Y down - COLMAP, OpenMVG
    COLMAP,
    /// +Z forward, +Y up - WebGPU, Three.js
    WebGPU,
}

impl CameraConvention {
    /// Returns the camera's viewing direction (+Z axis in camera frame)
    pub fn camera_forward(&self) -> Vector3<f32> {
        match self {
            CameraConvention::OpenCV
            | CameraConvention::COLMAP
            | CameraConvention::WebGPU => Vector3::new(0.0, 0.0, 1.0),
            CameraConvention::OpenGL => Vector3::new(0.0, 0.0, -1.0),
        }
    }

    /// Returns the "up" direction in camera frame
    pub fn camera_up(&self) -> Vector3<f32> {
        match self {
            CameraConvention::OpenCV => Vector3::new(0.0, -1.0, 0.0),
            CameraConvention::OpenGL | CameraConvention::WebGPU => Vector3::new(0.0, 1.0, 0.0),
            CameraConvention::COLMAP => Vector3::new(0.0, -1.0, 0.0),
        }
    }

    /// Returns the "right" direction in camera frame
    pub fn camera_right(&self) -> Vector3<f32> {
        Vector3::new(1.0, 0.0, 0.0)
    }
}

/// Rig orientation - where the rig origin is defined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigOrientation {
    /// Origin at first camera
    FirstCamera,
    /// Origin at body/IMU center
    BodyCenter,
    /// Origin at world origin (0,0,0)
    WorldOrigin,
}

/// Complete frame convention specification
#[derive(Debug, Clone)]
pub struct FrameConvention {
    pub handedness: Handedness,
    pub camera_convention: CameraConvention,
    pub rig_orientation: RigOrientation,
}

impl FrameConvention {
    /// Create a right-handed frame (OpenCV default)
    pub fn right_handed(camera: CameraConvention) -> Self {
        Self {
            handedness: Handedness::RightHanded,
            camera_convention: camera,
            rig_orientation: RigOrientation::FirstCamera,
        }
    }

    /// Create a left-handed frame (OpenGL/WebGPU default)
    pub fn left_handed(camera: CameraConvention) -> Self {
        Self {
            handedness: Handedness::LeftHanded,
            camera_convention: camera,
            rig_orientation: RigOrientation::FirstCamera,
        }
    }

    /// Standard OpenCV convention: RH, +Z forward, +Y down
    pub fn opencv() -> Self {
        Self::right_handed(CameraConvention::OpenCV)
    }

    /// Standard OpenGL convention: RH, +Z backward, +Y up
    pub fn opengl() -> Self {
        Self::right_handed(CameraConvention::OpenGL)
    }

    /// COLMAP convention: RH, +Z forward, +Y down
    pub fn colmap() -> Self {
        Self::right_handed(CameraConvention::COLMAP)
    }

    /// Convert a 3x3 rotation matrix from this convention to another
    pub fn convert_rotation(&self, other: &Self, rotation: &Matrix3<f32>) -> Matrix3<f32> {
        if self.handedness == other.handedness && self.camera_convention == other.camera_convention
        {
            return *rotation;
        }

        // Build rotation matrices for each convention
        let self_forward = self.camera_convention.camera_forward();
        let self_up = self.camera_convention.camera_up();
        let self_right = self.camera_convention.camera_right();

        let other_forward = other.camera_convention.camera_forward();
        let other_up = other.camera_convention.camera_up();
        let other_right = other.camera_convention.camera_right();

        // Create basis change matrices
        let self_to_world = Matrix3::new(
            self_right.x,
            self_up.x,
            self_forward.x,
            self_right.y,
            self_up.y,
            self_forward.y,
            self_right.z,
            self_up.z,
            self_forward.z,
        );

        let world_to_other = Matrix3::new(
            other_right.x,
            other_up.x,
            other_forward.x,
            other_right.y,
            other_up.y,
            other_forward.y,
            other_right.z,
            other_up.z,
            other_forward.z,
        );

        // Convert: B^T * R * A where A is source basis, B is target basis
        world_to_other.transpose() * rotation * self_to_world
    }

    /// Convert a 4x4 transform from this convention to another
    pub fn convert_transform(&self, other: &Self, transform: &Matrix4<f32>) -> Matrix4<f32> {
        let rotation = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vector3::new(
            transform[(0, 3)],
            transform[(1, 3)],
            transform[(2, 3)],
        );

        let new_rotation = self.convert_rotation(other, &rotation);

        // Build the basis-change matrix C = B^T * A to transform translation
        let self_forward = self.camera_convention.camera_forward();
        let self_up = self.camera_convention.camera_up();
        let self_right = self.camera_convention.camera_right();

        let other_forward = other.camera_convention.camera_forward();
        let other_up = other.camera_convention.camera_up();
        let other_right = other.camera_convention.camera_right();

        let self_to_world = Matrix3::new(
            self_right.x, self_up.x, self_forward.x,
            self_right.y, self_up.y, self_forward.y,
            self_right.z, self_up.z, self_forward.z,
        );

        let world_to_other = Matrix3::new(
            other_right.x, other_up.x, other_forward.x,
            other_right.y, other_up.y, other_forward.y,
            other_right.z, other_up.z, other_forward.z,
        );

        let basis_change = world_to_other.transpose() * self_to_world;
        let new_translation = basis_change * translation;

        let mut result = Matrix4::identity();
        result.fixed_view_mut::<3, 3>(0, 0).copy_from(&new_rotation);
        result[(0, 3)] = new_translation.x;
        result[(1, 3)] = new_translation.y;
        result[(2, 3)] = new_translation.z;

        result
    }
}

impl Default for FrameConvention {
    fn default() -> Self {
        Self::opencv()
    }
}

/// Pose representation with frame convention
#[derive(Debug, Clone)]
pub struct TypedPose {
    pub position: Point3<f32>,
    pub rotation: Matrix3<f32>,
    pub convention: FrameConvention,
}

impl TypedPose {
    /// Create a new typed pose
    pub fn new(position: Point3<f32>, rotation: Matrix3<f32>, convention: FrameConvention) -> Self {
        Self {
            position,
            rotation,
            convention,
        }
    }

    /// Transform to a different frame convention
    pub fn convert_to(&self, target_convention: &FrameConvention) -> Self {
        let new_rotation = self
            .convention
            .convert_rotation(target_convention, &self.rotation);

        Self {
            position: self.position,
            rotation: new_rotation,
            convention: target_convention.clone(),
        }
    }

    /// Get transformation matrix
    pub fn to_transform(&self) -> Matrix4<f32> {
        let mut t = Matrix4::identity();
        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        t[(0, 3)] = self.position.x;
        t[(1, 3)] = self.position.y;
        t[(2, 3)] = self.position.z;
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencv_convention() {
        let opencv = FrameConvention::opencv();
        assert_eq!(opencv.handedness, Handedness::RightHanded);
        assert_eq!(opencv.camera_convention, CameraConvention::OpenCV);

        let forward = opencv.camera_convention.camera_forward();
        assert!((forward.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_colmap_convention() {
        let colmap = FrameConvention::colmap();
        let forward = colmap.camera_convention.camera_forward();
        assert!((forward.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pose_conversion() {
        let rotation = Matrix3::identity();
        let pose = TypedPose::new(
            Point3::new(1.0, 2.0, 3.0),
            rotation,
            FrameConvention::opencv(),
        );

        let converted = pose.convert_to(&FrameConvention::colmap());

        assert_eq!(
            converted.convention.camera_convention,
            CameraConvention::COLMAP
        );
    }
}
