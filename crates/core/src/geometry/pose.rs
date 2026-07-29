use nalgebra::{Isometry3, Matrix3, Matrix4, Point2, Point3, Rotation3, UnitQuaternion, Vector3};

use super::camera::CameraIntrinsics;
use super::distortion::Distortion;
use super::Vector6;

/// A 3D rigid body transformation (Rotation + Translation).
/// `Pose` transforms points from the local frame to the parent frame (e.g. Camera to World).
/// Note: Sometimes conventions differ. Here, `transform_point` applies R*p + t.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl Pose {
    /// Create a new Pose from a rotation matrix and translation vector
    /// Converts the rotation matrix to a quaternion internally
    pub fn new(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Self {
        let quat =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation));
        Self {
            rotation: quat,
            translation,
        }
    }

    /// Create a Pose from a rotation matrix (reference) and translation vector
    pub fn from_rotation_translation(r: &Rotation3<f64>, t: &Vector3<f64>) -> Self {
        Self {
            rotation: UnitQuaternion::from_rotation_matrix(r),
            translation: *t,
        }
    }

    /// Create a Pose from a quaternion and translation vector
    pub fn from_quat_translation(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Get the rotation as a 3x3 matrix
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix().into_inner()
    }

    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation_matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        let rotated = self.rotation * point;
        rotated + self.translation
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.inverse();
        Self {
            rotation: inv_rot,
            translation: -(inv_rot * self.translation),
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }
}

impl From<Isometry3<f64>> for Pose {
    fn from(iso: Isometry3<f64>) -> Self {
        Self {
            rotation: iso.rotation,
            translation: iso.translation.vector,
        }
    }
}

impl From<Pose> for Isometry3<f64> {
    fn from(pose: Pose) -> Self {
        Isometry3::from_parts(
            nalgebra::Translation3::from(pose.translation),
            pose.rotation,
        )
    }
}

pub fn twist_to_se3(twist: &Vector6<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let omega = Vector3::new(twist[0], twist[1], twist[2]);
    let v = Vector3::new(twist[3], twist[4], twist[5]);

    let theta = omega.norm();
    if theta < 1e-10 {
        (Matrix3::identity(), v)
    } else {
        let axis = omega / theta;
        let skew = skew_symmetric(&axis);
        let r = Matrix3::identity() + theta.sin() * skew + (1.0 - theta.cos()) * skew * skew;
        let v_mat = Matrix3::identity()
            + ((1.0 - theta.cos()) / theta) * skew
            + ((theta - theta.sin()) / theta) * skew * skew;
        let t = v_mat * v;
        (r, t)
    }
}

pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// Unified 3D-to-2D point projection with optional extrinsics and distortion.
///
/// This is the canonical single-point projection function combining camera extrinsics,
/// intrinsics, and distortion into one operation. It handles:
/// - Transformation from world to camera coordinates (via extrinsics)
/// - Normalization to image plane
/// - Lens distortion (radial and tangential)
/// - Projection to pixel coordinates
///
/// # Arguments
/// * `point` - 3D point in world coordinates
/// * `intrinsics` - Camera intrinsic matrix (focal length, principal point)
/// * `extrinsics` - Optional camera extrinsic transformation (pose).
///   If `None`, assumes point is already in camera coordinates.
/// * `distortion` - Optional lens distortion coefficients.
///   If `None`, performs ideal pinhole projection.
///
/// # Returns
/// - `Some(Point2<f64>)` with pixel coordinates if projection succeeds
/// - `None` if depth is non-positive (point behind or at camera), non-finite, or invalid
///
/// # Computational Flow
/// 1. Apply extrinsics: `p_cam = R * p_world + t`
/// 2. Check depth: `if p_cam.z <= 1e-10, return None`
/// 3. Normalize: `x_norm = p_cam.x / p_cam.z, y_norm = p_cam.y / p_cam.z`
/// 4. Apply distortion: `(x_dist, y_dist) = distortion.apply(x_norm, y_norm)`
/// 5. Project to pixels: `u = fx * x_dist + cx, v = fy * y_dist + cy`
///
/// # Examples
/// ```
/// use cv_core::{project_point, CameraIntrinsics, Pose, Distortion};
/// use nalgebra::{Point3, Matrix3, Vector3};
///
/// // Simple projection without extrinsics or distortion
/// let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
/// let point = Point3::new(0.1, 0.05, 1.0);
/// let projected = project_point(&point, &intrinsics, None, None);
/// assert!(projected.is_some());
///
/// // With extrinsics (camera pose)
/// let pose = Pose::identity();
/// let with_extrinsics = project_point(&point, &intrinsics, Some(&pose), None);
/// assert_eq!(projected, with_extrinsics);
/// ```
pub fn project_point(
    point: &Point3<f64>,
    intrinsics: &CameraIntrinsics,
    extrinsics: Option<&Pose>,
    distortion: Option<&Distortion>,
) -> Option<Point2<f64>> {
    // Apply extrinsics if provided
    let p_cam = if let Some(ext) = extrinsics {
        ext.rotation * point.coords + ext.translation
    } else {
        point.coords
    };

    // Check depth guard - must have positive, finite depth
    if !p_cam.iter().all(|v| v.is_finite()) || p_cam[2] <= 1e-10 {
        return None;
    }

    // Normalize to image plane
    let x = p_cam[0] / p_cam[2];
    let y = p_cam[1] / p_cam[2];

    // Apply distortion if provided
    let (xd, yd) = if let Some(dist) = distortion {
        dist.apply(x, y)
    } else {
        (x, y)
    };

    // Project to pixel coordinates
    Some(Point2::new(
        intrinsics.fx * xd + intrinsics.cx,
        intrinsics.fy * yd + intrinsics.cy,
    ))
}
