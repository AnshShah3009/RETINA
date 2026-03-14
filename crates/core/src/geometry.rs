use nalgebra::{Isometry3, Matrix3, Matrix4, Point2, Point3, Rotation3, UnitQuaternion, Vector3};

pub type Vector6<T> = nalgebra::Vector6<T>;

/// Trait defining a camera model for projecting 3D points to 2D pixels and vice versa.
pub trait CameraModel<T: nalgebra::Scalar> {
    /// Projects a 3D point in camera coordinates to 2D pixel coordinates.
    fn project(&self, point: &Point3<T>) -> Point2<T>;

    /// Unprojects a 2D pixel coordinate to a 3D point at a given depth.
    fn unproject(&self, pixel: &Point2<T>, depth: T) -> Point3<T>;

    /// Returns the image width in pixels.
    fn width(&self) -> u32;

    /// Returns the image height in pixels.
    fn height(&self) -> u32;
}

/// A standard Pinhole Camera Model with radial and tangential distortion.
/// Uses `f64` precision.
#[derive(Debug, Clone, Copy)]
pub struct PinholeModel {
    pub intrinsics: CameraIntrinsics,
    pub distortion: Distortion,
}

impl PinholeModel {
    pub fn new(intrinsics: CameraIntrinsics, distortion: Distortion) -> Self {
        Self {
            intrinsics,
            distortion,
        }
    }
}

impl CameraModel<f64> for PinholeModel {
    fn project(&self, point: &Point3<f64>) -> Point2<f64> {
        let z = point.z;
        if z.abs() < 1e-10 {
            return Point2::new(self.intrinsics.cx, self.intrinsics.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        let (xd, yd) = self.distortion.apply(x, y);
        Point2::new(
            xd * self.intrinsics.fx + self.intrinsics.cx,
            yd * self.intrinsics.fy + self.intrinsics.cy,
        )
    }

    fn unproject(&self, pixel: &Point2<f64>, depth: f64) -> Point3<f64> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        if fx.abs() < 1e-10 || fy.abs() < 1e-10 {
            return Point3::new(0.0, 0.0, depth);
        }
        let x = (pixel.x - self.intrinsics.cx) / fx;
        let y = (pixel.y - self.intrinsics.cy) / fy;
        let (xu, yu) = self.distortion.remove(x, y);
        Point3::new(xu * depth, yu * depth, depth)
    }

    fn width(&self) -> u32 {
        self.intrinsics.width
    }
    fn height(&self) -> u32 {
        self.intrinsics.height
    }
}

/// A standard Pinhole Camera Model with radial and tangential distortion.
/// Uses `f32` precision.
#[derive(Debug, Clone, Copy)]
pub struct PinholeModelF32 {
    pub intrinsics: CameraIntrinsicsF32,
    pub distortion: DistortionF32,
}

impl PinholeModelF32 {
    pub fn new(intrinsics: CameraIntrinsicsF32, distortion: DistortionF32) -> Self {
        Self {
            intrinsics,
            distortion,
        }
    }
}

impl CameraModel<f32> for PinholeModelF32 {
    fn project(&self, point: &Point3<f32>) -> Point2<f32> {
        let z = point.z;
        if z.abs() < 1e-7 {
            return Point2::new(self.intrinsics.cx, self.intrinsics.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        let (xd, yd) = self.distortion.apply(x, y);
        Point2::new(
            xd * self.intrinsics.fx + self.intrinsics.cx,
            yd * self.intrinsics.fy + self.intrinsics.cy,
        )
    }

    fn unproject(&self, pixel: &Point2<f32>, depth: f32) -> Point3<f32> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        if fx.abs() < 1e-7 || fy.abs() < 1e-7 {
            return Point3::new(0.0, 0.0, depth);
        }
        let x = (pixel.x - self.intrinsics.cx) / fx;
        let y = (pixel.y - self.intrinsics.cy) / fy;
        let (xu, yu) = self.distortion.remove(x, y);
        Point3::new(xu * depth, yu * depth, depth)
    }

    fn width(&self) -> u32 {
        self.intrinsics.width
    }
    fn height(&self) -> u32 {
        self.intrinsics.height
    }
}

/// Camera intrinsic parameters (focal length, principal point) for `f64`.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    pub fn new_ideal(width: u32, height: u32) -> Self {
        let fx = width as f64;
        Self {
            fx,
            fy: fx,
            cx: fx / 2.0,
            cy: height as f64 / 2.0,
            width,
            height,
        }
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn inverse_matrix(&self) -> Matrix3<f64> {
        self.matrix().try_inverse().unwrap_or(Matrix3::identity())
    }

    pub fn project(&self, point: &Point3<f64>) -> Point2<f64> {
        const EPSILON: f64 = 1e-10;
        let z = point.z;
        if z.abs() < EPSILON {
            return Point2::new(self.cx, self.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        Point2::new(x * self.fx + self.cx, y * self.fy + self.cy)
    }

    pub fn unproject(&self, pixel: Point2<f64>, depth: f64) -> Point3<f64> {
        const EPSILON: f64 = 1e-10;
        let fx = if self.fx.abs() < EPSILON {
            1.0
        } else {
            self.fx
        };
        let fy = if self.fy.abs() < EPSILON {
            1.0
        } else {
            self.fy
        };
        let x = (pixel.x - self.cx) / fx;
        let y = (pixel.y - self.cy) / fy;
        Point3::new(x * depth, y * depth, depth)
    }
}

pub type CameraIntrinsicsf32 = CameraIntrinsicsF32;

/// Camera intrinsic parameters (focal length, principal point) for `f32`.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsicsF32 {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsicsF32 {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    pub fn from_intrinsics(i: &CameraIntrinsics) -> Self {
        Self {
            fx: i.fx as f32,
            fy: i.fy as f32,
            cx: i.cx as f32,
            cy: i.cy as f32,
            width: i.width,
            height: i.height,
        }
    }

    pub fn matrix(&self) -> Matrix3<f32> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn project(&self, point: &Point3<f32>) -> Point2<f32> {
        const EPSILON: f32 = 1e-7;
        let z = point.z;
        if z.abs() < EPSILON {
            return Point2::new(self.cx, self.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        Point2::new(x * self.fx + self.cx, y * self.fy + self.cy)
    }

    pub fn unproject(&self, pixel: Point2<f32>, depth: f32) -> Point3<f32> {
        const EPSILON: f32 = 1e-7;
        let fx = if self.fx.abs() < EPSILON {
            1.0
        } else {
            self.fx
        };
        let fy = if self.fy.abs() < EPSILON {
            1.0
        } else {
            self.fy
        };
        let x = (pixel.x - self.cx) / fx;
        let y = (pixel.y - self.cy) / fy;
        Point3::new(x * depth, y * depth, depth)
    }
}

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

/// Radial and Tangential distortion coefficients (Brown-Conrady model).
///
/// - `k1, k2, k3`: Radial distortion coefficients.
/// - `p1, p2`: Tangential distortion coefficients.
#[derive(Debug, Clone, Copy)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl Distortion {
    pub fn new(k1: f64, k2: f64, p1: f64, p2: f64, k3: f64) -> Self {
        Self { k1, k2, p1, p2, k3 }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    /// Apply distortion to normalized coordinates (x, y).
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx, y * radial + dy)
    }

    /// Remove distortion from distorted normalized coordinates (x, y) using iterative optimization.
    pub fn remove(&self, x: f64, y: f64) -> (f64, f64) {
        let mut xd = x;
        let mut yd = y;
        for _ in 0..10 {
            let (xu, yu) = self.apply(xd, yd);
            xd += x - xu;
            yd += y - yu;
        }
        (xd, yd)
    }
}

impl Default for Distortion {
    fn default() -> Self {
        Self::none()
    }
}

pub type Distortionf32 = DistortionF32;

/// Radial and Tangential distortion coefficients for `f32`.
#[derive(Debug, Clone, Copy)]
pub struct DistortionF32 {
    pub k1: f32,
    pub k2: f32,
    pub p1: f32,
    pub p2: f32,
    pub k3: f32,
}

impl DistortionF32 {
    pub fn new(k1: f32, k2: f32, p1: f32, p2: f32, k3: f32) -> Self {
        Self { k1, k2, p1, p2, k3 }
    }

    pub fn from_distortion(d: &Distortion) -> Self {
        Self {
            k1: d.k1 as f32,
            k2: d.k2 as f32,
            p1: d.p1 as f32,
            p2: d.p2 as f32,
            k3: d.k3 as f32,
        }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx, y * radial + dy)
    }

    pub fn remove(&self, x: f32, y: f32) -> (f32, f32) {
        let mut xd = x;
        let mut yd = y;
        for _ in 0..10 {
            let (xu, yu) = self.apply(xd, yd);
            xd += x - xu;
            yd += y - yu;
        }
        (xd, yd)
    }
}

impl Default for DistortionF32 {
    fn default() -> Self {
        Self::none()
    }
}

/// Fisheye camera distortion model (Kannala-Brandt).
/// Maps theta (angle from optical axis) to theta_d.
#[derive(Debug, Clone, Copy)]
pub struct FisheyeDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
}

impl FisheyeDistortion {
    pub fn new(k1: f64, k2: f64, k3: f64, k4: f64) -> Self {
        Self { k1, k2, k3, k4 }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        }
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let r = (x * x + y * y).sqrt();
        if r < 1e-10 {
            return (x, y);
        }
        let theta = r.atan();
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;

        let theta_d = theta
            * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8);
        let scale = theta_d / r;
        (x * scale, y * scale)
    }

    pub fn remove(&self, x: f64, y: f64) -> (f64, f64) {
        let r_d = (x * x + y * y).sqrt();
        if r_d < 1e-10 {
            return (x, y);
        }

        // Iterative solver for theta
        let mut theta = r_d;
        for _ in 0..10 {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let f = theta
                * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)
                - r_d;
            let df = 1.0
                + 3.0 * self.k1 * theta2
                + 5.0 * self.k2 * theta4
                + 7.0 * self.k3 * theta6
                + 9.0 * self.k4 * theta8;
            theta -= f / df;
        }

        let r = theta.tan();
        let scale = r / r_d;
        (x * scale, y * scale)
    }
}

impl Default for FisheyeDistortion {
    fn default() -> Self {
        Self::none()
    }
}

/// Fisheye camera distortion model (Kannala-Brandt) for `f32`.
#[derive(Debug, Clone, Copy)]
pub struct FisheyeDistortionF32 {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
}

impl FisheyeDistortionF32 {
    pub fn new(k1: f32, k2: f32, k3: f32, k4: f32) -> Self {
        Self { k1, k2, k3, k4 }
    }

    pub fn from_distortion(d: &FisheyeDistortion) -> Self {
        Self {
            k1: d.k1 as f32,
            k2: d.k2 as f32,
            k3: d.k3 as f32,
            k4: d.k4 as f32,
        }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let r = (x * x + y * y).sqrt();
        if r < 1e-7 {
            return (x, y);
        }
        let theta = r.atan();
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;

        let theta_d = theta
            * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8);
        let scale = theta_d / r;
        (x * scale, y * scale)
    }

    pub fn remove(&self, x: f32, y: f32) -> (f32, f32) {
        let r_d = (x * x + y * y).sqrt();
        if r_d < 1e-7 {
            return (x, y);
        }

        let mut theta = r_d;
        for _ in 0..10 {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let f = theta
                * (1.0 + self.k1 * theta2 + self.k2 * theta4 + self.k3 * theta6 + self.k4 * theta8)
                - r_d;
            let df = 1.0
                + 3.0 * self.k1 * theta2
                + 5.0 * self.k2 * theta4
                + 7.0 * self.k3 * theta6
                + 9.0 * self.k4 * theta8;
            theta -= f / df;
        }

        let r = theta.tan();
        let scale = r / r_d;
        (x * scale, y * scale)
    }
}

impl Default for FisheyeDistortionF32 {
    fn default() -> Self {
        Self::none()
    }
}

/// From/Into trait implementations for precision pairs (f64 ↔ f32)
///
/// These implementations enable seamless conversion between double-precision (f64)
/// and single-precision (f32) camera parameters, supporting code that works with
/// both precisions.
impl From<CameraIntrinsics> for CameraIntrinsicsF32 {
    /// Convert double-precision camera intrinsics to single-precision.
    ///
    /// # Example
    /// ```
    /// # use cv_core::{CameraIntrinsics, CameraIntrinsicsF32};
    /// let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    /// let intrinsics_f32: CameraIntrinsicsF32 = intrinsics.into();
    /// assert!((intrinsics_f32.fx - 500.0).abs() < 1e-5);
    /// ```
    fn from(c: CameraIntrinsics) -> Self {
        Self::from_intrinsics(&c)
    }
}

impl From<CameraIntrinsicsF32> for CameraIntrinsics {
    /// Convert single-precision camera intrinsics to double-precision.
    ///
    /// # Example
    /// ```
    /// # use cv_core::{CameraIntrinsics, CameraIntrinsicsF32};
    /// let intrinsics_f32 = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    /// let intrinsics: CameraIntrinsics = intrinsics_f32.into();
    /// assert!((intrinsics.fx - 500.0).abs() < 1e-10);
    /// ```
    fn from(c: CameraIntrinsicsF32) -> Self {
        CameraIntrinsics::new(
            c.fx as f64,
            c.fy as f64,
            c.cx as f64,
            c.cy as f64,
            c.width,
            c.height,
        )
    }
}

impl From<Distortion> for DistortionF32 {
    /// Convert double-precision distortion coefficients to single-precision.
    fn from(d: Distortion) -> Self {
        Self::from_distortion(&d)
    }
}

impl From<DistortionF32> for Distortion {
    /// Convert single-precision distortion coefficients to double-precision.
    fn from(d: DistortionF32) -> Self {
        Distortion::new(
            d.k1 as f64,
            d.k2 as f64,
            d.p1 as f64,
            d.p2 as f64,
            d.k3 as f64,
        )
    }
}

impl From<FisheyeDistortion> for FisheyeDistortionF32 {
    /// Convert double-precision fisheye distortion coefficients to single-precision.
    fn from(d: FisheyeDistortion) -> Self {
        Self {
            k1: d.k1 as f32,
            k2: d.k2 as f32,
            k3: d.k3 as f32,
            k4: d.k4 as f32,
        }
    }
}

impl From<FisheyeDistortionF32> for FisheyeDistortion {
    /// Convert single-precision fisheye distortion coefficients to double-precision.
    fn from(d: FisheyeDistortionF32) -> Self {
        Self {
            k1: d.k1 as f64,
            k2: d.k2 as f64,
            k3: d.k3 as f64,
            k4: d.k4 as f64,
        }
    }
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

/// A 2D axis-aligned rectangle.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn x1(&self) -> f32 {
        self.x
    }
    pub fn y1(&self) -> f32 {
        self.y
    }
    pub fn x2(&self) -> f32 {
        self.x + self.w
    }
    pub fn y2(&self) -> f32 {
        self.y + self.h
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn iou(&self, other: &Rect) -> f32 {
        let x1 = self.x1().max(other.x1());
        let y1 = self.y1().max(other.y1());
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        let intersection = w * h;

        if intersection == 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

/// A 2D rotated rectangle defined by center, size, and angle.
#[derive(Debug, Clone, Copy, Default)]
pub struct RotatedRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32, // Degrees
}

impl RotatedRect {
    pub fn new(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> Self {
        Self {
            cx,
            cy,
            w,
            h,
            angle,
        }
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Get the 4 corners of the rotated rectangle
    pub fn points(&self) -> [[f32; 2]; 4] {
        let angle_rad = self.angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let half_w = self.w / 2.0;
        let half_h = self.h / 2.0;

        let mut pts = [[0.0, 0.0]; 4];
        // Relative corners before rotation
        let corners = [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ];

        for i in 0..4 {
            pts[i][0] = self.cx + corners[i][0] * cos_a - corners[i][1] * sin_a;
            pts[i][1] = self.cy + corners[i][0] * sin_a + corners[i][1] * cos_a;
        }
        pts
    }
}

/// A generic 2D polygon defined by vertices.
#[derive(Debug, Clone, Default)]
pub struct Polygon {
    pub points: Vec<[f32; 2]>,
}

impl Polygon {
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        Self { points }
    }

    pub fn area(&self) -> f32 {
        if self.points.len() < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..self.points.len() {
            let p1 = self.points[i];
            let p2 = self.points[(i + 1) % self.points.len()];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }
        area * 0.5 // Keep signed area for winding check
    }

    pub fn is_clockwise(&self) -> bool {
        self.area() < 0.0
    }

    pub fn ensure_counter_clockwise(&mut self) {
        if self.is_clockwise() {
            self.points.reverse();
        }
    }

    pub fn unsigned_area(&self) -> f32 {
        self.area().abs()
    }
}

/// Calculates the Intersection over Union (IoU) of two rotated rectangles.
pub fn rotated_iou(r1: &RotatedRect, r2: &RotatedRect) -> f32 {
    let mut p1 = Polygon::new(r1.points().to_vec());
    let mut p2 = Polygon::new(r2.points().to_vec());
    p1.ensure_counter_clockwise();
    p2.ensure_counter_clockwise();
    polygon_iou(&p1, &p2)
}

/// Calculates the Intersection over Union (IoU) of two polygons.
pub fn polygon_iou(p1: &Polygon, p2: &Polygon) -> f32 {
    let inter_area = intersection_area_polygons(p1, p2);
    let a1 = p1.unsigned_area();
    let a2 = p2.unsigned_area();
    if inter_area <= 0.0 {
        return 0.0;
    }
    let union_area = a1 + a2 - inter_area;
    inter_area / union_area
}

/// Calculates the intersection area of two convex polygons using Sutherland-Hodgman clipping.
pub fn intersection_area_polygons(p1: &Polygon, p2: &Polygon) -> f32 {
    // Sutherland-Hodgman clipping for generic convex polygons
    let pts1 = &p1.points;
    let pts2 = &p2.points;

    if pts1.len() < 3 || pts2.len() < 3 {
        return 0.0;
    }

    let mut poly = pts1.clone();

    // Clip pts1 against each edge of pts2
    for i in 0..pts2.len() {
        let edge_p1 = pts2[i];
        let edge_p2 = pts2[(i + 1) % pts2.len()];

        let mut next_poly = Vec::new();
        if poly.is_empty() {
            return 0.0;
        }

        for j in 0..poly.len() {
            let cur = poly[j];
            let prev = poly[(j + poly.len() - 1) % poly.len()];

            let is_cur_inside = is_inside(edge_p1, edge_p2, cur);
            let is_prev_inside = is_inside(edge_p1, edge_p2, prev);

            if is_cur_inside {
                if !is_prev_inside {
                    next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
                }
                next_poly.push(cur);
            } else if is_prev_inside {
                next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
            }
        }
        poly = next_poly;
    }

    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..poly.len() {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % poly.len()];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }
    area.abs() * 0.5
}

fn is_inside(p1: [f32; 2], p2: [f32; 2], p: [f32; 2]) -> bool {
    (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= 0.0
}

fn intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> [f32; 2] {
    let a1 = b[1] - a[1];
    let b1 = a[0] - b[0];
    let c1 = a1 * a[0] + b1 * a[1];

    let a2 = d[1] - c[1];
    let b2 = c[0] - d[0];
    let c2 = a2 * c[0] + b2 * c[1];

    let det = a1 * b2 - a2 * b1;
    if det.abs() < 1e-6 {
        return a; // Parallel
    }
    [(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det]
}

/// A line in Hesse normal form (rho, theta).
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HoughLine {
    /// Distance from the origin.
    pub rho: f32,
    /// Angle in radians.
    pub theta: f32,
    /// Accumulator score.
    pub score: u32,
}

impl HoughLine {
    pub fn new(rho: f32, theta: f32, score: u32) -> Self {
        Self { rho, theta, score }
    }
}

/// A circle in (x, y, radius) form.
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HoughCircle {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    pub score: u32,
}

impl HoughCircle {
    pub fn new(cx: f32, cy: f32, r: f32, score: u32) -> Self {
        Self { cx, cy, r, score }
    }
}

/// An object detection result.
#[derive(Debug, Clone, Copy, Default)]
pub struct Detection {
    /// Bounding box of the detection.
    pub rect: Rect,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
    /// Class identifier.
    pub class_id: i32,
}

impl Detection {
    pub fn new(rect: Rect, score: f32, class_id: i32) -> Self {
        Self {
            rect,
            score,
            class_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point3, Vector3};

    mod pose_tests {
        use super::*;

        #[test]
        fn test_pose_identity() {
            let pose = Pose::identity();
            assert!((pose.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10);
            assert_eq!(pose.translation, Vector3::zeros());
        }

        #[test]
        fn test_pose_new() {
            let rotation = Matrix3::identity();
            let translation = Vector3::new(1.0, 2.0, 3.0);
            let pose = Pose::new(rotation, translation);

            assert_eq!(pose.translation.x, 1.0);
            assert_eq!(pose.translation.y, 2.0);
            assert_eq!(pose.translation.z, 3.0);
        }

        #[test]
        fn test_pose_inverse_roundtrip() {
            let rotation = Matrix3::identity();
            let translation = Vector3::new(1.0, 2.0, 3.0);
            let pose = Pose::new(rotation, translation);

            let result = pose.compose(&pose.inverse());
            assert!(
                (result.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10
            );
            assert!(result.translation.norm() < 1e-10);
        }

        #[test]
        fn test_pose_compose_identity() {
            let translation = Vector3::new(0.5, 0.6, 0.7);
            let pose = Pose::new(Matrix3::identity(), translation);

            let result = pose.compose(&Pose::identity());
            assert!((result.translation - translation).norm() < 1e-10);
        }

        #[test]
        fn test_pose_transform_point() {
            let pose = Pose::identity();
            let point = Point3::new(1.0, 2.0, 3.0);
            let result = pose.transform_point(&point);

            assert!((result - point).norm() < 1e-10);
        }

        #[test]
        fn test_pose_matrix() {
            let pose = Pose::identity();
            let matrix = pose.matrix();

            assert_eq!(matrix[(0, 0)], 1.0);
            assert_eq!(matrix[(1, 1)], 1.0);
            assert_eq!(matrix[(2, 2)], 1.0);
            assert_eq!(matrix[(3, 3)], 1.0);
        }

        #[test]
        fn test_pose_default() {
            let pose = Pose::default();
            assert!((pose.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10);
            assert_eq!(pose.translation, Vector3::zeros());
        }
    }

    mod camera_intrinsics_tests {
        use super::*;

        #[test]
        fn test_camera_intrinsics_new() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

            assert_eq!(intrinsics.fx, 500.0);
            assert_eq!(intrinsics.fy, 500.0);
            assert_eq!(intrinsics.cx, 320.0);
            assert_eq!(intrinsics.cy, 240.0);
        }

        #[test]
        fn test_camera_intrinsics_new_ideal() {
            let intrinsics = CameraIntrinsics::new_ideal(640, 480);

            assert_eq!(intrinsics.fx, 640.0);
            assert_eq!(intrinsics.fy, 640.0);
            assert_eq!(intrinsics.cx, 320.0);
            assert_eq!(intrinsics.cy, 240.0);
        }

        #[test]
        fn test_camera_intrinsics_f32() {
            let intrinsics = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);

            assert_eq!(intrinsics.fx, 500.0f32);
            assert_eq!(intrinsics.fy, 500.0f32);
        }
    }

    mod distortion_tests {
        use super::*;

        #[test]
        fn test_distortion_identity() {
            let dist = Distortion::new(0.0, 0.0, 0.0, 0.0, 0.0);

            let (xd, yd) = dist.apply(0.5, 0.5);
            assert!((xd - 0.5).abs() < 1e-10);
            assert!((yd - 0.5).abs() < 1e-10);
        }

        #[test]
        fn test_distortion_apply_remove_roundtrip() {
            let dist = Distortion::new(0.1, 0.01, 0.001, 0.001, 0.0);

            let (xd, yd) = dist.apply(0.3, 0.4);
            let (xr, yr) = dist.remove(xd, yd);

            assert!((xr - 0.3).abs() < 1e-4);
            assert!((yr - 0.4).abs() < 1e-4);
        }

        #[test]
        fn test_distortion_at_origin() {
            let dist = Distortion::new(0.5, 0.3, 0.01, 0.01, 0.1);

            let (xd, yd) = dist.apply(0.0, 0.0);
            assert!((xd).abs() < 1e-10);
            assert!((yd).abs() < 1e-10);
        }

        #[test]
        fn test_fisheye_distortion_identity() {
            let dist = FisheyeDistortion::new(0.0, 0.0, 0.0, 0.0);

            let (xd, yd) = dist.apply(0.0, 0.0);
            assert!((xd).abs() < 1e-10);
            assert!((yd).abs() < 1e-10);
        }

        #[test]
        fn test_fisheye_distortion_roundtrip() {
            let dist = FisheyeDistortion::new(0.1, 0.01, 0.001, 0.001);

            let (xd, yd) = dist.apply(0.3, 0.4);
            let (xr, yr) = dist.remove(xd, yd);

            assert!((xr - 0.3).abs() < 1e-3);
            assert!((yr - 0.4).abs() < 1e-3);
        }
    }

    mod rect_tests {
        use super::*;

        #[test]
        fn test_rect_area() {
            let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_area_negative() {
            let rect = Rect::new(0.0, 0.0, 0.0, 20.0);
            assert!((rect.area() - 0.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_identical() {
            let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
            assert!((rect.iou(&rect) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_no_overlap() {
            let rect1 = Rect::new(0.0, 0.0, 10.0, 10.0);
            let rect2 = Rect::new(20.0, 20.0, 10.0, 10.0);
            assert!((rect1.iou(&rect2)).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_partial_overlap() {
            let rect1 = Rect::new(0.0, 0.0, 10.0, 10.0);
            let rect2 = Rect::new(5.0, 5.0, 10.0, 10.0);

            let iou = rect1.iou(&rect2);
            assert!(iou > 0.0 && iou < 1.0);
        }

        #[test]
        fn test_rect_bounds() {
            let rect = Rect::new(1.0, 2.0, 10.0, 20.0);

            assert!((rect.x1() - 1.0).abs() < 1e-5);
            assert!((rect.y1() - 2.0).abs() < 1e-5);
            assert!((rect.x2() - 11.0).abs() < 1e-5);
            assert!((rect.y2() - 22.0).abs() < 1e-5);
        }
    }

    mod polygon_tests {
        use super::*;

        fn create_square() -> Polygon {
            Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            }
        }

        #[test]
        fn test_polygon_area_square() {
            let poly = create_square();
            let area = poly.unsigned_area();

            assert!((area - 100.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_area_triangle() {
            let poly = Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0]],
            };

            assert!((poly.unsigned_area() - 50.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_area_empty() {
            let poly = Polygon { points: vec![] };
            assert_eq!(poly.area(), 0.0);
        }

        #[test]
        fn test_polygon_iou_identical() {
            let poly = create_square();

            assert!((polygon_iou(&poly, &poly) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_iou_no_overlap() {
            let poly1 = Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            };
            let poly2 = Polygon {
                points: vec![[20.0f32, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]],
            };

            assert!(polygon_iou(&poly1, &poly2).abs() < 1e-5);
        }
    }

    mod geometry_util_tests {
        use super::*;

        #[test]
        fn test_skew_symmetric_zero() {
            let v = Vector3::zeros();
            let skew = skew_symmetric(&v);

            assert_eq!(skew, Matrix3::zeros());
        }

        #[test]
        fn test_skew_symmetric_unit_x() {
            let v = Vector3::x();
            let skew = skew_symmetric(&v);

            assert_eq!(skew[(0, 1)], 0.0);
            assert_eq!(skew[(0, 2)], 0.0);
            assert_eq!(skew[(1, 0)], 0.0);
            assert_eq!(skew[(1, 2)], -1.0);
            assert_eq!(skew[(2, 0)], 0.0);
            assert_eq!(skew[(2, 1)], 1.0);
        }

        #[test]
        fn test_twist_to_se3_zero() {
            let twist = Vector6::zeros();
            let (r, t) = twist_to_se3(&twist);

            assert!(r.is_identity(1e-10));
            assert!(t.norm() < 1e-10);
        }
    }

    mod rotated_rect_tests {
        use super::*;

        #[test]
        fn test_rotated_rect_area() {
            let rect = RotatedRect::new(10.0, 10.0, 20.0, 10.0, 0.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rotated_rect_area_rotated() {
            let rect = RotatedRect::new(10.0, 10.0, 20.0, 10.0, 45.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rotated_rect_points() {
            let rect = RotatedRect::new(50.0, 50.0, 10.0, 20.0, 0.0);
            let points = rect.points();
            assert_eq!(points.len(), 4);
        }
    }

    mod from_into_traits_tests {
        use super::*;

        #[test]
        fn test_camera_intrinsics_f64_to_f32() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let intrinsics_f32: CameraIntrinsicsF32 = intrinsics.into();
            assert!((intrinsics_f32.fx - 500.0).abs() < 1e-5);
            assert!((intrinsics_f32.fy - 500.0).abs() < 1e-5);
            assert!((intrinsics_f32.cx - 320.0).abs() < 1e-5);
            assert!((intrinsics_f32.cy - 240.0).abs() < 1e-5);
            assert_eq!(intrinsics_f32.width, 640);
            assert_eq!(intrinsics_f32.height, 480);
        }

        #[test]
        fn test_camera_intrinsics_f32_to_f64() {
            let intrinsics_f32 = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let intrinsics: CameraIntrinsics = intrinsics_f32.into();
            assert!((intrinsics.fx - 500.0).abs() < 1e-10);
            assert!((intrinsics.fy - 500.0).abs() < 1e-10);
            assert!((intrinsics.cx - 320.0).abs() < 1e-10);
            assert!((intrinsics.cy - 240.0).abs() < 1e-10);
            assert_eq!(intrinsics.width, 640);
            assert_eq!(intrinsics.height, 480);
        }

        #[test]
        fn test_distortion_f64_to_f32() {
            let distortion = Distortion::new(0.1, 0.01, 0.001, 0.0001, 0.00001);
            let distortion_f32: DistortionF32 = distortion.into();
            assert!((distortion_f32.k1 - 0.1).abs() < 1e-5);
            assert!((distortion_f32.k2 - 0.01).abs() < 1e-5);
            assert!((distortion_f32.p1 - 0.001).abs() < 1e-5);
            assert!((distortion_f32.p2 - 0.0001).abs() < 1e-5);
            assert!((distortion_f32.k3 - 0.00001).abs() < 1e-6);
        }

        #[test]
        fn test_distortion_f32_to_f64() {
            let distortion_f32 = DistortionF32::new(0.1, 0.01, 0.001, 0.0001, 0.00001);
            let distortion: Distortion = distortion_f32.into();
            // f32→f64 conversions have ~1e-7 relative error for values ~0.1
            assert!((distortion.k1 - 0.1).abs() < 1e-7);
            assert!((distortion.k2 - 0.01).abs() < 1e-8);
            assert!((distortion.p1 - 0.001).abs() < 1e-8);
            assert!((distortion.p2 - 0.0001).abs() < 1e-8);
            assert!((distortion.k3 - 0.00001).abs() < 1e-9);
        }

        #[test]
        fn test_fisheye_distortion_f64_to_f32() {
            let fisheye = FisheyeDistortion::new(0.1, 0.01, 0.001, 0.0001);
            let fisheye_f32: FisheyeDistortionF32 = fisheye.into();
            assert!((fisheye_f32.k1 - 0.1).abs() < 1e-5);
            assert!((fisheye_f32.k2 - 0.01).abs() < 1e-5);
            assert!((fisheye_f32.k3 - 0.001).abs() < 1e-5);
            assert!((fisheye_f32.k4 - 0.0001).abs() < 1e-5);
        }

        #[test]
        fn test_fisheye_distortion_f32_to_f64() {
            let fisheye_f32 = FisheyeDistortionF32::new(0.1, 0.01, 0.001, 0.0001);
            let fisheye: FisheyeDistortion = fisheye_f32.into();
            // f32→f64 conversions have ~1e-7 relative error for values ~0.1
            assert!((fisheye.k1 - 0.1).abs() < 1e-7);
            assert!((fisheye.k2 - 0.01).abs() < 1e-8);
            assert!((fisheye.k3 - 0.001).abs() < 1e-8);
            assert!((fisheye.k4 - 0.0001).abs() < 1e-8);
        }
    }

    mod projection_tests {
        use super::*;

        #[test]
        fn test_project_point_without_extrinsics_or_distortion() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_some());

            let proj = result.unwrap();
            // u = fx * x/z + cx = 500 * 0.1 + 320 = 370
            // v = fy * y/z + cy = 500 * 0.05 + 240 = 265
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_with_extrinsics() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let extrinsics = Pose::identity();
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, Some(&extrinsics), None);
            assert!(result.is_some());

            let proj = result.unwrap();
            // With identity pose, should be same as without extrinsics
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_with_distortion() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let distortion = Distortion::new(0.0, 0.0, 0.0, 0.0, 0.0); // No distortion
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, None, Some(&distortion));
            assert!(result.is_some());

            let proj = result.unwrap();
            // No distortion, should be same as without
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_guards_against_zero_depth() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, 0.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }

        #[test]
        fn test_project_point_guards_against_negative_depth() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, -1.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }

        #[test]
        fn test_project_point_guards_against_non_finite_coordinates() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, f64::NAN);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }
    }
}
