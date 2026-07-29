use nalgebra::{Matrix3, Point2, Point3};
use super::distortion::{Distortion, DistortionF32};

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
