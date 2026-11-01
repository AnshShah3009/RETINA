use super::distortion::{Distortion, DistortionF32};
use nalgebra::{Matrix3, Point2, Point3};

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

    /// Project a camera-frame point, or an error when it has no pixel.
    ///
    /// `z <= 0` is on or behind the focal plane. A non-finite `z` is not a
    /// depth. Neither is the principal point: `(100, 0, 0)` and `(1, 0, -1)`
    /// used to come back as `(cx, cy)` and `(-180, 240)` respectively.
    pub fn try_project(&self, point: &Point3<f64>) -> crate::Result<Point2<f64>> {
        let z = point.z;
        if !z.is_finite() || z <= 0.0 {
            return Err(crate::Error::InvalidInput(format!(
                "cannot project a point with non-positive or non-finite depth z={z}"
            )));
        }
        let x = point.x / z;
        let y = point.y / z;
        let (xd, yd) = self.distortion.apply(x, y);
        Ok(Point2::new(
            xd * self.intrinsics.fx + self.intrinsics.cx,
            yd * self.intrinsics.fy + self.intrinsics.cy,
        ))
    }

    /// Unproject a pixel at `depth`, or an error when the focal length cannot
    /// define a ray.
    ///
    /// A zero or non-finite focal length used to return the optical axis
    /// `(0, 0, depth)`, which is what unprojecting the principal point looks
    /// like. `fx = 500` is unchanged.
    pub fn try_unproject(&self, pixel: &Point2<f64>, depth: f64) -> crate::Result<Point3<f64>> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        if !fx.is_finite() || !fy.is_finite() || fx == 0.0 || fy == 0.0 {
            return Err(crate::Error::InvalidInput(format!(
                "cannot unproject with non-finite or zero focal length fx={fx}, fy={fy}"
            )));
        }
        let x = (pixel.x - self.intrinsics.cx) / fx;
        let y = (pixel.y - self.intrinsics.cy) / fy;
        let (xu, yu) = self.distortion.remove(x, y);
        Ok(Point3::new(xu * depth, yu * depth, depth))
    }
}

impl CameraModel<f64> for PinholeModel {
    fn project(&self, point: &Point3<f64>) -> Point2<f64> {
        // `CameraModel::project` returns a point, so a rejected depth is a
        // non-finite pixel rather than the principal point.
        self.try_project(point)
            .unwrap_or(Point2::new(f64::NAN, f64::NAN))
    }

    fn unproject(&self, pixel: &Point2<f64>, depth: f64) -> Point3<f64> {
        self.try_unproject(pixel, depth)
            .unwrap_or(Point3::new(f64::NAN, f64::NAN, f64::NAN))
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

    /// `f32` twin of [`PinholeModel::try_project`].
    pub fn try_project(&self, point: &Point3<f32>) -> crate::Result<Point2<f32>> {
        let z = point.z;
        if !z.is_finite() || z <= 0.0 {
            return Err(crate::Error::InvalidInput(format!(
                "cannot project a point with non-positive or non-finite depth z={z}"
            )));
        }
        let x = point.x / z;
        let y = point.y / z;
        let (xd, yd) = self.distortion.apply(x, y);
        Ok(Point2::new(
            xd * self.intrinsics.fx + self.intrinsics.cx,
            yd * self.intrinsics.fy + self.intrinsics.cy,
        ))
    }

    /// `f32` twin of [`PinholeModel::try_unproject`].
    pub fn try_unproject(&self, pixel: &Point2<f32>, depth: f32) -> crate::Result<Point3<f32>> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        if !fx.is_finite() || !fy.is_finite() || fx == 0.0 || fy == 0.0 {
            return Err(crate::Error::InvalidInput(format!(
                "cannot unproject with non-finite or zero focal length fx={fx}, fy={fy}"
            )));
        }
        let x = (pixel.x - self.intrinsics.cx) / fx;
        let y = (pixel.y - self.intrinsics.cy) / fy;
        let (xu, yu) = self.distortion.remove(x, y);
        Ok(Point3::new(xu * depth, yu * depth, depth))
    }
}

impl CameraModel<f32> for PinholeModelF32 {
    fn project(&self, point: &Point3<f32>) -> Point2<f32> {
        self.try_project(point)
            .unwrap_or(Point2::new(f32::NAN, f32::NAN))
    }

    fn unproject(&self, pixel: &Point2<f32>, depth: f32) -> Point3<f32> {
        self.try_unproject(pixel, depth)
            .unwrap_or(Point3::new(f32::NAN, f32::NAN, f32::NAN))
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

    /// Inverse of the intrinsic matrix, or `None` when it is singular.
    ///
    /// The intrinsic matrix has determinant `fx · fy`, so it is singular exactly
    /// when either focal length is zero - and the fields are public, so such a
    /// value is constructible.
    ///
    /// This was `try_inverse().unwrap_or(Matrix3::identity())`, and the identity
    /// is not a neutral fallback: it is a *valid* matrix that means "this camera
    /// has no intrinsics", so every point is used as its own pixel coordinate.
    /// Measured, with `fx = 0`: a pixel `(123, 456, 1)` maps to
    /// `(123, 456, 1)` unchanged. Eight call sites feed this into calibration,
    /// essential-matrix, PnP and triangulation solves, so a degenerate focal
    /// length would silently produce those answers in normalised coordinates and
    /// report success.
    ///
    /// [`CameraIntrinsics::inverse_matrix_or_identity`] keeps the old signature
    /// for callers that genuinely want that behaviour.
    pub fn try_inverse_matrix(&self) -> Option<Matrix3<f64>> {
        self.matrix().try_inverse()
    }

    /// Inverse of the intrinsic matrix, falling back to the identity when singular.
    ///
    /// The identity means "no intrinsics" - every point is its own pixel
    /// coordinate. Prefer [`CameraIntrinsics::try_inverse_matrix`] unless that is
    /// genuinely what a zero focal length should mean.
    pub fn inverse_matrix_or_identity(&self) -> Matrix3<f64> {
        self.try_inverse_matrix().unwrap_or_else(Matrix3::identity)
    }

    pub fn inverse_matrix(&self) -> Matrix3<f64> {
        self.inverse_matrix_or_identity()
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

#[cfg(test)]
mod projection_guard_tests {
    use super::*;
    use crate::geometry::distortion::{Distortion, DistortionF32};
    use nalgebra::{Point2, Point3};

    fn model_f64(fx: f64, fy: f64) -> PinholeModel {
        PinholeModel::new(
            CameraIntrinsics::new(fx, fy, 320.0, 240.0, 640, 480),
            Distortion::none(),
        )
    }

    fn model_f32(fx: f32, fy: f32) -> PinholeModelF32 {
        PinholeModelF32::new(
            CameraIntrinsicsF32::new(fx, fy, 320.0, 240.0, 640, 480),
            DistortionF32::none(),
        )
    }

    #[test]
    fn in_front_point_projects_unchanged() {
        let model = model_f64(500.0, 500.0);
        // On the optical axis, in front of the camera: the principal point.
        let axis = model.project(&Point3::new(0.0, 0.0, 5.0));
        assert_eq!(axis, Point2::new(320.0, 240.0));
        // (1, 0, 1) → u = 500 * 1 / 1 + 320 = 820.
        let off = model.project(&Point3::new(1.0, 0.0, 1.0));
        assert!((off.x - 820.0).abs() < 1e-9);
        assert!((off.y - 240.0).abs() < 1e-9);
    }

    #[test]
    fn non_positive_or_non_finite_depth_is_not_the_principal_point() {
        let model = model_f64(500.0, 500.0);
        let principal = Point2::new(320.0, 240.0);
        // (100, 0, 0) used to come back as the principal point.
        let on_plane = model.project(&Point3::new(100.0, 0.0, 0.0));
        assert!(
            !on_plane.x.is_finite() && !on_plane.y.is_finite(),
            "z = 0 projected to {on_plane:?}"
        );
        assert_ne!(on_plane, principal);

        // (1, 0, -1) used to project in front, to (-180, 240).
        let behind = model.project(&Point3::new(1.0, 0.0, -1.0));
        assert!(
            !behind.x.is_finite() && !behind.y.is_finite(),
            "z = -1 projected to {behind:?}"
        );
        assert_ne!(behind, Point2::new(-180.0, 240.0));

        let inf = model.project(&Point3::new(1.0, 0.0, f64::INFINITY));
        assert!(
            !inf.x.is_finite() && !inf.y.is_finite(),
            "z = +inf → {inf:?}"
        );
        let nan = model.project(&Point3::new(1.0, 0.0, f64::NAN));
        assert!(
            !nan.x.is_finite() && !nan.y.is_finite(),
            "z = NaN → {nan:?}"
        );
    }

    #[test]
    fn f32_non_positive_depth_is_not_the_principal_point() {
        let model = model_f32(500.0, 500.0);
        let on_plane = model.project(&Point3::new(100.0, 0.0, 0.0));
        assert!(
            !on_plane.x.is_finite() && !on_plane.y.is_finite(),
            "f32 z = 0 projected to {on_plane:?}"
        );
        let behind = model.project(&Point3::new(1.0, 0.0, -1.0));
        assert!(
            !behind.x.is_finite() && !behind.y.is_finite(),
            "f32 z = -1 projected to {behind:?}"
        );
        let front = model.project(&Point3::new(1.0, 0.0, 1.0));
        assert!((front.x - 820.0).abs() < 1e-3);
        assert!((front.y - 240.0).abs() < 1e-3);
    }

    #[test]
    fn zero_or_non_finite_focal_length_is_not_a_principal_ray() {
        let pixel = Point2::new(1000.0, 800.0);
        let bogus = Point3::new(0.0, 0.0, 2.0);
        for (fx, fy, label) in [
            (0.0, 500.0, "fx = 0"),
            (500.0, 0.0, "fy = 0"),
            (f64::INFINITY, 500.0, "fx = inf"),
            (500.0, f64::NAN, "fy = NaN"),
        ] {
            let ray = model_f64(fx, fy).unproject(&pixel, 2.0);
            assert!(
                !ray.x.is_finite() && !ray.y.is_finite(),
                "{label} unprojected to {ray:?}"
            );
            assert_ne!(ray, bogus, "{label}");
        }

        // fx = 500, pixel (1000, 800), depth 2 → ((1000-320)/500*2, (800-240)/500*2, 2).
        let ray = model_f64(500.0, 500.0).unproject(&pixel, 2.0);
        assert!((ray.x - 2.72).abs() < 1e-12, "x {}", ray.x);
        assert!((ray.y - 2.24).abs() < 1e-12, "y {}", ray.y);
        assert!((ray.z - 2.0).abs() < 1e-12, "z {}", ray.z);
    }

    #[test]
    fn f32_zero_focal_length_is_not_a_principal_ray() {
        let pixel = Point2::new(1000.0f32, 800.0);
        let ray = model_f32(0.0, 500.0).unproject(&pixel, 2.0);
        assert!(
            !ray.x.is_finite() && !ray.y.is_finite(),
            "f32 fx = 0 unprojected to {ray:?}"
        );
        let ok = model_f32(500.0, 500.0).unproject(&pixel, 2.0);
        assert!((ok.x - 2.72).abs() < 1e-5, "x {}", ok.x);
        assert!((ok.y - 2.24).abs() < 1e-5, "y {}", ok.y);
        assert!((ok.z - 2.0).abs() < 1e-5, "z {}", ok.z);
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
