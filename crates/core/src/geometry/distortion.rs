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
            // Guard the derivative: strong distortion coefficients can drive
            // df toward zero, producing inf/NaN steps that corrupt every
            // subsequent unprojection.
            if !df.is_finite() || df.abs() < 1e-12 || !f.is_finite() {
                break;
            }
            let step = f / df;
            if !step.is_finite() {
                break;
            }
            theta -= step;
        }

        // Non-finite theta would poison tan() and the unprojected ray.
        if !theta.is_finite() {
            return (x, y);
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
            // Same guards the f64 version carries, for the same reason: strong
            // distortion coefficients drive `df` toward zero, and `f / df` then
            // steps theta to inf or NaN. The loop has no recovery from that, and
            // `tan()` of a NaN is NaN, so the caller silently receives a NaN ray
            // where the f64 path returns the input unchanged.
            if !df.is_finite() || df.abs() < 1e-6 || !f.is_finite() {
                break;
            }
            let step = f / df;
            if !step.is_finite() {
                break;
            }
            theta -= step;
        }

        // A non-finite theta would poison tan() and the returned ray, so leave
        // the input unchanged rather than emitting NaN.
        if !theta.is_finite() {
            return (x, y);
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

#[cfg(test)]
mod fisheye_f32_guards {
    use super::*;

    /// The f32 fisheye must not emit NaN where the f64 version returns its input.
    ///
    /// Strong distortion coefficients drive `df` toward zero, so the Newton step
    /// `f / df` becomes inf or NaN. The f32 version had no guard and fed that
    /// straight into `tan()`, so the caller silently received a NaN ray; the f64
    /// version has guarded this since it was written and returns the input
    /// unchanged.
    #[test]
    fn extreme_coefficients_do_not_produce_nan() {
        // df = 1 + 3*k1*theta^2 reaches zero at theta^2 = -1/(3*k1). For
        // k1 = -1 that is theta = 1/sqrt(3) = 0.5774, and `remove` starts Newton
        // from theta = r_d, so a point at exactly that radius puts the very first
        // step on the singularity and `f / df` is inf. A blunter choice of
        // coefficients does not exercise this: a large k1 puts the singularity at
        // a radius a typical pixel never reaches, which is why the first version
        // of this test passed against the unguarded code.
        let d = FisheyeDistortionF32 {
            k1: -1.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        };
        let r = 1.0f32 / 3.0f32.sqrt();
        for &(x, y) in &[(r, 0.0f32), (0.0, r), (r * 0.5, r * 0.5), (-r, r * 0.5)] {
            let (ox, oy) = d.remove(x, y);
            assert!(
                ox.is_finite() && oy.is_finite(),
                "remove(({x}, {y})) returned ({ox}, {oy}); the f32 path lost the \\
                 guards the f64 path has"
            );
        }
    }

    /// The two precisions must agree on a well-conditioned model.
    #[test]
    fn f32_and_f64_agree_on_ordinary_coefficients() {
        let k1 = 0.02f32;
        let d32 = FisheyeDistortionF32 {
            k1,
            k2: 0.001,
            k3: 0.0,
            k4: 0.0,
        };
        let d64 = FisheyeDistortion {
            k1: k1 as f64,
            k2: 0.001,
            k3: 0.0,
            k4: 0.0,
        };
        for &(x, y) in &[(0.3f32, 0.4f32), (-0.2, 0.5), (0.6, -0.1)] {
            let (ax, ay) = d32.remove(x, y);
            let (bx, by) = d64.remove(x as f64, y as f64);
            assert!(
                (ax as f64 - bx).abs() < 1e-3 && (ay as f64 - by).abs() < 1e-3,
                "remove(({x}, {y})): f32 gave ({ax}, {ay}), f64 gave ({bx}, {by})"
            );
        }
    }
}
