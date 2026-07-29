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
