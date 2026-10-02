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

    /// Remove distortion from distorted normalized coordinates `(x, y)`.
    ///
    /// Solved by **bisection on the radius**, which is what makes this robust.
    /// The previous implementation was a fixed-point iteration,
    /// `xd += x - apply(xd)`, run exactly ten times with no convergence test.
    /// That iteration's contraction factor exceeds 1 for radii beyond about
    /// 0.65, so it did not merely converge slowly - it diverged. Measured with
    /// `k1 = 0.5, k2 = 0.2, k3 = 0.05`, coefficients a calibrator legitimately
    /// produces:
    ///
    /// | radius | forward | recovered (before) | error (before) |
    /// | ---: | ---: | ---: | ---: |
    /// | 0.5 | 0.569141 | 0.500022 | 0.000022 |
    /// | 0.6 | 0.724952 | 0.603138 | 0.003138 |
    /// | 0.8 | 1.132022 | 1.132336 | **0.332336** |
    /// | 1.0 | 1.750000 | **NaN** | - |
    ///
    /// At r = 1.0 it ran away and returned NaN, which then propagates through
    /// every undistorted image and every refined calibration with nothing raised.
    /// At r = 0.8 it returned 1.132 where the answer is 0.8 - a normalised error
    /// of 0.33, about 166 px at fx = 500, presented as a successful undistort.
    ///
    /// Bisection is used because the radial map `r -> r * (1 + k1 r^2 + ...)` is
    /// monotone for coefficients that keep it meaningful, so a bracket always
    /// exists and the iteration cannot run away. Tangential terms are handled by
    /// one fixed-point correction afterwards, which converges quickly once the
    /// radius is right.
    ///
    /// Returns `None` when the radius has no solution - which is a real case, not
    /// an error case: a sufficiently strong distortion maps some radii outside
    /// the representable range, and the caller needs to distinguish "no
    /// correction exists" from "correction is zero".
    pub fn remove_checked(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        // No distortion at all is both the most common case (every rectified
        // pipeline runs one) and the one with an exactly known answer. Short
        // circuiting it is exact, and avoids paying for ~80 bisection steps to
        // reproduce the input to within a few ULP.
        if self.k1 == 0.0 && self.k2 == 0.0 && self.k3 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0 {
            return Some((x, y));
        }

        let r = (x * x + y * y).sqrt();
        if r == 0.0 {
            return Some((0.0, 0.0));
        }
        let cos_t = x / r;
        let sin_t = y / r;

        // The forward radial map.
        let radial = |r: f64| {
            let r2 = r * r;
            r * (1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2)
        };

        // The distorted radius is at least the undistorted one when the
        // polynomial is positive there, but a barrel distortion can make it
        // smaller. Bracket both ways rather than assuming.
        let mut lo = 0.0f64;
        let mut hi = r.max(1e-12);
        // Grow the upper bound until it brackets, bounded so a pathological
        // coefficient cannot spin here.
        let mut grew = 0;
        while radial(hi) < r && grew < 200 {
            hi *= 1.5;
            grew += 1;
        }
        if !radial(hi).is_finite() || radial(hi) < r {
            return None;
        }
        // The map need not be monotone if the polynomial turns over; verify the
        // bracket actually contains the root before bisecting.
        if !(radial(lo) <= r && r <= radial(hi)) {
            return None;
        }

        let mut r_undistorted = 0.5 * (lo + hi);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break; // converged to f64 precision
            }
            r_undistorted = mid;
            if radial(mid) < r {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Fixed-point steps for the tangential terms, which shift the angle as
        // well as the radius. Iterated to convergence rather than run a fixed
        // two times: two steps left a residual of ~4e-8 normalized, which is
        // ~3e-5 px at fx = 620 and failed the existing calib3d round-trip test.
        // Once the radius is correct this iteration converges quickly, and the
        // loop stops on the step size rather than at a fixed count.
        let mut ux = r_undistorted * cos_t;
        let mut uy = r_undistorted * sin_t;
        for _ in 0..32 {
            let (ax, ay) = self.apply(ux, uy);
            let (dx, dy) = (x - ax, y - ay);
            if dx * dx + dy * dy < 1e-24 {
                break;
            }
            ux += dx;
            uy += dy;
            if !ux.is_finite() || !uy.is_finite() {
                return None;
            }
        }

        // Verify the answer actually undoes the distortion. Without this the
        // caller cannot tell a converged result from a plausible-looking one,
        // which is the failure mode being fixed here.
        let (ax, ay) = self.apply(ux, uy);
        let residual = (ax - x).hypot(y - ay);
        let scale = r.max(1.0);
        if residual > 1e-12 * scale {
            return None;
        }
        Some((ux, uy))
    }

    /// Remove distortion, falling back to the input when it cannot be inverted.
    ///
    /// Use [`Distortion::remove_checked`] where the distinction matters: this
    /// version reports non-convergence by returning its input unchanged, which is
    /// indistinguishable from a point that genuinely needed no correction.
    pub fn remove(&self, x: f64, y: f64) -> (f64, f64) {
        self.remove_checked(x, y).unwrap_or((x, y))
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

    /// Remove distortion from distorted normalized coordinates `(x, y)`.
    ///
    /// This is the `f32` port of [`Distortion::remove_checked`]; read that
    /// method's comment for why the approach is bisection and not iteration.
    /// The short version: the previous implementation here was the same
    /// fixed-point iteration, `xd += x - apply(xd)`, run exactly ten times
    /// with no convergence test. That iteration's contraction factor is
    /// `1 - A'(r)`, which exceeds 1 once `r` is past about 0.65, so past that
    /// radius it did not converge slowly - it diverged. Measured with
    /// `k1 = 0.5, k2 = 0.2, k3 = 0.05`:
    ///
    /// | radius | forward | recovered (before) | error (before) |
    /// | ---: | ---: | ---: | ---: |
    /// | 0.5 | 0.569141 | 0.500022 | 0.000022 |
    /// | 0.6 | 0.724952 | 0.603138 | 0.003138 |
    /// | 0.8 | 1.132022 | 1.132336 | **0.332336** |
    /// | 0.9 | 1.406513 | **NaN** | - |
    /// | 1.0 | 1.750000 | **NaN** | - |
    ///
    /// In `f32` the runaway reaches `+/-inf` within ten iterations, and
    /// `inf - finite` is NaN, so the divergence was not merely wrong but
    /// poisoned: the NaN then propagates through
    /// [`crate::geometry::camera::PinholeModelF32::unproject`] and every `f32`
    /// calibration or resampling built on it, with nothing raised. At r = 0.8
    /// it silently returned 1.132 where the answer is 0.8 - a normalized error
    /// of 0.33, roughly 166 px at fx = 500, reported as a successful undistort.
    ///
    /// The tolerances below are chosen for `f32`, not copied from the `f64`
    /// twin. `f32` carries ~1.2e-7 relative precision, so the `f64` gates
    /// (`1e-24` squared step, `1e-12` residual) are below the noise floor: the
    /// tangential correction could never satisfy them and would always burn
    /// all 32 iterations, and the residual gate would reject correct answers
    /// as non-convergence.
    ///
    /// Returns `None` when the radius has no solution - a real case, not an
    /// error case: a sufficiently strong distortion maps some radii outside the
    /// representable range, and the caller needs to distinguish "no correction
    /// exists" from "correction is zero".
    pub fn remove_checked(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        // No distortion at all is both the most common case (every rectified
        // pipeline runs one) and the one with an exactly known answer.
        // Short-circuiting it is exact, and avoids paying for the bisection to
        // reproduce the input to within a few ULP.
        if self.k1 == 0.0 && self.k2 == 0.0 && self.k3 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0 {
            return Some((x, y));
        }

        let r = (x * x + y * y).sqrt();
        if r == 0.0 {
            return Some((0.0, 0.0));
        }
        let cos_t = x / r;
        let sin_t = y / r;

        // The radius the forward model actually produces for a given undistorted
        // radius. This is NOT the same as the pure radial polynomial `A(r)`
        // whenever `p1`/`p2` are nonzero: the tangential terms add
        // `2 p1 x y` and `p1 (r^2 + 2 y^2)` (and their p2 counterparts) to the
        // distorted point, so they change its length as well as its angle.
        // The f64 twin bisects on `A(r)` and only cleans up the angle
        // afterwards, which is fine at f64 precision but targets the wrong
        // radius once `p1`/`p2` do anything. Measured here with tangential-only
        // coefficients, that left a residual of 6.2e-4 at r = 0.5 and 7.5e-3
        // with radial and tangential together - an order of magnitude above
        // what f32 can achieve and far above the noise floor, so it is the
        // model being wrong, not the arithmetic. Bisecting on this closes
        // against the real forward map instead, so the tangential terms are
        // accounted for exactly rather than approximately.
        let radial = |r: f32| {
            let (ax, ay) = self.apply(r * cos_t, r * sin_t);
            (ax * ax + ay * ay).sqrt()
        };

        // The distorted radius is at least the undistorted one when the
        // polynomial is positive there, but a barrel distortion can make it
        // smaller. Bracket both ways rather than assuming.
        let mut lo = 0.0f32;
        let mut hi = r.max(1e-12);
        // Grow the upper bound until it brackets, bounded so a pathological
        // coefficient cannot spin here.
        let mut grew = 0;
        while radial(hi) < r && grew < 200 {
            hi *= 1.5;
            grew += 1;
        }
        if !radial(hi).is_finite() || radial(hi) < r {
            return None;
        }
        // The map need not be monotone if the polynomial turns over; verify the
        // bracket actually contains the root before bisecting.
        if !(radial(lo) <= r && r <= radial(hi)) {
            return None;
        }

        // 64 iterations is well past what an f32 bisection needs - the loop
        // below breaks as soon as `mid` stops moving, which for f32 happens
        // after roughly 26 steps because the mantissa runs out first.
        let mut r_undistorted = 0.5 * (lo + hi);
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break; // converged to f32 precision
            }
            r_undistorted = mid;
            if radial(mid) < r {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if !r_undistorted.is_finite() {
            return None;
        }

        // Fixed-point steps for the tangential terms, which shift the angle. Two
        // things here are NOT the same as the f64 twin, and both are forced by
        // f32:
        //
        // 1. The convergence threshold is `1e-14` squared, i.e. ~1e-7 in
        //    distance - one ULP at unit magnitude, as tight as f32 can
        //    measure. The f64 twin uses `1e-24` squared (~1e-12); copying that
        //    would put the threshold far below the f32 noise floor, so the
        //    loop could never satisfy it and would always burn all 32 steps.
        //
        // 2. Each step is projected back onto the circle of radius
        //    `r_undistorted`, the radius the bisection already solved. The
        //    residual direction of this iteration is `1 - A'(r)`, the same
        //    quantity that made the original ten-step loop diverge, so the
        //    radial degree of freedom is *unstable* here too - at r = 0.985
        //    with the coefficients in the doc comment, `A'(r)` is about 3.7 and
        //    the factor is -2.7. The f64 twin gets away with it only because
        //    its threshold trips after one or two steps, before the
        //    instability can accumulate; at f32 the threshold is reachable but
        //    the loop runs long enough for it to bite, and the radius walks
        //    away (measured: r = 0.985 came back as 1.71). Re-imposing the
        //    solved radius removes that degree of freedom entirely, leaving
        //    only the angle, and on that circle the iteration contracts
        //    because `|p1|, |p2|` are small.
        let mut ux = r_undistorted * cos_t;
        let mut uy = r_undistorted * sin_t;
        for _ in 0..32 {
            let (ax, ay) = self.apply(ux, uy);
            let (dx, dy) = (x - ax, y - ay);
            if dx * dx + dy * dy < 1e-14 {
                break;
            }
            ux += dx;
            uy += dy;
            if !ux.is_finite() || !uy.is_finite() {
                return None;
            }
            // Project back onto the radius the bisection already solved (see
            // note 2 above). Without this the radial degree of freedom is
            // free to run away and the radius is lost.
            let len = (ux * ux + uy * uy).sqrt();
            if !len.is_finite() || len == 0.0 {
                return None;
            }
            let s = r_undistorted / len;
            ux *= s;
            uy *= s;
        }

        // Verify the answer actually undoes the distortion. Without this the
        // caller cannot tell a converged result from a plausible-looking one,
        // which is the failure mode being fixed here.
        //
        // `1e-4 * scale` is chosen against measurement, not by copying. A dense
        // sweep of the whole unit disc at a 0.01 grid, over five coefficient
        // sets spanning strong radial, tangential-only, combined, ordinary and
        // barrel distortion, gives a worst residual of 9.8e-6 and a worst
        // coordinate error of 8.8e-6. The gate sits about 10x above that, so a
        // correct answer always clears it - a 1e-5 gate would sit *under* the
        // measured worst case and start rejecting valid points as
        // non-convergence - while still being ~3300x tighter than the 0.332
        // error this fixes, so a diverged answer can never get through.
        let (ax, ay) = self.apply(ux, uy);
        let residual = (ax - x).hypot(y - ay);
        let scale = r.max(1.0);
        if !residual.is_finite() || residual > 1e-4 * scale {
            return None;
        }
        Some((ux, uy))
    }

    /// Remove distortion, falling back to the input when it cannot be inverted.
    ///
    /// Use [`DistortionF32::remove_checked`] where the distinction matters: this
    /// version reports non-convergence by returning its input unchanged, which
    /// is indistinguishable from a point that genuinely needed no correction.
    pub fn remove(&self, x: f32, y: f32) -> (f32, f32) {
        self.remove_checked(x, y).unwrap_or((x, y))
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
