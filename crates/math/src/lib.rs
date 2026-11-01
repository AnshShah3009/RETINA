#![forbid(unsafe_code)]
//! Scientific Computing (scipy equivalents)
//!
//! This crate provides scientific computing functions equivalent to Python's scipy:
//! - [`integrate`]: Numerical integration (quad, simpson, trapezoid)
//! - [`special`]: Special functions (erf, gamma, bessel)
//! - [`geometry`]: Geometric operations (Shapely equivalents)
//! - [`point_cloud`]: Point cloud utilities
//! - [`signal`]: Signal processing (windows, filters, spectral analysis, resampling)
//! - [`jit`]: JIT vectorization helpers
//!
//! ## Example: Numerical Integration
//!
//! ```rust
//! use cv_math::integrate::quad;
//!
//! // Integrate f(x) = x^2 from 0 to 1
//! let (result, _err) = quad(|x| x * x, 0.0, 1.0).unwrap();
//! println!("Integral: {}", result);  // 1/3 to ~1e-16
//! ```
//!
//! ## Example: Special Functions
//!
//! ```rust
//! use cv_math::special::{erf, gamma, bessel_j0};
//!
//! let e = erf(1.0);
//! let g = gamma(5.0);  // 4! = 24
//! let j0 = bessel_j0(1.0);
//! ```
//!
//! ## Statistical Functions
//!
//! - [`mean`] -- Arithmetic mean
//! - [`std`] -- Standard deviation
//! - [`lerp`] -- Linear interpolation
//! - [`Interp1d`] -- 1D linear interpolation
//! - `fft` -- FFT (1D/2D forward, inverse, real-valued, PSD)
//! - `interpolate` -- Interpolation (linear, cubic spline, Akima, 2D bilinear)
//! - `stats` -- Statistics (descriptive, correlation, t-tests, distributions, regression)

pub mod geometry;
pub mod integrate;
pub mod interpolate;
pub mod jit;
pub mod linalg;
pub mod sparse;
pub mod spatial;
pub mod special;
pub mod stats;

pub type Error = cv_core::Error;
pub type Result<T> = cv_core::Result<T>;

pub use geometry::*;
pub use integrate::*;
pub use jit::*;
pub use special::*;

pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum: f64 = data.iter().sum();
    Some(sum / data.len() as f64)
}

pub fn std(data: &[f64]) -> Option<f64> {
    let m = mean(data)?;
    let variance = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    Some(variance.sqrt())
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

pub struct Interp1d {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Interp1d {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Option<Self> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }
        Some(Self { x, y })
    }

    pub fn call(&self, x_val: f64) -> f64 {
        if self.x.is_empty() || self.y.is_empty() || x_val.is_nan() {
            return f64::NAN;
        }

        // A single knot describes a constant series: there is no interval to
        // interpolate over, so the only value on offer is y[0]. Bailing out
        // here is also what keeps the `len - 2` clamp below from underflowing.
        // A NaN knot reaches this point because every clamp comparison against
        // NaN is false, so a one-knot NaN series used to fall through to
        // `1usize - 2` and panic with "attempt to subtract with overflow" (and,
        // in release, wrap to usize::MAX and index `self.x[idx + 1]` out of
        // bounds). The knot's position cannot be known, but a lone sample is
        // still a well-defined constant.
        let n = self.x.len();
        if n == 1 {
            return self.y[0];
        }

        // `interpolate::interp1d_linear` rejects a non-increasing knot sequence
        // outright; `call` returns a bare f64 and cannot report an error, so it
        // reads the orientation from the data instead of assuming ascending.
        // Previously a descending series was searched as if it were ascending:
        // the bracketing interval was wrong and the blend used the wrong pair
        // of knots, so [1.0, 0.0] -> [0.0, 10.0] at 0.5 returned 10 instead of
        // 5. A NaN endpoint leaves the axis with no orientation at all and no
        // meaningful blend, so it answers NaN.
        let ascending = match self.x[0].partial_cmp(&self.x[n - 1]) {
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => true,
            Some(std::cmp::Ordering::Greater) => false,
            None => return f64::NAN,
        };

        // Clamping respects the caller's orientation: x[0] is the low end when
        // ascending and the high end when descending, and y[0] belongs to it.
        if ascending {
            if x_val <= self.x[0] {
                return self.y[0];
            }
            if x_val >= self.x[n - 1] {
                return self.y[n - 1];
            }
        } else {
            if x_val >= self.x[0] {
                return self.y[0];
            }
            if x_val <= self.x[n - 1] {
                return self.y[n - 1];
            }
        }

        // NaN queries previously panicked here: binary_search_by returned
        // Err(0) and `pos - 1` underflowed usize to usize::MAX, so the
        // `self.x[idx + 1]` below went out of bounds. Guarded above, and the
        // result is clamped so the index can never run off either end.
        //
        // Mirroring both the knots and the query for a descending series keeps
        // one ascending search for both orientations: negation reverses the
        // order, so the same comparison and the same index clamping apply.
        let sign = if ascending { 1.0 } else { -1.0 };
        let query = sign * x_val;
        let idx = self
            .x
            .binary_search_by(|v| {
                let knot = sign * v;
                knot.partial_cmp(&query).unwrap_or_else(|| {
                    if knot.is_nan() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
            })
            .unwrap_or_else(|pos| pos.saturating_sub(1))
            .min(n - 2);

        let denominator = self.x[idx + 1] - self.x[idx];
        // Real linear blend. Forcing t = 0.5 whenever the knots were closer
        // than 1e-10 made (0, 0) and (1e-12, 100) queried at 1e-15 return 50
        // instead of 0.1, which is what `interp1d_linear` returns.
        let t = (x_val - self.x[idx]) / denominator;
        lerp(self.y[idx], self.y[idx + 1], t)
    }
}
