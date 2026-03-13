//! Scientific Computing (scipy equivalents)
//!
//! This crate provides scientific computing functions equivalent to Python's scipy:
//! - [`integrate`]: Numerical integration (quad, simpson, trapezoid)
//! - [`special`]: Special functions (erf, gamma, bessel)
//! - [`geometry`]: Geometric operations (Shapely equivalents)
//! - [`point_cloud`]: Point cloud utilities
//! - [`jit`]: JIT vectorization helpers
//!
//! ## Example: Numerical Integration
//!
//! ```rust
//! use cv_scientific::integrate::quad;
//!
//! // Integrate f(x) = x^2 from 0 to 1
//! let result = quad(|x| x * x, 0.0, 1.0);
//! println!("Integral: {}", result.0);  // Should be ~0.333
//! ```
//!
//! ## Example: Special Functions
//!
//! ```rust
//! use cv_scientific::special::{erf, gamma, bessel_j0};
//!
//! let e = erf(1.0);
//! let g = gamma(5.0);  // 4! = 24
//! let j0 = bessel_j0(1.0);
//! ```
//!
//! ## Statistical Functions
//!
//! - [`mean`]: Arithmetic mean
//! - [`std`]: Standard deviation
//! - [`lerp`]: Linear interpolation
//! - [`Interp1d`]: 1D linear interpolation
//! - [`fft`]: FFT (1D/2D forward, inverse, real-valued, PSD)
//! - [`interpolate`]: Interpolation (linear, cubic spline, Akima, 2D bilinear)
//! - [`stats`]: Statistics (descriptive, correlation, t-tests, distributions, regression)

pub mod fft;
pub mod geometry;
pub mod geometry2d;
pub mod integrate;
pub mod interpolate;
pub mod jit;
pub mod matching;
pub mod multiview;
pub mod optimize;
pub mod point_cloud;
pub mod special;
pub mod stats;

pub type Error = cv_core::Error;
pub type Result<T> = cv_core::Result<T>;

pub use geometry::*;
pub use integrate::*;
pub use jit::*;
pub use matching::*;
pub use multiview::*;
pub use point_cloud::*;
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
        if self.x.is_empty() || self.y.is_empty() {
            return f64::NAN;
        }
        if x_val <= self.x[0] {
            return self.y[0];
        }
        if x_val >= *self.x.last().unwrap() {
            return *self.y.last().unwrap();
        }

        let idx = self
            .x
            .binary_search_by(|v| {
                v.partial_cmp(&x_val).unwrap_or_else(|| {
                    if v.is_nan() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
            })
            .unwrap_or_else(|pos| pos - 1);

        let denominator = self.x[idx + 1] - self.x[idx];
        let t = if denominator.abs() < 1e-10 {
            0.5
        } else {
            (x_val - self.x[idx]) / denominator
        };
        lerp(self.y[idx], self.y[idx + 1], t)
    }
}
