use std::fmt::{Debug, Display};

/// A trait for floating-point numbers, providing a common interface for `f32`, `f64`,
/// and optionally half-precision types (`f16`, `bf16`).
///
/// This trait abstracts over standard mathematical operations and constants required
/// for computer vision algorithms.
pub trait Float:
    Sized
    + Copy
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + Send
    + Sync
    + num_traits::Zero
    + num_traits::One
    + num_traits::NumCast
{
    /// The additive identity (0.0).
    const ZERO: Self;
    /// The multiplicative identity (1.0).
    const ONE: Self;
    /// The mathematical constant PI.
    const PI: Self;
    /// The difference between 1.0 and the next larger representable number.
    const EPSILON: Self;

    /// Converts an `f32` to `Self`.
    fn from_f32(val: f32) -> Self;
    /// Converts `Self` to `f32`.
    fn to_f32(self) -> f32;
    /// Converts an `f64` to `Self`.
    fn from_f64(val: f64) -> Self;
    /// Converts `Self` to `f64`.
    fn to_f64(self) -> f64;

    /// Returns the absolute value of `self`.
    fn abs(self) -> Self;
    /// Returns the square root of `self`.
    fn sqrt(self) -> Self;
    /// Returns the reciprocal (1/x) of `self`.
    fn recip(self) -> Self;
    /// Returns the reciprocal square root (1/sqrt(x)) of `self`.
    fn rsqrt(self) -> Self;
    /// Returns `self` raised to the power of `n`.
    fn powf(self, n: Self) -> Self;
    /// Returns `e^(self)`.
    fn exp(self) -> Self;
    /// Returns the natural logarithm of `self`.
    fn ln(self) -> Self;

    /// Returns the sine of `self` (in radians).
    fn sin(self) -> Self;
    /// Returns the cosine of `self` (in radians).
    fn cos(self) -> Self;
    /// Returns the tangent of `self` (in radians).
    fn tan(self) -> Self;
    /// Returns the arcsine of `self` (in radians).
    fn asin(self) -> Self;
    /// Returns the arccosine of `self` (in radians).
    fn acos(self) -> Self;
    /// Returns the four-quadrant arctangent of `self` (y) and `other` (x) in radians.
    fn atan2(self, other: Self) -> Self;

    /// Returns the maximum of two values.
    fn max(self, other: Self) -> Self;
    /// Returns the minimum of two values.
    fn min(self, other: Self) -> Self;
    /// Restricts a value to a certain interval.
    fn clamp(self, min: Self, max: Self) -> Self;

    /// Returns `true` if this value is NaN.
    fn is_nan(self) -> bool;
    /// Returns `true` if this value is positive infinity or negative infinity.
    fn is_infinite(self) -> bool;
    /// Returns `true` if this value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

macro_rules! impl_float {
    ($t:ty, $pi:expr, $epsilon:expr) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const PI: Self = $pi;
            const EPSILON: Self = $epsilon;

            #[inline]
            fn from_f32(val: f32) -> Self {
                val as $t
            }

            #[inline]
            fn to_f32(self) -> f32 {
                self as f32
            }

            #[inline]
            fn from_f64(val: f64) -> Self {
                val as $t
            }

            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }

            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }

            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline]
            fn recip(self) -> Self {
                self.recip()
            }

            #[inline]
            fn rsqrt(self) -> Self {
                Self::ONE / self.sqrt()
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                self.powf(n)
            }

            #[inline]
            fn exp(self) -> Self {
                self.exp()
            }

            #[inline]
            fn ln(self) -> Self {
                self.ln()
            }

            #[inline]
            fn sin(self) -> Self {
                self.sin()
            }

            #[inline]
            fn cos(self) -> Self {
                self.cos()
            }

            #[inline]
            fn tan(self) -> Self {
                self.tan()
            }

            #[inline]
            fn asin(self) -> Self {
                self.asin()
            }

            #[inline]
            fn acos(self) -> Self {
                self.acos()
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                self.clamp(min, max)
            }

            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }

            #[inline]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            #[inline]
            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}

impl_float!(f32, std::f32::consts::PI, std::f32::EPSILON);
impl_float!(f64, std::f64::consts::PI, std::f64::EPSILON);

macro_rules! impl_half_float {
    ($t:ty) => {
        #[cfg(feature = "half-precision")]
        impl Float for $t {
            const ZERO: Self = <$t>::ZERO;
            const ONE: Self = <$t>::ONE;
            const PI: Self = <$t>::from_f32_const(std::f32::consts::PI);
            const EPSILON: Self = <$t>::EPSILON;

            #[inline]
            fn from_f32(val: f32) -> Self {
                <$t>::from_f32(val)
            }

            #[inline]
            fn to_f32(self) -> f32 {
                <$t>::to_f32(self)
            }

            #[inline]
            fn from_f64(val: f64) -> Self {
                <$t>::from_f64(val)
            }

            #[inline]
            fn to_f64(self) -> f64 {
                <$t>::to_f64(self)
            }

            #[inline]
            fn abs(self) -> Self {
                Self::from_f32(self.to_f32().abs())
            }

            #[inline]
            fn sqrt(self) -> Self {
                Self::from_f32(self.to_f32().sqrt())
            }

            #[inline]
            fn recip(self) -> Self {
                Self::from_f32(self.to_f32().recip())
            }

            #[inline]
            fn rsqrt(self) -> Self {
                Self::from_f32(1.0 / self.to_f32().sqrt())
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                Self::from_f32(self.to_f32().powf(n.to_f32()))
            }

            #[inline]
            fn exp(self) -> Self {
                Self::from_f32(self.to_f32().exp())
            }

            #[inline]
            fn ln(self) -> Self {
                Self::from_f32(self.to_f32().ln())
            }

            #[inline]
            fn sin(self) -> Self {
                Self::from_f32(self.to_f32().sin())
            }

            #[inline]
            fn cos(self) -> Self {
                Self::from_f32(self.to_f32().cos())
            }

            #[inline]
            fn tan(self) -> Self {
                Self::from_f32(self.to_f32().tan())
            }

            #[inline]
            fn asin(self) -> Self {
                Self::from_f32(self.to_f32().asin())
            }

            #[inline]
            fn acos(self) -> Self {
                Self::from_f32(self.to_f32().acos())
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().atan2(other.to_f32()))
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().max(other.to_f32()))
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().min(other.to_f32()))
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                Self::from_f32(self.to_f32().clamp(min.to_f32(), max.to_f32()))
            }

            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }

            #[inline]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            #[inline]
            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}

impl_half_float!(half::f16);
impl_half_float!(half::bf16);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_float_impl {
        ($t:ty, $name:ident) => {
            #[test]
            fn $name() {
                let x = <$t>::from_f64(2.0);
                let y = <$t>::from_f64(3.0);
                assert!(abs_diff_eq(x + y, <$t>::from_f64(5.0), <$t>::EPSILON));
                assert!(abs_diff_eq(x.sqrt(), <$t>::from_f64(2.0f64.sqrt()), <$t>::EPSILON));
                assert!(abs_diff_eq(x.recip(), <$t>::from_f64(0.5), <$t>::EPSILON));
                assert!(abs_diff_eq(x.rsqrt(), <$t>::from_f64(1.0 / 2.0f64.sqrt()), <$t>::EPSILON));
                
                // Constants
                assert!(abs_diff_eq(<$t>::ZERO, <$t>::from_f64(0.0), <$t>::EPSILON));
                assert!(abs_diff_eq(<$t>::ONE, <$t>::from_f64(1.0), <$t>::EPSILON));
                
                // Edge cases
                let nan = <$t>::from_f64(f64::NAN);
                let inf = <$t>::from_f64(f64::INFINITY);
                let neg_inf = <$t>::from_f64(f64::NEG_INFINITY);
                
                assert!(nan.is_nan());
                assert!(!nan.is_finite());
                
                assert!(inf.is_infinite());
                assert!(!inf.is_finite());
                
                assert!(neg_inf.is_infinite());
                assert!(!neg_inf.is_finite());
                
                assert!(<$t>::ONE.is_finite());
                assert!(!<$t>::ONE.is_nan());
                assert!(!<$t>::ONE.is_infinite());
            }
        };
    }

    fn abs_diff_eq<T: Float>(a: T, b: T, epsilon: T) -> bool {
        (a - b).abs() <= epsilon
    }

    test_float_impl!(f32, test_f32_float);
    test_float_impl!(f64, test_f64_float);

    #[cfg(feature = "half-precision")]
    test_float_impl!(half::f16, test_f16_float);
    #[cfg(feature = "half-precision")]
    test_float_impl!(half::bf16, test_bf16_float);
}
