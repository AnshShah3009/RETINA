use crate::float::Float;
use crate::vector::{Matrix3x3, Matrix4x4, Point3D, Vector};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Wrapper for a 3D vector, compatible with nalgebra.
///
/// This struct provides a bridge between the internal `Vector` trait and nalgebra's `Vector3`.
/// It supports standard mathematical operators and conversions to/from nalgebra types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3Wrapper<T: Float> {
    /// The underlying data as a 3D array.
    pub data: [T; 3],
}

impl<T: Float> Vector3Wrapper<T> {
    /// Creates a new `Vector3Wrapper` from x, y, z components.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { data: [x, y, z] }
    }

    /// Returns the X component of the vector.
    pub fn x(&self) -> T {
        self.data[0]
    }
    /// Returns the Y component of the vector.
    pub fn y(&self) -> T {
        self.data[1]
    }
    /// Returns the Z component of the vector.
    pub fn z(&self) -> T {
        self.data[2]
    }

    /// Converts the vector to a nalgebra `Vector3<f32>`.
    pub fn to_na_f32(&self) -> Vector3<f32> {
        Vector3::new(
            self.data[0].to_f32(),
            self.data[1].to_f32(),
            self.data[2].to_f32(),
        )
    }

    /// Converts the vector to a nalgebra `Vector3<f64>`.
    pub fn to_na_f64(&self) -> Vector3<f64> {
        Vector3::new(
            self.data[0].to_f64(),
            self.data[1].to_f64(),
            self.data[2].to_f64(),
        )
    }

    /// Creates a `Vector3Wrapper` from a nalgebra `Vector3<f32>`.
    pub fn from_na_f32(v: &Vector3<f32>) -> Self {
        Self::new(T::from_f32(v.x), T::from_f32(v.y), T::from_f32(v.z))
    }

    /// Creates a `Vector3Wrapper` from a nalgebra `Vector3<f64>`.
    pub fn from_na_f64(v: &Vector3<f64>) -> Self {
        Self::new(T::from_f64(v.x), T::from_f64(v.y), T::from_f64(v.z))
    }
}

impl<T: Float> Vector<T> for Vector3Wrapper<T> {
    fn len(&self) -> usize {
        3
    }

    fn dot(&self, other: &Self) -> T {
        self.data[0] * other.data[0] + self.data[1] * other.data[1] + self.data[2] * other.data[2]
    }

    fn normalize(&self) -> Self {
        let n = self.norm();
        if n > T::EPSILON {
            Self::new(self.data[0] / n, self.data[1] / n, self.data[2] / n)
        } else {
            // Return zero vector if it's already nearly zero
            Self::new(T::ZERO, T::ZERO, T::ZERO)
        }
    }

    fn cross_3d(&self, other: &Self) -> Self {
        Self::new(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        )
    }
}

// Operator implementations for Vector3Wrapper

impl<T: Float> Add for Vector3Wrapper<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(
            self.data[0] + other.data[0],
            self.data[1] + other.data[1],
            self.data[2] + other.data[2],
        )
    }
}

impl<T: Float> Sub for Vector3Wrapper<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(
            self.data[0] - other.data[0],
            self.data[1] - other.data[1],
            self.data[2] - other.data[2],
        )
    }
}

impl<T: Float> Mul<T> for Vector3Wrapper<T> {
    type Output = Self;
    fn mul(self, scalar: T) -> Self {
        Self::new(
            self.data[0] * scalar,
            self.data[1] * scalar,
            self.data[2] * scalar,
        )
    }
}

impl<T: Float> Div<T> for Vector3Wrapper<T> {
    type Output = Self;
    fn div(self, scalar: T) -> Self {
        if scalar.abs() < T::EPSILON {
            return Self::new(T::ZERO, T::ZERO, T::ZERO);
        }
        Self::new(
            self.data[0] / scalar,
            self.data[1] / scalar,
            self.data[2] / scalar,
        )
    }
}

impl<T: Float> AddAssign for Vector3Wrapper<T> {
    fn add_assign(&mut self, other: Self) {
        self.data[0] += other.data[0];
        self.data[1] += other.data[1];
        self.data[2] += other.data[2];
    }
}

impl<T: Float> SubAssign for Vector3Wrapper<T> {
    fn sub_assign(&mut self, other: Self) {
        self.data[0] -= other.data[0];
        self.data[1] -= other.data[1];
        self.data[2] -= other.data[2];
    }
}

impl<T: Float> MulAssign<T> for Vector3Wrapper<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.data[0] *= scalar;
        self.data[1] *= scalar;
        self.data[2] *= scalar;
    }
}

impl<T: Float> DivAssign<T> for Vector3Wrapper<T> {
    fn div_assign(&mut self, scalar: T) {
        if scalar.abs() > T::EPSILON {
            self.data[0] /= scalar;
            self.data[1] /= scalar;
            self.data[2] /= scalar;
        }
    }
}

/// Wrapper for a 3D point, compatible with nalgebra.
///
/// This struct provides a bridge between the internal `Point3D` trait and nalgebra's `Point3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3Wrapper<T: Float> {
    /// The underlying data as a 3D array.
    pub data: [T; 3],
}

impl<T: Float> Point3Wrapper<T> {
    /// Creates a new `Point3Wrapper` from x, y, z coordinates.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { data: [x, y, z] }
    }

    /// Returns the X coordinate of the point.
    pub fn x(&self) -> T {
        self.data[0]
    }
    /// Returns the Y coordinate of the point.
    pub fn y(&self) -> T {
        self.data[1]
    }
    /// Returns the Z coordinate of the point.
    pub fn z(&self) -> T {
        self.data[2]
    }

    /// Converts the point to a nalgebra `Point3<f32>`.
    pub fn to_na_f32(&self) -> Point3<f32> {
        Point3::new(
            self.data[0].to_f32(),
            self.data[1].to_f32(),
            self.data[2].to_f32(),
        )
    }

    /// Converts the point to a nalgebra `Point3<f64>`.
    pub fn to_na_f64(&self) -> Point3<f64> {
        Point3::new(
            self.data[0].to_f64(),
            self.data[1].to_f64(),
            self.data[2].to_f64(),
        )
    }

    /// Creates a `Point3Wrapper` from a nalgebra `Point3<f32>`.
    pub fn from_na_f32(p: &Point3<f32>) -> Self {
        Self::new(T::from_f32(p.x), T::from_f32(p.y), T::from_f32(p.z))
    }

    /// Creates a `Point3Wrapper` from a nalgebra `Point3<f64>`.
    pub fn from_na_f64(p: &Point3<f64>) -> Self {
        Self::new(T::from_f64(p.x), T::from_f64(p.y), T::from_f64(p.z))
    }
}

impl<T: Float> Point3D<T> for Point3Wrapper<T> {
    fn x(&self) -> T {
        self.data[0]
    }
    fn y(&self) -> T {
        self.data[1]
    }
    fn z(&self) -> T {
        self.data[2]
    }

    fn distance_to(&self, other: &Self) -> T {
        let dx = self.data[0] - other.data[0];
        let dy = self.data[1] - other.data[1];
        let dz = self.data[2] - other.data[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn transform<M: Matrix4x4<T>>(&self, matrix: &M) -> Self {
        let v = [self.data[0], self.data[1], self.data[2], T::ONE];
        let res = matrix.mul_vector(&v);
        // Robustness: check for zero division in homogeneous coordinates
        if res[3].abs() > T::EPSILON {
            if res[3] == T::ONE {
                Self::new(res[0], res[1], res[2])
            } else {
                Self::new(res[0] / res[3], res[1] / res[3], res[2] / res[3])
            }
        } else {
            // If w is zero, it's a point at infinity, which we can't represent as Point3
            // Fallback to original point or zero point
            Self::new(res[0], res[1], res[2])
        }
    }
}

// Operator implementations for Point3Wrapper

impl<T: Float> Add<Vector3Wrapper<T>> for Point3Wrapper<T> {
    type Output = Self;
    fn add(self, v: Vector3Wrapper<T>) -> Self {
        Self::new(
            self.data[0] + v.data[0],
            self.data[1] + v.data[1],
            self.data[2] + v.data[2],
        )
    }
}

impl<T: Float> Sub<Vector3Wrapper<T>> for Point3Wrapper<T> {
    type Output = Self;
    fn sub(self, v: Vector3Wrapper<T>) -> Self {
        Self::new(
            self.data[0] - v.data[0],
            self.data[1] - v.data[1],
            self.data[2] - v.data[2],
        )
    }
}

impl<T: Float> Sub for Point3Wrapper<T> {
    type Output = Vector3Wrapper<T>;
    fn sub(self, other: Self) -> Vector3Wrapper<T> {
        Vector3Wrapper::new(
            self.data[0] - other.data[0],
            self.data[1] - other.data[1],
            self.data[2] - other.data[2],
        )
    }
}

/// Wrapper for a 3x3 matrix, compatible with nalgebra.
///
/// This struct provides a bridge between the internal `Matrix3x3` trait and nalgebra's `Matrix3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3Wrapper<T: Float> {
    /// The underlying data as a 3x3 array.
    pub data: [[T; 3]; 3],
}

impl<T: Float> Matrix3Wrapper<T> {
    /// Creates a new `Matrix3Wrapper` from a 3x3 array.
    pub fn new(data: [[T; 3]; 3]) -> Self {
        Self { data }
    }

    /// Converts the matrix to a nalgebra `Matrix3<f32>`.
    pub fn to_na_f32(&self) -> Matrix3<f32> {
        let mut m = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = self.data[i][j].to_f32();
            }
        }
        m
    }

    /// Converts the matrix to a nalgebra `Matrix3<f64>`.
    pub fn to_na_f64(&self) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = self.data[i][j].to_f64();
            }
        }
        m
    }

    /// Creates a `Matrix3Wrapper` from a nalgebra `Matrix3<f32>`.
    pub fn from_na_f32(m: &Matrix3<f32>) -> Self {
        let mut data = [[T::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                data[i][j] = T::from_f32(m[(i, j)]);
            }
        }
        Self { data }
    }

    /// Creates a `Matrix3Wrapper` from a nalgebra `Matrix3<f64>`.
    pub fn from_na_f64(m: &Matrix3<f64>) -> Self {
        let mut data = [[T::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                data[i][j] = T::from_f64(m[(i, j)]);
            }
        }
        Self { data }
    }
}

impl<T: Float> Matrix3x3<T> for Matrix3Wrapper<T> {
    fn identity() -> Self {
        let mut data = [[T::ZERO; 3]; 3];
        data[0][0] = T::ONE;
        data[1][1] = T::ONE;
        data[2][2] = T::ONE;
        Self { data }
    }

    fn mul_vector(&self, v: &[T; 3]) -> [T; 3] {
        let mut res = [T::ZERO; 3];
        for (i, res_i) in res.iter_mut().enumerate() {
            *res_i = self.data[i][0] * v[0] + self.data[i][1] * v[1] + self.data[i][2] * v[2];
        }
        res
    }

    fn mul_matrix(&self, other: &Self) -> Self {
        let mut res = [[T::ZERO; 3]; 3];
        for (i, res_row) in res.iter_mut().enumerate() {
            for (j, res_ij) in res_row.iter_mut().enumerate() {
                let mut sum = T::ZERO;
                for k in 0..3 {
                    sum += self.data[i][k] * other.data[k][j];
                }
                *res_ij = sum;
            }
        }
        Self { data: res }
    }

    fn determinant(&self) -> T {
        let m = self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    fn inverse(&self) -> Option<Self> {
        // Delegate to nalgebra via f64 for stability
        let m = self.to_na_f64();

        m.try_inverse().map(|inv| Self::from_na_f64(&inv))
    }
}

impl<T: Float> Mul for Matrix3Wrapper<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul_matrix(&rhs)
    }
}

/// Wrapper for a 4x4 matrix, compatible with nalgebra.
///
/// This struct provides a bridge between the internal `Matrix4x4` trait and nalgebra's `Matrix4`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix4Wrapper<T: Float> {
    /// The underlying data as a 4x4 array.
    pub data: [[T; 4]; 4],
}

impl<T: Float> Matrix4Wrapper<T> {
    /// Creates a new `Matrix4Wrapper` from a 4x4 array.
    pub fn new(data: [[T; 4]; 4]) -> Self {
        Self { data }
    }

    /// Converts the matrix to a nalgebra `Matrix4<f32>`.
    pub fn to_na_f32(&self) -> Matrix4<f32> {
        let mut m = Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = self.data[i][j].to_f32();
            }
        }
        m
    }

    /// Converts the matrix to a nalgebra `Matrix4<f64>`.
    pub fn to_na_f64(&self) -> Matrix4<f64> {
        let mut m = Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = self.data[i][j].to_f64();
            }
        }
        m
    }

    /// Creates a `Matrix4Wrapper` from a nalgebra `Matrix4<f32>`.
    pub fn from_na_f32(m: &Matrix4<f32>) -> Self {
        let mut data = [[T::ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                data[i][j] = T::from_f32(m[(i, j)]);
            }
        }
        Self { data }
    }

    /// Creates a `Matrix4Wrapper` from a nalgebra `Matrix4<f64>`.
    pub fn from_na_f64(m: &Matrix4<f64>) -> Self {
        let mut data = [[T::ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                data[i][j] = T::from_f64(m[(i, j)]);
            }
        }
        Self { data }
    }
}

impl<T: Float> Matrix4x4<T> for Matrix4Wrapper<T> {
    fn identity() -> Self {
        let mut data = [[T::ZERO; 4]; 4];
        data[0][0] = T::ONE;
        data[1][1] = T::ONE;
        data[2][2] = T::ONE;
        data[3][3] = T::ONE;
        Self { data }
    }

    fn mul_vector(&self, v: &[T; 4]) -> [T; 4] {
        let mut res = [T::ZERO; 4];
        for (i, res_i) in res.iter_mut().enumerate() {
            *res_i = self.data[i][0] * v[0]
                + self.data[i][1] * v[1]
                + self.data[i][2] * v[2]
                + self.data[i][3] * v[3];
        }
        res
    }

    fn mul_matrix(&self, other: &Self) -> Self {
        let mut res = [[T::ZERO; 4]; 4];
        for (i, res_row) in res.iter_mut().enumerate() {
            for (j, res_ij) in res_row.iter_mut().enumerate() {
                let mut sum = T::ZERO;
                for k in 0..4 {
                    sum += self.data[i][k] * other.data[k][j];
                }
                *res_ij = sum;
            }
        }
        Self { data: res }
    }

    fn inverse(&self) -> Option<Self> {
        // Delegate to nalgebra via f64 for stability
        let m = self.to_na_f64();

        m.try_inverse().map(|inv| Self::from_na_f64(&inv))
    }
}

impl<T: Float> Mul for Matrix4Wrapper<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul_matrix(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;

    #[test]
    fn test_vector3_dot() {
        let v1 = Vector3Wrapper::new(1.0f32, 2.0, 3.0);
        let v2 = Vector3Wrapper::new(4.0f32, 5.0, 6.0);
        let dot = v1.dot(&v2);
        assert_eq!(dot, 32.0);
    }

    #[test]
    fn test_matrix3_inverse() {
        let mut data = [[0.0f32; 3]; 3];
        data[0][0] = 2.0;
        data[1][1] = 2.0;
        data[2][2] = 2.0;
        let m = Matrix3Wrapper::new(data);
        let inv = m.inverse().unwrap();
        assert_eq!(inv.data[0][0], 0.5);
        assert_eq!(inv.data[1][1], 0.5);
        assert_eq!(inv.data[2][2], 0.5);
    }

    #[test]
    fn test_vector3_ops() {
        let v1 = Vector3Wrapper::new(1.0f32, 2.0, 3.0);
        let v2 = Vector3Wrapper::new(4.0f32, 5.0, 6.0);

        let v3 = v1 + v2;
        assert_eq!(v3, Vector3Wrapper::new(5.0, 7.0, 9.0));

        let mut v4 = v1 * 2.0;
        assert_eq!(v4, Vector3Wrapper::new(2.0, 4.0, 6.0));

        v4 /= 2.0;
        assert_eq!(v4, v1);
    }
}
