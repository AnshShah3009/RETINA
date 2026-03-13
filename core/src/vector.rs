use crate::float::Float;

/// A generic trait for N-dimensional vectors.
///
/// This trait defines common operations for vectors such as dot product, norm calculation,
/// and normalization. It is generic over the scalar type `T` which must implement `Float`.
pub trait Vector<T: Float>: Sized + Clone {
    /// Returns the number of dimensions of the vector.
    fn len(&self) -> usize;
    
    /// Returns `true` if the vector has zero dimensions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Computes the dot product between this vector and another vector of the same type.
    ///
    /// The dot product is the sum of the products of the corresponding entries of the two
    /// sequences of numbers.
    fn dot(&self, other: &Self) -> T;
    
    /// Computes the L2 norm (magnitude) of the vector.
    ///
    /// The L2 norm is defined as the square root of the dot product of the vector with itself.
    fn norm(&self) -> T {
        self.dot(self).sqrt()
    }
    
    /// Returns a normalized version of the vector (a unit vector with the same direction).
    ///
    /// If the vector's norm is zero, the behavior depends on the implementation (usually returns the same vector).
    fn normalize(&self) -> Self;
    
    /// Computes the cross product of this 3D vector with another 3D vector.
    ///
    /// This operation is only defined for 3D vectors.
    fn cross_3d(&self, other: &Self) -> Self;
}

/// A trait for 3D points in Euclidean space.
///
/// Unlike vectors, points represent a location in space rather than a direction and magnitude.
pub trait Point3D<T: Float>: Sized + Clone {
    /// Returns the X coordinate of the point.
    fn x(&self) -> T;
    /// Returns the Y coordinate of the point.
    fn y(&self) -> T;
    /// Returns the Z coordinate of the point.
    fn z(&self) -> T;
    
    /// Computes the Euclidean distance between this point and another point.
    fn distance_to(&self, other: &Self) -> T;
    
    /// Transforms the point by a 4x4 transformation matrix.
    ///
    /// This usually involves converting the point to homogeneous coordinates [x, y, z, 1],
    /// multiplying by the matrix, and then dividing by the w-component.
    fn transform<M: Matrix4x4<T>>(&self, matrix: &M) -> Self;
}

/// A trait for 3x3 matrices.
///
/// Commonly used for rotations, scaling, and camera intrinsics.
pub trait Matrix3x3<T: Float>: Sized + Clone {
    /// Returns the 3x3 identity matrix.
    fn identity() -> Self;
    
    /// Multiplies the matrix by a 3D vector represented as an array.
    ///
    /// Returns the resulting 3D vector as an array.
    fn mul_vector(&self, v: &[T; 3]) -> [T; 3];
    
    /// Multiplies this matrix by another 3x3 matrix.
    ///
    /// Returns the resulting 3x3 matrix.
    fn mul_matrix(&self, other: &Self) -> Self;
    
    /// Computes the determinant of the 3x3 matrix.
    fn determinant(&self) -> T;
    
    /// Computes the inverse of the matrix, if it is invertible.
    ///
    /// Returns `Some(inverse)` if the matrix is invertible, or `None` if it is singular.
    fn inverse(&self) -> Option<Self>;
}

/// A trait for 4x4 matrices.
///
/// Commonly used for 3D transformations including translation, rotation, and projection.
pub trait Matrix4x4<T: Float>: Sized + Clone {
    /// Returns the 4x4 identity matrix.
    fn identity() -> Self;
    
    /// Multiplies the matrix by a 4D vector represented as an array.
    ///
    /// Returns the resulting 4D vector as an array.
    fn mul_vector(&self, v: &[T; 4]) -> [T; 4];
    
    /// Multiplies this matrix by another 4x4 matrix.
    ///
    /// Returns the resulting 4x4 matrix.
    fn mul_matrix(&self, other: &Self) -> Self;
    
    /// Computes the inverse of the matrix, if it is invertible.
    ///
    /// Returns `Some(inverse)` if the matrix is invertible, or `None` if it is singular.
    fn inverse(&self) -> Option<Self>;
}
