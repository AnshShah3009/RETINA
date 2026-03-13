use crate::float::Float;
use nalgebra::{Matrix3, Matrix4, Point3, Vector3};

/// Allow nalgebra types to work with our Float trait
/// For f32/f64: direct delegation via to_f32/from_f32
/// For bf16/f16: convert via f32
pub fn na_point3_to_array<T: Float>(p: &Point3<f32>) -> [T; 3] {
    [T::from_f32(p.x), T::from_f32(p.y), T::from_f32(p.z)]
}

pub fn array_to_na_point3<T: Float>(arr: &[T; 3]) -> Point3<f32> {
    Point3::new(arr[0].to_f32(), arr[1].to_f32(), arr[2].to_f32())
}

pub fn na_vector3_to_array<T: Float>(v: &Vector3<f32>) -> [T; 3] {
    [T::from_f32(v.x), T::from_f32(v.y), T::from_f32(v.z)]
}

pub fn array_to_na_vector3<T: Float>(arr: &[T; 3]) -> Vector3<f32> {
    Vector3::new(arr[0].to_f32(), arr[1].to_f32(), arr[2].to_f32())
}

pub fn na_matrix3_to_array<T: Float>(m: &Matrix3<f32>) -> [[T; 3]; 3] {
    let mut data = [[T::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            data[i][j] = T::from_f32(m[(i, j)]);
        }
    }
    data
}

pub fn array_to_na_matrix3<T: Float>(arr: &[[T; 3]; 3]) -> Matrix3<f32> {
    let mut m = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = arr[i][j].to_f32();
        }
    }
    m
}

pub fn na_matrix4_to_array<T: Float>(m: &Matrix4<f32>) -> [[T; 4]; 4] {
    let mut data = [[T::ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            data[i][j] = T::from_f32(m[(i, j)]);
        }
    }
    data
}

pub fn array_to_na_matrix4<T: Float>(arr: &[[T; 4]; 4]) -> Matrix4<f32> {
    let mut m = Matrix4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = arr[i][j].to_f32();
        }
    }
    m
}
