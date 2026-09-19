use nalgebra::{Point3, Vector3};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn pts_from_py(points: &[(f32, f32, f32)]) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|&(x, y, z)| Point3::new(x, y, z))
        .collect()
}

pub fn normals_to_py(normals: Vec<Vector3<f32>>) -> Vec<(f32, f32, f32)> {
    normals.into_iter().map(|n| (n.x, n.y, n.z)).collect()
}

/// Convert a `catch_unwind` panic payload into a human-readable string.
pub fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

pub fn ndarray_to_points(pts: &PyReadonlyArray2<f32>) -> Vec<Point3<f32>> {
    let arr = pts.as_array();
    let n = arr.shape()[0];
    (0..n)
        .map(|i| Point3::new(arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
        .collect()
}

pub fn normals_to_ndarray<'py>(
    py: Python<'py>,
    normals: Vec<Vector3<f32>>,
) -> Bound<'py, PyArray2<f32>> {
    let n = normals.len();
    let mut arr = Array2::<f32>::zeros((n, 3));
    for (i, v) in normals.iter().enumerate() {
        arr[[i, 0]] = v.x;
        arr[[i, 1]] = v.y;
        arr[[i, 2]] = v.z;
    }
    arr.into_pyarray_bound(py)
}
