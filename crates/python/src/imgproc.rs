use cv_core::tensor::{CpuTensor, TensorShape};
use pyo3::prelude::*;

/// Inpaint masked regions using the Telea Fast Marching Method.
#[pyfunction]
fn inpaint_telea(
    py: Python<'_>,
    image: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    mask: Vec<u8>,
    radius: f32,
) -> PyResult<Vec<f32>> {
    let img_tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mask_tensor = CpuTensor::<u8>::from_vec(mask, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Release the GIL across the compute-heavy section so other Python
    // threads keep running.
    let result = py
        .allow_threads(|| cv_photo::inpaint_telea(&img_tensor, &mask_tensor, radius))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Inpaint masked regions using Navier-Stokes PDE diffusion.
#[pyfunction]
#[pyo3(signature = (image, channels, height, width, mask, radius=3.0, iterations=100))]
fn inpaint_ns(
    py: Python<'_>,
    image: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    mask: Vec<u8>,
    radius: f32,
    iterations: u32,
) -> PyResult<Vec<f32>> {
    let img_tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mask_tensor = CpuTensor::<u8>::from_vec(mask, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = py
        .allow_threads(|| cv_photo::inpaint_ns(&img_tensor, &mask_tensor, radius, iterations))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Reinhard tone mapping: maps HDR radiance to LDR [0,1].
#[pyfunction]
fn tonemap_reinhard(
    py: Python<'_>,
    hdr_data: Vec<f64>,
    channels: usize,
    height: usize,
    width: usize,
    gamma: f64,
) -> PyResult<Vec<f32>> {
    let tensor = CpuTensor::<f64>::from_vec(hdr_data, TensorShape::new(channels, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = py
        .allow_threads(|| cv_photo::tonemap_reinhard(&tensor, gamma))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Mertens exposure fusion: fuse multiple LDR images into one.
#[pyfunction]
fn merge_mertens(
    py: Python<'_>,
    images: Vec<Vec<f32>>,
    channels: usize,
    height: usize,
    width: usize,
) -> PyResult<Vec<f32>> {
    let tensors: Result<Vec<CpuTensor<f32>>, _> = images
        .into_iter()
        .map(|data| CpuTensor::<f32>::from_vec(data, TensorShape::new(channels, height, width)))
        .collect();
    let tensors =
        tensors.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = py
        .allow_threads(|| cv_photo::merge_mertens(&tensors))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Non-local means denoising for single-channel images.
#[pyfunction]
fn fast_nl_means_denoising(
    py: Python<'_>,
    image: Vec<f32>,
    height: usize,
    width: usize,
    h: f32,
    template_window: usize,
    search_window: usize,
) -> PyResult<Vec<f32>> {
    let tensor = CpuTensor::<f32>::from_vec(image, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    // O(H·W·template²·search²) — seconds per megapixel; must not hold the GIL.
    let result = py
        .allow_threads(|| {
            cv_photo::fast_nl_means_denoising(&tensor, h, template_window, search_window)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute the distance transform of a binary image.
#[pyfunction]
fn distance_transform(
    py: Python<'_>,
    binary: Vec<f32>,
    height: usize,
    width: usize,
    dist_type: &str,
) -> PyResult<Vec<f32>> {
    let dt = match dist_type {
        "l1" => cv_imgproc::distance_transform::DistanceType::L1,
        "l2" => cv_imgproc::distance_transform::DistanceType::L2,
        "chessboard" => cv_imgproc::distance_transform::DistanceType::Chessboard,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown distance type '{}'. Use 'l1', 'l2', or 'chessboard'.",
                other
            )));
        }
    };
    let tensor = CpuTensor::<f32>::from_vec(binary, TensorShape::new(1, height, width))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result = py
        .allow_threads(|| cv_imgproc::distance_transform::distance_transform(&tensor, dt))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    result
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inpaint_telea, m)?)?;
    m.add_function(wrap_pyfunction!(inpaint_ns, m)?)?;
    m.add_function(wrap_pyfunction!(tonemap_reinhard, m)?)?;
    m.add_function(wrap_pyfunction!(merge_mertens, m)?)?;
    m.add_function(wrap_pyfunction!(fast_nl_means_denoising, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, m)?)?;
    Ok(())
}
