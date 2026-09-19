use crate::runtime::PyWorkloadHint;
use cv_dnn::DnnNet;
use cv_runtime::orchestrator::scheduler;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct PyDnnNet {
    inner: DnnNet,
    runner: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PyDnnNet {
    #[new]
    pub fn new(path: &str, hint: PyWorkloadHint) -> PyResult<Self> {
        let inner = DnnNet::load(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let runner = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner, runner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        image: Vec<u8>,
        width: usize,
        height: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let gray =
            image::GrayImage::from_raw(width as u32, height as u32, image).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image dimensions")
            })?;

        let dyn_img = image::DynamicImage::ImageLuma8(gray);

        let input_tensor = self
            .inner
            .preprocess(&dyn_img, &self.runner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Full DNN inference — release the GIL for the compute.
        let outputs = py
            .allow_threads(|| self.inner.forward(&input_tensor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut py_outputs = Vec::with_capacity(outputs.len());
        for (i, tensor) in outputs.iter().enumerate() {
            let slice = tensor.as_slice().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to read output tensor {}: {}",
                    i, e
                ))
            })?;
            py_outputs.push(slice.to_vec());
        }

        Ok(py_outputs)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDnnNet>()?;
    Ok(())
}
