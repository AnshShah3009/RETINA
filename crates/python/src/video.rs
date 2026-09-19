use crate::runtime::PyWorkloadHint;
use cv_core::CameraIntrinsics;
use cv_runtime::orchestrator::scheduler;
use cv_slam::Slam as RustSlam;
use cv_videoio::{open_video, VideoCapture};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct PyVideoCapture {
    inner: Box<dyn VideoCapture>,
}

#[pymethods]
impl PyVideoCapture {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let cap = open_video(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: cap })
    }

    pub fn read(&mut self) -> PyResult<Option<Vec<u8>>> {
        match self.inner.read() {
            Ok(img) => Ok(Some(img.into_vec())),
            Err(_) => Ok(None),
        }
    }

    pub fn is_opened(&self) -> bool {
        self.inner.is_opened()
    }
}

#[pyclass]
pub struct PySlam {
    inner: RustSlam,
    _group: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PySlam {
    #[new]
    #[pyo3(signature = (width, height, fx=None, fy=None, cx=None, cy=None, hint=PyWorkloadHint::Default))]
    pub fn new(
        width: u32,
        height: u32,
        fx: Option<f64>,
        fy: Option<f64>,
        cx: Option<f64>,
        cy: Option<f64>,
        hint: PyWorkloadHint,
    ) -> PyResult<Self> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let group = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let intrinsics = match (fx, fy, cx, cy) {
            (Some(fx_val), Some(fy_val), Some(cx_val), Some(cy_val)) => {
                CameraIntrinsics::new(fx_val, fy_val, cx_val, cy_val, width, height)
            }
            _ => {
                let default_fx = fx.unwrap_or(width as f64 * 0.866);
                let default_fy = fy.unwrap_or(default_fx);
                let default_cx = cx.unwrap_or(width as f64 / 2.0);
                let default_cy = cy.unwrap_or(height as f64 / 2.0);
                CameraIntrinsics::new(
                    default_fx, default_fy, default_cx, default_cy, width, height,
                )
            }
        };
        Ok(Self {
            inner: RustSlam::new(group.clone(), intrinsics),
            _group: group,
        })
    }

    pub fn process_frame(
        &mut self,
        image: Vec<u8>,
        width: usize,
        height: usize,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let (pose, tracked) = self
            .inner
            .process_image(&gray)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        let mat = pose.matrix();
        let flat_mat: Vec<f32> = mat.iter().map(|&v| v as f32).collect();

        Ok((flat_mat, tracked))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PySlam>()?;
    Ok(())
}
