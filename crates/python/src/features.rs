use crate::core::PyKeyPoints;
use cv_features::{fast, gftt, harris};
use pyo3::prelude::*;

#[pyclass]
pub struct PyFeatureDetector;

#[pymethods]
impl PyFeatureDetector {
    #[staticmethod]
    pub fn fast_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        threshold: u8,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = fast::fast_detect(&gray, threshold, 1000);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn harris_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        k: f64,
        threshold: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::harris_detect(&gray, 3, 3, k, threshold);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn gftt_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        max_corners: usize,
        quality_level: f64,
        min_distance: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = gftt::gftt_detect(&gray, max_corners, quality_level, min_distance);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn shi_tomasi_detect(
        image: Vec<u8>,
        width: usize,
        height: usize,
        max_corners: usize,
        quality_level: f64,
    ) -> PyResult<PyKeyPoints> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::shi_tomasi_detect(&gray, max_corners, quality_level, 1.0);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFeatureDetector>()?;
    Ok(())
}
