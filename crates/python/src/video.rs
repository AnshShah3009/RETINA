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

    /// Track one grayscale frame. `depth` (optional, row-major metric depth,
    /// `width * height` values) enables RGB-D mode, where the first frame
    /// bootstraps the map. Without depth the first frame raises while it is
    /// held as the monocular initialisation reference. Frames without a
    /// measured pose (initialising, tracking lost) raise `RuntimeError`.
    #[pyo3(signature = (image, width, height, depth=None))]
    pub fn process_frame(
        &mut self,
        image: Vec<u8>,
        width: usize,
        height: usize,
        depth: Option<Vec<f32>>,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let gray = image::GrayImage::from_raw(width as u32, height as u32, image)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let result = match depth {
            Some(d) => self.inner.process_image_with_depth(&gray, &d),
            None => self.inner.process_image(&gray),
        };
        let (pose, tracked) = result.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        Ok((pose_matrix_row_major(&pose), tracked))
    }
}

/// The 4x4 world -> camera pose matrix flattened for Python, so that
/// `np.array(flat).reshape(4, 4)` is the matrix itself.
///
/// nalgebra stores matrices column-major, so `matrix().iter()` walks columns;
/// flattening that directly handed numpy the transpose (translation in the
/// bottom row). Iterate the transpose to emit rows.
fn pose_matrix_row_major(pose: &cv_core::Pose) -> Vec<f32> {
    pose.matrix()
        .transpose()
        .iter()
        .map(|&v| v as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::pose_matrix_row_major;
    use cv_core::Pose;
    use nalgebra::{UnitQuaternion, Vector3};

    /// numpy's `reshape(4, 4)` is row-major (C order): element `[r][c]` is
    /// `flat[4 * r + c]`. The translation must therefore land at indices 3, 7
    /// and 11, and the bottom row must be `[0, 0, 0, 1]`.
    #[test]
    fn pose_is_flattened_row_major_for_numpy() {
        let pose = Pose::from_quat_translation(
            UnitQuaternion::from_euler_angles(0.1, -0.4, 0.25),
            Vector3::new(1.5, -2.0, 3.25),
        );
        let flat = pose_matrix_row_major(&pose);
        let m = pose.matrix();
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    flat[4 * r + c],
                    m[(r, c)] as f32,
                    "flat[{}] should be M[{r}][{c}]; flat = {flat:?}",
                    4 * r + c
                );
            }
        }
        assert_eq!([flat[3], flat[7], flat[11]], [1.5, -2.0, 3.25]);
        assert_eq!(&flat[12..], &[0.0, 0.0, 0.0, 1.0]);
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PySlam>()?;
    Ok(())
}
