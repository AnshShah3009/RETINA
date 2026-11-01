use crate::runtime::PyWorkloadHint;
use cv_dnn::DnnNet;
use cv_runtime::orchestrator::scheduler;
use image::GrayImage;
use pyo3::prelude::*;
use std::sync::Arc;

/// Why a `(bytes, width, height)` triple handed to `PyDnnNet::forward` is not a
/// single-channel frame.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FrameError {
    /// `bytes.len()` is not `width * height`, so it cannot be a luma frame of
    /// that size.
    Size { got: usize, expected: usize },
    /// The dimensions overflowed `usize` when multiplied out.
    DimensionsOverflow,
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Size { got, expected } => write!(
                f,
                "image has {got} values but width * height is {expected}; the buffer is \
                 treated as a single-channel frame, so an RGB/RGBA frame must be converted \
                 to luma first"
            ),
            FrameError::DimensionsOverflow => {
                write!(f, "width * height overflows the addressable size")
            }
        }
    }
}

/// Build the `width x height` luma frame that `forward` implies, after checking
/// the buffer really is that big.
///
/// `image::GrayImage::from_raw` rejects a buffer that is *too small* but
/// **accepts one that is too large**: its check is `width * height <= len`
/// (`image-0.25.9/src/images/buffer.rs:809`). So `dnn.forward(rgb_frame, w, h)`
/// — 48 bytes for a 4x4 frame — built a 4x4 `GrayImage` that still held all 48
/// bytes. Everything downstream trusts `dimensions()` and reads only the first
/// 16, so the caller's frame silently became a **black image**: measured,
/// `resize_ctx` of that image returned pixels `[0, 0, 0, 0, 0, 0, 0, 0]` while
/// the exact-length buffer returned `[0, 1, 2, 3, 4, 5, 6, 7]`. Every other
/// input path in this workspace reports a size mismatch rather than inventing
/// pixels — `cv_dnn::blob::blob_to_image` asserts on both directions — so the
/// binding rejects it too.
pub(crate) fn luma_frame(
    image: Vec<u8>,
    width: usize,
    height: usize,
) -> Result<GrayImage, FrameError> {
    let expected = width
        .checked_mul(height)
        .ok_or(FrameError::DimensionsOverflow)?;
    if image.len() != expected {
        return Err(FrameError::Size {
            got: image.len(),
            expected,
        });
    }
    // Exact length, so `from_raw` cannot drop anything; `None` here would mean
    // the dimensions overflowed, which the check above already excluded.
    GrayImage::from_raw(width as u32, height as u32, image).ok_or(FrameError::DimensionsOverflow)
}

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
        let gray = luma_frame(image, width, height).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("PyDnnNet.forward: {e}"))
        })?;

        let dyn_img = image::DynamicImage::ImageLuma8(gray);

        let input_tensor = self
            .inner
            .preprocess(&dyn_img, &self.runner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Full DNN inference — release the GIL for the compute.
        let outputs = py
            .detach(|| self.inner.forward(&input_tensor))
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

#[cfg(test)]
mod tests {
    use super::{FrameError, luma_frame};

    /// Regression: an oversized buffer used to be accepted and then silently
    /// truncated to the first `width * height` bytes.
    ///
    /// `image::GrayImage::from_raw` only rejects a buffer that is too *small*
    /// (`check_image_fits` is `width * height <= len`), so passing a 4x4 RGB
    /// frame (48 bytes) where `forward` expects a 4x4 luma frame (16 bytes)
    /// produced a `GrayImage` whose `dimensions()` said 4x4 but whose `as_raw()`
    /// was still 48 long. `preprocess` then resampled it and every pixel came
    /// back 0: the model was fed a black image built from the caller's frame,
    /// with no error anywhere.
    #[test]
    fn an_oversized_frame_is_rejected_not_truncated_to_black() {
        let rgb = vec![7u8; 48];
        let err = luma_frame(rgb, 4, 4).expect_err("48 bytes is not a 4x4 luma frame");
        assert_eq!(
            err,
            FrameError::Size {
                got: 48,
                expected: 16
            }
        );
        let msg = err.to_string();
        assert!(
            msg.contains("48") && msg.contains("16"),
            "the error must name both lengths, got {msg}"
        );

        // Too few is the same defect in the other direction, and used to map to
        // the unhelpful "Invalid image dimensions".
        let err = luma_frame(vec![0u8; 15], 4, 4).expect_err("15 bytes is not a 4x4 frame");
        assert_eq!(
            err,
            FrameError::Size {
                got: 15,
                expected: 16
            }
        );
    }

    /// A zero-sized frame is a caller error too: `from_raw(0, 0, vec![])`
    /// succeeds and hands the resampler an empty image, which `resize_ctx`
    /// turns into a fresh all-zero image of the target size.
    #[test]
    fn a_zero_sized_frame_is_rejected() {
        assert_eq!(
            luma_frame(vec![1u8, 2], 0, 0).expect_err("2 values for a 0x0 frame"),
            FrameError::Size {
                got: 2,
                expected: 0
            }
        );
    }

    /// Control: an exact buffer is still accepted, unchanged and in order.
    #[test]
    fn an_exact_frame_still_round_trips() {
        let bytes: Vec<u8> = (0..16).collect();
        let img = luma_frame(bytes.clone(), 4, 4).expect("an exact 4x4 frame");
        assert_eq!(img.dimensions(), (4, 4));
        assert_eq!(img.as_raw(), &bytes);

        // And a 1x1 frame, the smallest legal one.
        let one = luma_frame(vec![200], 1, 1).expect("a 1x1 frame");
        assert_eq!(one.get_pixel(0, 0)[0], 200);
    }
}
