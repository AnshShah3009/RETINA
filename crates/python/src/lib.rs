mod core;
mod dnn;
mod features;
mod helpers;
mod imgproc;
mod optimize;
mod runtime;
mod scientific;
mod three_d;
mod video;

use pyo3::prelude::*;

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::register(m)?;
    core::register(m)?;
    features::register(m)?;
    video::register(m)?;
    dnn::register(m)?;
    optimize::register(m)?;
    three_d::register(m)?;
    imgproc::register(m)?;
    scientific::register(m)?;
    Ok(())
}
