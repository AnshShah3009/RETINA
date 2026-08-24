use pyo3::prelude::*;

/// Compute the 1D FFT of real-valued input.
#[pyfunction]
fn fft(py: Python<'_>, data: Vec<f64>) -> Vec<(f64, f64)> {
    py.allow_threads(|| cv_scientific::fft::fft(&data))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Compute the 1D inverse FFT.
#[pyfunction]
fn ifft(py: Python<'_>, data: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let complex: Vec<rustfft::num_complex::Complex<f64>> = data
        .iter()
        .map(|&(re, im)| rustfft::num_complex::Complex::new(re, im))
        .collect();
    py.allow_threads(|| cv_scientific::fft::ifft(&complex))
        .into_iter()
        .map(|c| (c.re, c.im))
        .collect()
}

/// Welch's method for power spectral density estimation.
#[pyfunction]
fn welch_psd(
    py: Python<'_>,
    data: Vec<f64>,
    nperseg: usize,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    py.allow_threads(|| cv_scientific::signal::welch(&data, nperseg, None, sample_rate))
}

/// Design a Butterworth lowpass filter.
#[pyfunction]
fn butter_lowpass(order: usize, cutoff: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    cv_scientific::signal::butter(order, cutoff, sample_rate)
}

/// Zero-phase forward-backward digital filtering.
#[pyfunction]
fn filtfilt(py: Python<'_>, b: Vec<f64>, a: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    py.allow_threads(|| cv_scientific::signal::filtfilt(&b, &a, &x))
}

/// Find peaks (local maxima) in a 1D signal.
#[pyfunction]
#[pyo3(signature = (data, min_height=None, min_distance=None))]
fn find_peaks(
    py: Python<'_>,
    data: Vec<f64>,
    min_height: Option<f64>,
    min_distance: Option<usize>,
) -> Vec<usize> {
    py.allow_threads(|| cv_scientific::signal::find_peaks(&data, min_height, min_distance))
}

/// Simple linear regression: y = slope * x + intercept.
#[pyfunction]
fn linear_regression(py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<(f64, f64, f64)> {
    let result = py.allow_threads(|| cv_scientific::stats::linregress(&x, &y))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    Ok((result.slope, result.intercept, result.r_squared))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(welch_psd, m)?)?;
    m.add_function(wrap_pyfunction!(butter_lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(filtfilt, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    Ok(())
}
