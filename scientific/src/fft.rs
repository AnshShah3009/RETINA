//! FFT (Fast Fourier Transform) functions.
//!
//! Provides 1D/2D forward and inverse FFT, real-valued FFT,
//! frequency bin generation, and power spectral density estimation.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// 1D FFT of real-valued input.
///
/// Returns the full complex spectrum of length `input.len()`.
pub fn fft(input: &[f64]) -> Vec<Complex<f64>> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    buffer
}

/// 1D inverse FFT.
///
/// Returns the complex result (divide by N is applied). For real-valued
/// original signals the imaginary parts will be near zero.
pub fn ifft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);

    let mut buffer = input.to_vec();
    ifft.process(&mut buffer);

    let scale = 1.0 / n as f64;
    for c in &mut buffer {
        *c *= scale;
    }
    buffer
}

/// Real-valued FFT returning only positive frequencies.
///
/// Returns the first `N/2 + 1` complex coefficients (the non-redundant half).
pub fn rfft(input: &[f64]) -> Vec<Complex<f64>> {
    let spectrum = fft(input);
    let n = input.len();
    spectrum[..n / 2 + 1].to_vec()
}

/// FFT frequency bins for a signal of length `n` with sample spacing `d`.
///
/// Returns frequencies in cycles per unit of the sample spacing.
/// Layout matches NumPy's `fftfreq`: `[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)`.
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let nd = n as f64 * d;
    let mut freqs = Vec::with_capacity(n);
    let half = (n + 1) / 2; // ceil(n/2)
    for i in 0..half {
        freqs.push(i as f64 / nd);
    }
    for i in half..n {
        freqs.push((i as f64 - n as f64) / nd);
    }
    freqs
}

/// 2D FFT on row-major data of shape `height x width`.
///
/// Performs FFT along rows, then along columns.
pub fn fft2(input: &[f64], height: usize, width: usize) -> Vec<Complex<f64>> {
    assert_eq!(
        input.len(),
        height * width,
        "fft2: input length must equal height * width"
    );
    if height == 0 || width == 0 {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();

    // Row-wise FFT
    let fft_row = planner.plan_fft_forward(width);
    let mut data: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    for row in 0..height {
        let start = row * width;
        fft_row.process(&mut data[start..start + width]);
    }

    // Column-wise FFT
    let fft_col = planner.plan_fft_forward(height);
    let mut col_buf = vec![Complex::new(0.0, 0.0); height];
    for c in 0..width {
        for r in 0..height {
            col_buf[r] = data[r * width + c];
        }
        fft_col.process(&mut col_buf);
        for r in 0..height {
            data[r * width + c] = col_buf[r];
        }
    }

    data
}

/// 2D inverse FFT on row-major complex data of shape `height x width`.
pub fn ifft2(input: &[Complex<f64>], height: usize, width: usize) -> Vec<Complex<f64>> {
    assert_eq!(
        input.len(),
        height * width,
        "ifft2: input length must equal height * width"
    );
    if height == 0 || width == 0 {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();

    // Row-wise IFFT
    let ifft_row = planner.plan_fft_inverse(width);
    let mut data = input.to_vec();
    for row in 0..height {
        let start = row * width;
        ifft_row.process(&mut data[start..start + width]);
    }

    // Column-wise IFFT
    let ifft_col = planner.plan_fft_inverse(height);
    let mut col_buf = vec![Complex::new(0.0, 0.0); height];
    for c in 0..width {
        for r in 0..height {
            col_buf[r] = data[r * width + c];
        }
        ifft_col.process(&mut col_buf);
        for r in 0..height {
            data[r * width + c] = col_buf[r];
        }
    }

    let scale = 1.0 / (height * width) as f64;
    for c in &mut data {
        *c *= scale;
    }
    data
}

/// Power spectral density via periodogram.
///
/// Returns `(frequencies, power)` where `power[i] = |X[i]|^2 / (N * sample_rate)`
/// for the positive-frequency half of the spectrum.
pub fn psd(input: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    if input.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n = input.len();
    let spectrum = fft(input);
    let n_freqs = n / 2 + 1;

    let freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate / n as f64)
        .collect();

    let scale = 1.0 / (n as f64 * sample_rate);
    let mut power: Vec<f64> = spectrum[..n_freqs]
        .iter()
        .map(|c| c.norm_sqr() * scale)
        .collect();

    // Double non-DC, non-Nyquist bins to account for negative frequencies
    let nyquist = if n % 2 == 0 { n_freqs - 1 } else { n_freqs };
    for p in power.iter_mut().take(nyquist).skip(1) {
        *p *= 2.0;
    }

    (freqs, power)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_known_sinusoid() {
        // Pure sinusoid at frequency 1 Hz, sampled at 8 Hz, 8 samples
        let n = 8;
        let freq = 1.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();

        let spectrum = fft(&input);
        assert_eq!(spectrum.len(), n);

        // The peak should be at bin 1 (frequency = 1 Hz with sample_rate = n)
        let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let max_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(max_bin == 1 || max_bin == n - 1);
    }

    #[test]
    fn test_ifft_recovers_input() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft(&input);
        let recovered = ifft(&spectrum);

        for (orig, rec) in input.iter().zip(recovered.iter()) {
            assert!((orig - rec.re).abs() < 1e-10, "Real part mismatch");
            assert!(rec.im.abs() < 1e-10, "Imaginary part should be near zero");
        }
    }

    #[test]
    fn test_parseval_theorem() {
        // Parseval's theorem: sum |x[n]|^2 = (1/N) * sum |X[k]|^2
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let n = input.len();
        let spectrum = fft(&input);

        let time_energy: f64 = input.iter().map(|x| x * x).sum();
        let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

        assert!(
            (time_energy - freq_energy).abs() < 1e-10,
            "Parseval's theorem violated: time={} freq={}",
            time_energy,
            freq_energy
        );
    }

    #[test]
    fn test_fft2_ifft2_roundtrip() {
        let height = 4;
        let width = 4;
        let input: Vec<f64> = (0..16).map(|i| i as f64).collect();

        let spectrum = fft2(&input, height, width);
        let recovered = ifft2(&spectrum, height, width);

        for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec.re).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                rec.re
            );
        }
    }

    #[test]
    fn test_fftfreq() {
        let freqs = fftfreq(8, 1.0);
        assert_eq!(freqs.len(), 8);
        assert!((freqs[0] - 0.0).abs() < 1e-10);
        assert!((freqs[1] - 0.125).abs() < 1e-10);
        assert!((freqs[4] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_psd_sinusoid() {
        // A sinusoid should produce a peak at its frequency in the PSD
        let n = 256;
        let sample_rate = 256.0;
        let freq = 32.0;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect();

        let (freqs, power) = psd(&input, sample_rate);
        let max_idx = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert!(
            (freqs[max_idx] - freq).abs() < sample_rate / n as f64 + 1e-10,
            "PSD peak at {} Hz, expected {} Hz",
            freqs[max_idx],
            freq
        );
    }
}
