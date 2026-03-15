//! Signal processing functions (scipy.signal equivalents).
//!
//! Provides window functions, digital filter design, spectral analysis,
//! convolution/correlation, peak finding, and resampling.

use crate::fft::{fft, ifft};
use crate::special::bessel_i0;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Window Functions
// ---------------------------------------------------------------------------

/// Hann (raised cosine) window of length `n`.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Hamming window of length `n`.
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Blackman window of length `n`.
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nm1 = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 / nm1;
            0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
        })
        .collect()
}

/// Kaiser window of length `n` with shape parameter `beta`.
///
/// Uses the modified Bessel function I0 from `crate::special::bessel_i0`.
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let nm1 = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let alpha = 2.0 * i as f64 / nm1 - 1.0;
            bessel_i0(beta * (1.0 - alpha * alpha).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Bartlett (triangular) window of length `n`.
pub fn bartlett_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nm1 = (n - 1) as f64;
    (0..n)
        .map(|i| 1.0 - (2.0 * i as f64 / nm1 - 1.0).abs())
        .collect()
}

// ---------------------------------------------------------------------------
// Digital Filters
// ---------------------------------------------------------------------------

/// Apply an IIR filter (single pass, causal) using Direct Form II transposed.
///
/// `b` are the numerator (feedforward) coefficients,
/// `a` are the denominator (feedback) coefficients.
/// `a[0]` is used to normalise; typically `a[0] == 1.0`.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    assert!(!a.is_empty() && a[0] != 0.0, "a[0] must be nonzero");
    assert!(!b.is_empty(), "b must be non-empty");

    let a0 = a[0];
    let order = b.len().max(a.len());
    let mut z = vec![0.0; order]; // delay line

    let mut y = Vec::with_capacity(x.len());
    for &xi in x {
        let yi = b[0] / a0 * xi + z[0];
        y.push(yi);

        for j in 0..order - 1 {
            let bj1 = if j + 1 < b.len() { b[j + 1] } else { 0.0 };
            let aj1 = if j + 1 < a.len() { a[j + 1] } else { 0.0 };
            let zn = if j + 1 < order - 1 { z[j + 1] } else { 0.0 };
            z[j] = bj1 / a0 * xi - aj1 / a0 * yi + zn;
        }
    }
    y
}

/// Zero-phase IIR filtering (forward + reverse pass).
///
/// The signal is padded with reflected edges to reduce transient artifacts.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let nfilt = b.len().max(a.len());
    let pad_len = 3 * nfilt;

    // Reflect-pad the signal at both ends
    let n = x.len();
    if n <= pad_len {
        // Signal too short for padding; just do forward+reverse without padding
        let fwd = lfilter(b, a, x);
        let mut rev: Vec<f64> = fwd.into_iter().rev().collect();
        rev = lfilter(b, a, &rev);
        rev.reverse();
        return rev;
    }

    let mut padded = Vec::with_capacity(n + 2 * pad_len);
    // Left reflection: x[pad_len], x[pad_len-1], ..., x[1]  reflected about x[0]
    for i in (1..=pad_len).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    // Right reflection
    for i in 1..=pad_len {
        padded.push(2.0 * x[n - 1] - x[n - 1 - i]);
    }

    // Forward pass
    let fwd = lfilter(b, a, &padded);
    // Reverse pass
    let mut rev: Vec<f64> = fwd.into_iter().rev().collect();
    rev = lfilter(b, a, &rev);
    rev.reverse();

    // Trim padding
    rev[pad_len..pad_len + n].to_vec()
}

/// Design Butterworth lowpass filter coefficients.
///
/// Returns `(b, a)` digital transfer function coefficients.
/// `cutoff` is the cutoff frequency in Hz, `sample_rate` in Hz.
pub fn butter(order: usize, cutoff: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(order > 0, "order must be >= 1");
    assert!(
        cutoff > 0.0 && cutoff < sample_rate / 2.0,
        "cutoff must be in (0, Nyquist)"
    );

    // Pre-warp
    let wc = (PI * cutoff / sample_rate).tan();

    // Analog Butterworth poles (left half-plane)
    let n = order;
    let mut poles: Vec<Complex<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
        poles.push(Complex::new(wc * theta.cos(), wc * theta.sin()));
    }

    // Bilinear transform: s = 2*(z-1)/(z+1)  =>  z = (1 + s/2) / (1 - s/2)
    let digital_poles: Vec<Complex<f64>> = poles
        .iter()
        .map(|&p| (Complex::new(1.0, 0.0) + p) / (Complex::new(1.0, 0.0) - p))
        .collect();

    // Digital zeros: all at z = -1 for lowpass
    let digital_zeros: Vec<Complex<f64>> = vec![Complex::new(-1.0, 0.0); n];

    // Build polynomials from roots
    let a = poly_from_roots(&digital_poles);
    let b_raw = poly_from_roots(&digital_zeros);

    // Normalise gain at DC (z = 1)
    let gain_a: f64 = a.iter().sum();
    let gain_b: f64 = b_raw.iter().sum();
    let gain = gain_a / gain_b;

    let b: Vec<f64> = b_raw.iter().map(|&c| c * gain).collect();
    (b, a)
}

/// Design Butterworth highpass filter coefficients.
pub fn butter_highpass(order: usize, cutoff: f64, sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(order > 0, "order must be >= 1");
    assert!(
        cutoff > 0.0 && cutoff < sample_rate / 2.0,
        "cutoff must be in (0, Nyquist)"
    );

    // Pre-warp
    let wc = (PI * cutoff / sample_rate).tan();

    let n = order;
    // Analog lowpass prototype poles at unit cutoff, then transform to highpass: s -> wc^2/s
    // Equivalent: analog highpass poles
    let mut poles: Vec<Complex<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
        let lp_pole = Complex::new(theta.cos(), theta.sin()); // unit Butterworth pole
                                                              // Lowpass to highpass in analog: s -> wc / s  =>  pole_hp = wc / pole_lp
        let hp_pole = Complex::new(wc, 0.0) / lp_pole;
        poles.push(hp_pole);
    }

    // Bilinear transform
    let digital_poles: Vec<Complex<f64>> = poles
        .iter()
        .map(|&p| (Complex::new(1.0, 0.0) + p) / (Complex::new(1.0, 0.0) - p))
        .collect();

    // Digital zeros: all at z = +1 for highpass
    let digital_zeros: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0); n];

    let a = poly_from_roots(&digital_poles);
    let b_raw = poly_from_roots(&digital_zeros);

    // Normalise gain at Nyquist (z = -1)
    let gain_a: f64 = a
        .iter()
        .enumerate()
        .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
        .sum();
    let gain_b: f64 = b_raw
        .iter()
        .enumerate()
        .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
        .sum();
    let gain = gain_a / gain_b;

    let b: Vec<f64> = b_raw.iter().map(|&c| c * gain).collect();
    (b, a)
}

/// Design Butterworth bandpass filter coefficients.
///
/// `low` and `high` are the band edges in Hz.
pub fn butter_bandpass(
    order: usize,
    low: f64,
    high: f64,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(order > 0, "order must be >= 1");
    assert!(low > 0.0 && high > low && high < sample_rate / 2.0);

    // Pre-warp both edges
    let wl = (PI * low / sample_rate).tan();
    let wh = (PI * high / sample_rate).tan();
    let bw = wh - wl;
    let w0 = (wl * wh).sqrt();

    let n = order;
    // Analog lowpass prototype poles
    let mut analog_bp_poles: Vec<Complex<f64>> = Vec::with_capacity(2 * n);
    for k in 0..n {
        let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
        let lp_pole = Complex::new(theta.cos(), theta.sin());
        // LP to BP transform: s -> (s^2 + w0^2) / (bw * s)
        // Quadratic: s^2 - bw*lp*s + w0^2 = 0
        let half_bw_p = Complex::new(bw, 0.0) * lp_pole * Complex::new(0.5, 0.0);
        let disc = half_bw_p * half_bw_p - Complex::new(w0 * w0, 0.0);
        let sqrt_disc = disc.sqrt();
        analog_bp_poles.push(half_bw_p + sqrt_disc);
        analog_bp_poles.push(half_bw_p - sqrt_disc);
    }

    // Bilinear transform
    let digital_poles: Vec<Complex<f64>> = analog_bp_poles
        .iter()
        .map(|&p| (Complex::new(1.0, 0.0) + p) / (Complex::new(1.0, 0.0) - p))
        .collect();

    // Zeros: n at z=+1, n at z=-1
    let mut digital_zeros: Vec<Complex<f64>> = Vec::with_capacity(2 * n);
    for _ in 0..n {
        digital_zeros.push(Complex::new(1.0, 0.0));
    }
    for _ in 0..n {
        digital_zeros.push(Complex::new(-1.0, 0.0));
    }

    let a = poly_from_roots(&digital_poles);
    let b_raw = poly_from_roots(&digital_zeros);

    // Normalise gain at center frequency
    let w_center = 2.0 * PI * (low + high) / 2.0 / sample_rate;
    let ejw = Complex::new(0.0, w_center).exp();
    let num: Complex<f64> = b_raw
        .iter()
        .enumerate()
        .fold(Complex::new(0.0, 0.0), |acc, (i, &c)| {
            acc + Complex::new(c, 0.0) * ejw.powi(-(i as i32))
        });
    let den: Complex<f64> = a
        .iter()
        .enumerate()
        .fold(Complex::new(0.0, 0.0), |acc, (i, &c)| {
            acc + Complex::new(c, 0.0) * ejw.powi(-(i as i32))
        });
    let gain = (den / num).norm();

    let b: Vec<f64> = b_raw.iter().map(|&c| c * gain).collect();
    (b, a)
}

/// FIR filter design using the windowed-sinc method (Hamming window).
///
/// `numtaps` is the filter length (must be odd for type I),
/// `cutoff` in Hz, `sample_rate` in Hz.
pub fn firwin(numtaps: usize, cutoff: f64, sample_rate: f64) -> Vec<f64> {
    assert!(numtaps > 0, "numtaps must be >= 1");
    assert!(cutoff > 0.0 && cutoff < sample_rate / 2.0);

    let fc = cutoff / sample_rate; // normalised cutoff
    let m = (numtaps - 1) as f64;
    let window = hamming_window(numtaps);

    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let n = i as f64 - m / 2.0;
            if n.abs() < 1e-12 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * n).sin() / (PI * n)
            }
        })
        .collect();

    // Apply window
    for (hi, wi) in h.iter_mut().zip(window.iter()) {
        *hi *= wi;
    }

    // Normalise to unit DC gain
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-15 {
        for hi in &mut h {
            *hi /= sum;
        }
    }

    h
}

// Helper: build real polynomial coefficients from complex roots.
// Returns coefficients [1, c1, c2, ...] of (z - r0)(z - r1)...
fn poly_from_roots(roots: &[Complex<f64>]) -> Vec<f64> {
    let n = roots.len();
    let mut coeffs: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n + 1];
    coeffs[0] = Complex::new(1.0, 0.0);
    for (i, &r) in roots.iter().enumerate() {
        for j in (1..=i + 1).rev() {
            coeffs[j] = coeffs[j] - r * coeffs[j - 1];
        }
    }
    coeffs.iter().map(|c| c.re).collect()
}

// ---------------------------------------------------------------------------
// Spectral Analysis
// ---------------------------------------------------------------------------

/// Welch's method for power spectral density estimation.
///
/// Returns `(frequencies, psd)`.
/// `nperseg` is the segment length, `noverlap` defaults to `nperseg / 2`.
pub fn welch(
    x: &[f64],
    nperseg: usize,
    noverlap: Option<usize>,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(nperseg > 0 && nperseg <= x.len(), "nperseg out of range");
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg - noverlap;
    assert!(step > 0, "noverlap must be < nperseg");

    let window = hann_window(nperseg);
    let win_ss: f64 = window.iter().map(|w| w * w).sum();
    let n_freqs = nperseg / 2 + 1;

    let mut psd = vec![0.0; n_freqs];
    let mut n_segments = 0usize;

    let mut offset = 0;
    while offset + nperseg <= x.len() {
        // Window the segment and FFT
        let segment: Vec<f64> = x[offset..offset + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = fft(&segment);

        for i in 0..n_freqs {
            psd[i] += spectrum[i].norm_sqr();
        }
        n_segments += 1;
        offset += step;
    }

    // Average and normalise
    let scale = 1.0 / (sample_rate * win_ss * n_segments as f64);
    for p in &mut psd {
        *p *= scale;
    }
    // Double non-DC, non-Nyquist
    let nyquist_idx = if nperseg.is_multiple_of(2) {
        n_freqs - 1
    } else {
        n_freqs
    };
    for p in psd.iter_mut().take(nyquist_idx).skip(1) {
        *p *= 2.0;
    }

    let freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate / nperseg as f64)
        .collect();

    (freqs, psd)
}

/// Short-Time Fourier Transform.
///
/// Returns `(times, frequencies, complex_spectrogram)` where each element
/// of the spectrogram is `(real, imag)`.
#[allow(clippy::type_complexity)]
pub fn stft(
    x: &[f64],
    nperseg: usize,
    noverlap: Option<usize>,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<(f64, f64)>>) {
    assert!(nperseg > 0 && nperseg <= x.len());
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg - noverlap;
    assert!(step > 0);

    let window = hann_window(nperseg);
    let n_freqs = nperseg / 2 + 1;

    let freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate / nperseg as f64)
        .collect();

    let mut times = Vec::new();
    let mut spectro = Vec::new();

    let mut offset = 0;
    while offset + nperseg <= x.len() {
        let segment: Vec<f64> = x[offset..offset + nperseg]
            .iter()
            .zip(window.iter())
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = fft(&segment);

        let frame: Vec<(f64, f64)> = spectrum[..n_freqs].iter().map(|c| (c.re, c.im)).collect();
        spectro.push(frame);
        times.push((offset as f64 + nperseg as f64 / 2.0) / sample_rate);
        offset += step;
    }

    (times, freqs, spectro)
}

/// Spectrogram (magnitude-squared of STFT).
///
/// Returns `(times, frequencies, power_spectrogram)`.
pub fn spectrogram(
    x: &[f64],
    nperseg: usize,
    noverlap: Option<usize>,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let (times, freqs, complex_spectro) = stft(x, nperseg, noverlap, sample_rate);
    let power: Vec<Vec<f64>> = complex_spectro
        .iter()
        .map(|frame| frame.iter().map(|&(re, im)| re * re + im * im).collect())
        .collect();
    (times, freqs, power)
}

// ---------------------------------------------------------------------------
// Convolution & Correlation
// ---------------------------------------------------------------------------

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvMode {
    /// Full convolution output of length `len(signal) + len(kernel) - 1`.
    Full,
    /// Output is the same length as `signal` (centred).
    Same,
    /// Only parts where the kernel fully overlaps the signal.
    Valid,
}

/// 1D convolution.
pub fn convolve1d(signal: &[f64], kernel: &[f64], mode: ConvMode) -> Vec<f64> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }
    let ns = signal.len();
    let nk = kernel.len();
    let full_len = ns + nk - 1;

    // Compute full convolution
    let mut full = vec![0.0; full_len];
    for (i, &si) in signal.iter().enumerate() {
        for (j, &kj) in kernel.iter().enumerate() {
            full[i + j] += si * kj;
        }
    }

    match mode {
        ConvMode::Full => full,
        ConvMode::Same => {
            let start = (nk - 1) / 2;
            full[start..start + ns].to_vec()
        }
        ConvMode::Valid => {
            if ns >= nk {
                full[nk - 1..ns].to_vec()
            } else {
                full[ns - 1..nk].to_vec()
            }
        }
    }
}

/// 1D cross-correlation.
///
/// Equivalent to convolving `a` with the time-reversed `b`.
pub fn correlate1d(a: &[f64], b: &[f64], mode: ConvMode) -> Vec<f64> {
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    convolve1d(a, &b_rev, mode)
}

/// Find peaks (local maxima) in a 1D signal.
///
/// Returns indices of peaks that satisfy optional minimum height and minimum
/// inter-peak distance constraints.
pub fn find_peaks(
    data: &[f64],
    min_height: Option<f64>,
    min_distance: Option<usize>,
) -> Vec<usize> {
    if data.len() < 3 {
        return Vec::new();
    }

    // Find all local maxima
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            if let Some(h) = min_height {
                if data[i] < h {
                    continue;
                }
            }
            peaks.push(i);
        }
    }

    // Enforce minimum distance (greedy, prefer taller peaks)
    if let Some(dist) = min_distance {
        if dist > 0 {
            // Sort by height descending
            let mut sorted: Vec<usize> = peaks.clone();
            sorted.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());

            let mut keep = vec![false; data.len()];
            let mut result = Vec::new();
            for &idx in &sorted {
                let too_close = result
                    .iter()
                    .any(|&r: &usize| (idx as isize - r as isize).unsigned_abs() < dist);
                if !too_close {
                    keep[idx] = true;
                    result.push(idx);
                }
            }
            // Return in original order
            peaks.retain(|i| keep[*i]);
        }
    }

    peaks
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample a signal to a new length using the FFT method.
///
/// Zero-pads or truncates in the frequency domain, then applies inverse FFT.
pub fn resample(x: &[f64], num: usize) -> Vec<f64> {
    if x.is_empty() || num == 0 {
        return Vec::new();
    }
    let n = x.len();
    if n == num {
        return x.to_vec();
    }

    let spectrum = fft(x);

    let mut new_spectrum = vec![Complex::new(0.0, 0.0); num];

    if num > n {
        // Zero-pad: copy positive freqs, then negative freqs
        let half = n.div_ceil(2); // positive freq bins including DC
                                  // Copy positive frequencies
        new_spectrum[..half].copy_from_slice(&spectrum[..half]);
        // Copy negative frequencies at the end
        let neg_start_src = half;
        let neg_count = n - half;
        let neg_start_dst = num - neg_count;
        new_spectrum[neg_start_dst..neg_start_dst + neg_count]
            .copy_from_slice(&spectrum[neg_start_src..neg_start_src + neg_count]);
        // If n is even, split the Nyquist bin
        if n.is_multiple_of(2) {
            let nyq = n / 2;
            new_spectrum[nyq] = spectrum[nyq] * Complex::new(0.5, 0.0);
            new_spectrum[num - nyq] = spectrum[nyq] * Complex::new(0.5, 0.0);
        }
    } else {
        // Truncate
        let half = num.div_ceil(2);
        new_spectrum[..half].copy_from_slice(&spectrum[..half]);
        let neg_count = num - half;
        new_spectrum[half..half + neg_count].copy_from_slice(&spectrum[n - neg_count..n]);
    }

    let result = ifft(&new_spectrum);
    let scale = num as f64 / n as f64;
    result.iter().map(|c| c.re * scale).collect()
}

/// Decimate (downsample with anti-aliasing lowpass filter).
///
/// Applies a lowpass Butterworth filter at Nyquist/factor, then takes every
/// `factor`-th sample.
pub fn decimate(x: &[f64], factor: usize) -> Vec<f64> {
    assert!(factor > 0, "factor must be >= 1");
    if factor == 1 {
        return x.to_vec();
    }
    // 8th order Butterworth is standard (like scipy default)
    let order = (8.min(x.len() / 3)).max(1);
    // Cutoff at 0.8 * Nyquist/factor to allow some transition band
    // Using normalised sample_rate = 1.0
    let cutoff = 0.8 / (2.0 * factor as f64);
    let cutoff = cutoff.clamp(0.001, 0.499);
    let (b, a) = butter(order, cutoff, 1.0);

    let filtered = filtfilt(&b, &a, x);
    filtered.iter().step_by(factor).copied().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_hann_window_symmetry() {
        let n = 64;
        let w = hann_window(n);
        assert_eq!(w.len(), n);
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < TOL,
                "Hann not symmetric at {}",
                i
            );
        }
        // Endpoints should be 0
        assert!(w[0].abs() < TOL);
        assert!(w[n - 1].abs() < TOL);
    }

    #[test]
    fn test_hamming_window_properties() {
        let n = 64;
        let w = hamming_window(n);
        assert_eq!(w.len(), n);
        // Hamming is symmetric
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < TOL);
        }
        // Hamming endpoints are 0.08 (not zero)
        assert!((w[0] - 0.08).abs() < 1e-6);
    }

    #[test]
    fn test_blackman_window_endpoints() {
        let n = 64;
        let w = blackman_window(n);
        // Blackman endpoints are very close to 0
        assert!(w[0].abs() < 1e-4);
        assert!(w[n - 1].abs() < 1e-4);
        // Peak at center
        assert!(w[n / 2] > 0.9);
    }

    #[test]
    fn test_kaiser_window_beta_zero_is_rectangular() {
        let n = 32;
        let w = kaiser_window(n, 0.0);
        for &wi in &w {
            assert!(
                (wi - 1.0).abs() < 1e-6,
                "Kaiser(beta=0) should be rectangular"
            );
        }
    }

    #[test]
    fn test_bartlett_window_triangular() {
        let n = 33;
        let w = bartlett_window(n);
        // Peak at center
        assert!((w[16] - 1.0).abs() < TOL);
        // Endpoints zero
        assert!(w[0].abs() < TOL);
        assert!(w[n - 1].abs() < TOL);
    }

    #[test]
    fn test_butter_dc_and_nyquist_response() {
        // 4th order lowpass at 100 Hz, sample rate 1000 Hz
        let (b, a) = butter(4, 100.0, 1000.0);

        // DC gain (z=1): sum(b)/sum(a) should be ~1.0
        let dc_num: f64 = b.iter().sum();
        let dc_den: f64 = a.iter().sum();
        let dc_gain = (dc_num / dc_den).abs();
        assert!(
            (dc_gain - 1.0).abs() < 1e-6,
            "DC gain should be 1.0, got {}",
            dc_gain
        );

        // At Nyquist (z=-1) the response should be very small
        let nyq_num: f64 = b
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
            .sum();
        let nyq_den: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
            .sum();
        let nyq_gain = (nyq_num / nyq_den).abs();
        assert!(
            nyq_gain < 0.01,
            "Nyquist gain should be near 0, got {}",
            nyq_gain
        );
    }

    #[test]
    fn test_lfilter_impulse_response() {
        // For an FIR filter (a=[1]), impulse response = b coefficients
        let b = vec![1.0, 0.5, 0.25];
        let a = vec![1.0];
        let impulse = vec![1.0, 0.0, 0.0, 0.0, 0.0];

        let y = lfilter(&b, &a, &impulse);
        assert!((y[0] - 1.0).abs() < TOL);
        assert!((y[1] - 0.5).abs() < TOL);
        assert!((y[2] - 0.25).abs() < TOL);
        assert!((y[3]).abs() < TOL);
    }

    #[test]
    fn test_filtfilt_zero_phase() {
        // Filter a sinusoid; filtfilt should preserve phase
        let n = 500;
        let fs = 1000.0;
        let f = 50.0; // 50 Hz sinusoid (well within passband)
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();

        let (b, a) = butter(4, 200.0, fs);
        let filtered = filtfilt(&b, &a, &signal);

        // Check that peak locations match (zero phase = same peak positions)
        let sig_peaks = find_peaks(&signal, Some(0.5), None);
        let filt_peaks = find_peaks(&filtered, Some(0.5), None);

        assert!(!sig_peaks.is_empty());
        // Peaks should be at the same indices (or within 1 sample)
        for (sp, fp) in sig_peaks.iter().zip(filt_peaks.iter()) {
            assert!(
                (*sp as isize - *fp as isize).unsigned_abs() <= 1,
                "Phase shift detected: sig peak at {}, filtered peak at {}",
                sp,
                fp
            );
        }
    }

    #[test]
    fn test_welch_psd_peak() {
        // 100 Hz sinusoid, sampled at 1000 Hz
        let fs = 1000.0;
        let f = 100.0;
        let n = 2048;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();

        let (freqs, psd) = welch(&signal, 256, None, fs);

        let max_idx = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert!(
            (freqs[max_idx] - f).abs() < fs / 256.0 * 2.0,
            "PSD peak at {} Hz, expected ~{} Hz",
            freqs[max_idx],
            f
        );
    }

    #[test]
    fn test_convolve1d_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];

        let full = convolve1d(&a, &b, ConvMode::Full);
        // [1*0, 1*1+2*0, 1*0.5+2*1+3*0, 2*0.5+3*1, 3*0.5]
        // = [0, 1, 2.5, 4, 1.5]
        assert_eq!(full.len(), 5);
        assert!((full[0] - 0.0).abs() < TOL);
        assert!((full[1] - 1.0).abs() < TOL);
        assert!((full[2] - 2.5).abs() < TOL);
        assert!((full[3] - 4.0).abs() < TOL);
        assert!((full[4] - 1.5).abs() < TOL);

        let same = convolve1d(&a, &b, ConvMode::Same);
        assert_eq!(same.len(), 3);
        assert!((same[0] - 1.0).abs() < TOL);
        assert!((same[1] - 2.5).abs() < TOL);
        assert!((same[2] - 4.0).abs() < TOL);

        let valid = convolve1d(&a, &b, ConvMode::Valid);
        assert_eq!(valid.len(), 1);
        assert!((valid[0] - 2.5).abs() < TOL);
    }

    #[test]
    fn test_find_peaks_sinusoid() {
        let n = 1000;
        let fs = 1000.0;
        let f = 10.0; // 10 Hz => 10 peaks in 1 second
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();

        let peaks = find_peaks(&signal, Some(0.5), None);
        // Expect ~10 peaks (one per cycle)
        assert!(
            peaks.len() >= 9 && peaks.len() <= 11,
            "Expected ~10 peaks, got {}",
            peaks.len()
        );

        // Peaks should be roughly 100 samples apart
        for pair in peaks.windows(2) {
            let diff = pair[1] - pair[0];
            assert!(
                (diff as f64 - 100.0).abs() < 2.0,
                "Peak spacing {} not ~100",
                diff
            );
        }
    }

    #[test]
    fn test_resample_double_length() {
        // Resample a sine wave to double length; should preserve the frequency
        let n = 128;
        let f = 5.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / n as f64).sin())
            .collect();

        let resampled = resample(&signal, 2 * n);
        assert_eq!(resampled.len(), 2 * n);

        // Check that the resampled signal still has the same frequency content
        // by verifying values at half-sample points
        for i in 0..n - 1 {
            let expected = (2.0 * PI * f * (2 * i + 1) as f64 / (2 * n) as f64).sin();
            let actual = resampled[2 * i + 1];
            assert!(
                (expected - actual).abs() < 0.15,
                "Resample mismatch at {}: expected {}, got {}",
                2 * i + 1,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_decimate_preserves_low_freq() {
        // Create signal with low + high frequency
        let n = 2000;
        let fs = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin() + (2.0 * PI * 400.0 * t).sin()
            })
            .collect();

        let decimated = decimate(&signal, 4);
        // After decimation by 4, effective sample rate is 250 Hz
        // 10 Hz component should survive, 400 Hz should be gone

        // Check that decimated length is roughly n/4
        assert!(
            decimated.len() == n / 4,
            "Expected {} samples, got {}",
            n / 4,
            decimated.len()
        );

        // The decimated signal should mostly contain the 10 Hz component
        // Check that it oscillates at roughly the right frequency
        let peaks = find_peaks(&decimated, Some(0.3), None);
        // 10 Hz with 250 Hz sample rate: period = 25 samples, ~20 peaks in 500 samples
        assert!(
            peaks.len() >= 15,
            "Expected many peaks from 10 Hz component, got {}",
            peaks.len()
        );
    }

    #[test]
    fn test_spectrogram_output_shape() {
        let n = 1024;
        let nperseg = 256;
        let noverlap = 128;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let (times, freqs, power) = spectrogram(&signal, nperseg, Some(noverlap), 1000.0);

        let expected_frames = (n - nperseg) / (nperseg - noverlap) + 1;
        assert_eq!(times.len(), expected_frames);
        assert_eq!(freqs.len(), nperseg / 2 + 1);
        assert_eq!(power.len(), expected_frames);
        for frame in &power {
            assert_eq!(frame.len(), nperseg / 2 + 1);
        }
    }

    #[test]
    fn test_firwin_lowpass() {
        let h = firwin(51, 100.0, 1000.0);
        assert_eq!(h.len(), 51);
        // Sum should be ~1 (unit DC gain)
        let sum: f64 = h.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "FIR DC gain should be 1.0, got {}",
            sum
        );
        // Should be symmetric
        for i in 0..25 {
            assert!((h[i] - h[50 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_butter_highpass_dc_zero() {
        let (b, a) = butter_highpass(4, 100.0, 1000.0);

        // DC gain should be ~0
        let dc_num: f64 = b.iter().sum();
        let dc_den: f64 = a.iter().sum();
        let dc_gain = (dc_num / dc_den).abs();
        assert!(dc_gain < 1e-6, "HP DC gain should be ~0, got {}", dc_gain);

        // Nyquist gain should be ~1
        let nyq_num: f64 = b
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
            .sum();
        let nyq_den: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(i as i32))
            .sum();
        let nyq_gain = (nyq_num / nyq_den).abs();
        assert!(
            (nyq_gain - 1.0).abs() < 0.01,
            "HP Nyquist gain should be ~1, got {}",
            nyq_gain
        );
    }

    #[test]
    fn test_dwt_haar_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, detail) = dwt(&x, Wavelet::Haar);
        let reconstructed = idwt(&approx, &detail, Wavelet::Haar);
        for (a, b) in x.iter().zip(reconstructed.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "DWT roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_dwt_db2_roundtrip() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).sin()).collect();
        let (approx, detail) = dwt(&x, Wavelet::Db2);
        let reconstructed = idwt(&approx, &detail, Wavelet::Db2);
        for (a, b) in x.iter().zip(reconstructed.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "DB2 roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_wavedec_levels() {
        let x: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let coeffs = wavedec(&x, Wavelet::Haar, 3);
        assert_eq!(coeffs.len(), 4); // 3 detail levels + 1 approx
        let reconstructed = waverec(&coeffs, Wavelet::Haar);
        for (a, b) in x.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    #[test]
    fn test_cwt_morlet() {
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        let scales = vec![1.0, 2.0, 4.0, 8.0];
        let result = cwt(&x, &scales, CwtWavelet::Morlet);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].len(), 64);
    }
}

// ── Wavelet Transforms ──────────────────────────────────────────────────────

/// Wavelet family for DWT.
#[derive(Debug, Clone, Copy)]
pub enum Wavelet {
    /// Haar wavelet (simplest, equivalent to db1)
    Haar,
    /// Daubechies-2 wavelet
    Db2,
    /// Daubechies-4 wavelet
    Db4,
}

impl Wavelet {
    /// Low-pass decomposition filter coefficients.
    pub fn lo_d(&self) -> &'static [f64] {
        match self {
            Wavelet::Haar => &[
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            Wavelet::Db2 => &[
                -0.12940952255092145,
                0.22414386804185735,
                0.836516303737469,
                0.48296291314469025,
            ],
            Wavelet::Db4 => &[
                -0.010597401784997278,
                0.032883011666982945,
                0.030841381835986965,
                -0.18703481171888114,
                -0.02798376941698385,
                0.6308807679295904,
                0.7148465705525415,
                0.23037781330885523,
            ],
        }
    }

    /// High-pass decomposition filter coefficients.
    pub fn hi_d(&self) -> Vec<f64> {
        let lo = self.lo_d();
        let n = lo.len();
        (0..n)
            .map(|i| (-1.0f64).powi(i as i32) * lo[n - 1 - i])
            .collect()
    }

    /// Low-pass reconstruction filter coefficients.
    pub fn lo_r(&self) -> Vec<f64> {
        self.lo_d().iter().rev().copied().collect()
    }

    /// High-pass reconstruction filter coefficients.
    pub fn hi_r(&self) -> Vec<f64> {
        let hi = self.hi_d();
        hi.iter().rev().copied().collect()
    }
}

/// 1D Discrete Wavelet Transform (single level) with periodic boundary.
///
/// Returns (approximation_coefficients, detail_coefficients).
pub fn dwt(signal: &[f64], wavelet: Wavelet) -> (Vec<f64>, Vec<f64>) {
    let lo = wavelet.lo_d();
    let hi = wavelet.hi_d();
    let n = signal.len();
    let fl = lo.len();
    let out_len = n / 2;

    let mut approx = Vec::with_capacity(out_len);
    let mut detail = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let mut a = 0.0;
        let mut d = 0.0;
        for k in 0..fl {
            let idx = (2 * i + fl - 1 - k) % n;
            a += signal[idx] * lo[k];
            d += signal[idx] * hi[k];
        }
        approx.push(a);
        detail.push(d);
    }

    (approx, detail)
}

/// 1D Inverse Discrete Wavelet Transform (single level) with periodic boundary.
pub fn idwt(approx: &[f64], detail: &[f64], wavelet: Wavelet) -> Vec<f64> {
    let lo = wavelet.lo_r();
    let hi = wavelet.hi_r();
    let fl = lo.len();
    let half = approx.len();
    let n = half * 2;

    let mut result = vec![0.0; n];

    for i in 0..half {
        for k in 0..fl {
            let idx = (2 * i + k) % n;
            result[idx] += approx[i] * lo[k] + detail[i] * hi[k];
        }
    }

    result
}

/// Multi-level DWT decomposition.
///
/// Returns `[detail_level_n, ..., detail_level_1, approx_level_n]`.
pub fn wavedec(signal: &[f64], wavelet: Wavelet, levels: usize) -> Vec<Vec<f64>> {
    let mut coeffs = Vec::with_capacity(levels + 1);
    let mut current = signal.to_vec();

    for _ in 0..levels {
        let (approx, detail) = dwt(&current, wavelet);
        coeffs.push(detail);
        current = approx;
    }
    coeffs.push(current); // final approximation
    coeffs.reverse(); // [approx, detail_1, detail_2, ..., detail_n]
    coeffs
}

/// Multi-level DWT reconstruction.
///
/// Input format: `[approx, detail_1, detail_2, ..., detail_n]` (same as wavedec output).
pub fn waverec(coeffs: &[Vec<f64>], wavelet: Wavelet) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![];
    }
    let mut current = coeffs[0].clone();
    for i in 1..coeffs.len() {
        current = idwt(&current, &coeffs[i], wavelet);
    }
    current
}

/// Wavelet family for CWT (Continuous Wavelet Transform).
#[derive(Debug, Clone, Copy)]
pub enum CwtWavelet {
    /// Morlet wavelet (complex sinusoid × Gaussian)
    Morlet,
    /// Mexican hat (Ricker) wavelet
    MexicanHat,
}

/// Continuous Wavelet Transform.
///
/// Computes the CWT of `signal` at the given `scales`.
/// Returns a 2D vector: `result[scale_idx][time_idx]`.
pub fn cwt(signal: &[f64], scales: &[f64], wavelet: CwtWavelet) -> Vec<Vec<f64>> {
    let n = signal.len();
    scales
        .iter()
        .map(|&scale| {
            (0..n)
                .map(|t| {
                    let mut sum = 0.0;
                    for (k, &s) in signal.iter().enumerate() {
                        let x = (k as f64 - t as f64) / scale;
                        let psi = match wavelet {
                            CwtWavelet::Morlet => (-0.5 * x * x).exp() * (5.0 * x).cos(),
                            CwtWavelet::MexicanHat => {
                                let x2 = x * x;
                                (2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25)))
                                    * (1.0 - x2)
                                    * (-0.5 * x2).exp()
                            }
                        };
                        sum += s * psi;
                    }
                    sum / scale.sqrt()
                })
                .collect()
        })
        .collect()
}
