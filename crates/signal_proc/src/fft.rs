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
    if input.is_empty() {
        // `fft` returns an empty spectrum for empty input, so slicing
        // `..n/2 + 1` (== `..1`) would panic.
        return Vec::new();
    }
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
    let half = n.div_ceil(2); // ceil(n/2)
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
    let nyquist = if n.is_multiple_of(2) {
        n_freqs - 1
    } else {
        n_freqs
    };
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
    fn test_rfft_empty_input() {
        // Regression: rfft(&[]) used to slice `spectrum[..1]` on an empty
        // Vec returned by fft, panicking.
        assert!(rfft(&[]).is_empty());
        assert_eq!(rfft(&[1.0, 2.0, 3.0, 4.0]).len(), 3);
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

    #[test]
    fn test_dct_roundtrip() {
        let x: Vec<f64> = (0..8).map(|i| (i as f64 * 0.5).sin()).collect();
        let c = dct(&x);
        let r = idct(&c);
        for (a, b) in x.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-10, "DCT roundtrip: {} vs {}", a, b);
        }
    }

    fn signal(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 + (i as f64 * 0.01).sin()
            })
            .collect()
    }

    /// `max |a - b| / ||x||_2` (the transforms are orthonormal, so outputs
    /// have the input's norm).
    fn rel_err(a: &[f64], b: &[f64], x: &[f64]) -> f64 {
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max)
            / norm
    }

    /// FFT-based DCT-II / DCT-III / DST-I against the O(n^2) definitions they
    /// replaced: odd, even, prime, power-of-two lengths around the switch-over.
    #[test]
    fn fft_dct_dst_match_direct_definitions() {
        let lengths = [
            0usize, 1, 2, 3, 7, 16, 17, 18, 31, 32, 33, 64, 100, 127, 255, 256, 257, 1000, 1024,
            2047,
        ];
        for (seed, &n) in lengths.iter().enumerate() {
            let x = signal(n, seed as u64);
            let tol = 1e-12;
            let e = rel_err(&dct(&x), &dct_direct(&x), &x);
            assert!(e <= tol, "dct n={n}: {e:e}");
            let e = rel_err(&idct(&x), &idct_direct(&x), &x);
            assert!(e <= tol, "idct n={n}: {e:e}");
            let e = rel_err(&dst(&x), &dst_direct(&x), &x);
            assert!(e <= tol, "dst n={n}: {e:e}");
            let e = rel_err(&idct(&dct(&x)), &x, &x);
            assert!(e <= tol, "idct(dct) n={n}: {e:e}");
            let e = rel_err(&idst(&dst(&x)), &x, &x);
            assert!(e <= tol, "idst(dst) n={n}: {e:e}");
        }
    }

    /// `cargo test --release -p cv-signal --lib dct_dst_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dct_dst_timing() {
        use std::time::Instant;
        let time = |f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], reps: usize| {
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(f(x));
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1e3
        };
        for n in [64usize, 256, 1024, 4096, 16384] {
            let x = signal(n, 1);
            let reps = if n >= 4096 { 3 } else { 50 };
            let worst = [
                rel_err(&dct(&x), &dct_direct(&x), &x),
                rel_err(&idct(&x), &idct_direct(&x), &x),
                rel_err(&dst(&x), &dst_direct(&x), &x),
            ]
            .into_iter()
            .fold(0.0, f64::max);
            println!(
                "n={n:5}: dct {:9.3} -> {:7.3} ms, idct {:9.3} -> {:7.3} ms, dst {:9.3} -> {:7.3} ms, max rel diff {worst:.1e}",
                time(&dct_direct, &x, reps),
                time(&dct, &x, reps),
                time(&idct_direct, &x, reps),
                time(&idct, &x, reps),
                time(&dst_direct, &x, reps),
                time(&dst, &x, reps),
            );
        }
    }

    #[test]
    fn test_dst_roundtrip() {
        let x: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).cos()).collect();
        let s = dst(&x);
        let r = idst(&s);
        for (a, b) in x.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-10, "DST roundtrip: {} vs {}", a, b);
        }
    }
}

// ── DCT / DST (Discrete Cosine / Sine Transform) ───────────────────────────
//
// Each transform is evaluated through one complex FFT (O(n log n)); below
// `DIRECT_MAX_LEN` points the O(n^2) definition is cheaper than planning an
// FFT and is used as-is. Both agree to ~1e-15 relative to the signal norm
// (see the tests).

/// Lengths up to this use the O(n^2) direct sums.
const DIRECT_MAX_LEN: usize = 16;

/// Type-II Discrete Cosine Transform (DCT-II), the "standard" DCT.
///
/// Orthonormal scaling: `X[0] = sqrt(1/N) * sum(x)`, `X[k] = sqrt(2/N) *
/// sum_i x[i] cos(pi (i + 1/2) k / N)`. Used in JPEG compression, BM3D, and
/// signal processing.
pub fn dct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n <= DIRECT_MAX_LEN {
        return dct_direct(input);
    }
    // Makhoul (1980): with v = [x0, x2, x4, ..., x5, x3, x1] (even samples
    // forward, odd samples backward) and V = FFT(v),
    // sum_i x[i] cos(pi (2i + 1) k / 2N) = Re(exp(-i pi k / 2N) V[k]).
    let mut v: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for (i, &x) in input.iter().enumerate() {
        let j = if i % 2 == 0 { i / 2 } else { n - 1 - i / 2 };
        v[j] = Complex::new(x, 0.0);
    }
    FftPlanner::<f64>::new().plan_fft_forward(n).process(&mut v);
    let norm0 = (1.0 / n as f64).sqrt();
    let norm = (2.0 / n as f64).sqrt();
    v.iter()
        .enumerate()
        .map(|(k, c)| {
            let (s, co) = (-std::f64::consts::PI * k as f64 / (2 * n) as f64).sin_cos();
            let re = c.re * co - c.im * s;
            re * if k == 0 { norm0 } else { norm }
        })
        .collect()
}

/// Type-III Discrete Cosine Transform (inverse of the orthonormal [`dct`]).
pub fn idct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n <= DIRECT_MAX_LEN {
        return idct_direct(input);
    }
    // Undo the orthonormal scaling to unnormalised DCT-II coefficients X
    // (x[i] = (X[0] + 2 sum_{k>0} X[k] cos(..)) / N), rebuild the FFT of
    // Makhoul's reordered sequence, V[k] = exp(i pi k / 2N) (X[k] - i X[N-k])
    // with X[N] = 0, invert it, and undo the reordering.
    let nf = n as f64;
    let x_un = |k: usize| -> f64 {
        if k == 0 {
            input[0] * nf * (1.0 / nf).sqrt()
        } else if k < n {
            input[k] * nf * 0.5 * (2.0 / nf).sqrt()
        } else {
            0.0
        }
    };
    let mut v: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let (s, co) = (std::f64::consts::PI * k as f64 / (2 * n) as f64).sin_cos();
            let (a, b) = (x_un(k), -x_un(n - k));
            // (co + i s) * (a + i b)
            Complex::new(co * a - s * b, co * b + s * a)
        })
        .collect();
    FftPlanner::<f64>::new().plan_fft_inverse(n).process(&mut v);
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate() {
        let j = if i % 2 == 0 { i / 2 } else { n - 1 - i / 2 };
        *o = v[j].re / nf;
    }
    out
}

/// Discrete Sine Transform (DST-I, orthogonal).
///
/// Uses the Type-I DST basis: `sin(pi * (i+1) * (k+1) / (N+1))`.
/// Orthogonal and self-inverse (up to scaling): `idst(dst(x)) == x`.
pub fn dst(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n <= DIRECT_MAX_LEN {
        return dst_direct(input);
    }
    // Odd extension a = [0, x0..x_{N-1}, 0, -x_{N-1}..-x0] of length 2N + 2:
    // FFT(a)[m] = -2i sum_i x[i] sin(pi (i + 1) m / (N + 1)).
    let m = 2 * n + 2;
    let mut a: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); m];
    for (i, &x) in input.iter().enumerate() {
        a[i + 1] = Complex::new(x, 0.0);
        a[m - 1 - i] = Complex::new(-x, 0.0);
    }
    FftPlanner::<f64>::new().plan_fft_forward(m).process(&mut a);
    let norm = (2.0 / (n + 1) as f64).sqrt();
    (0..n).map(|k| -a[k + 1].im * 0.5 * norm).collect()
}

/// O(n^2) evaluation of [`dct`] straight from the definition.
#[allow(clippy::needless_range_loop)]
fn dct_direct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mut output = vec![0.0; n];
    let scale = std::f64::consts::PI / n as f64;
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += input[i] * (scale * (i as f64 + 0.5) * k as f64).cos();
        }
        output[k] = sum;
    }
    // Orthogonal normalization
    output[0] *= (1.0 / n as f64).sqrt();
    let norm = (2.0 / n as f64).sqrt();
    for k in 1..n {
        output[k] *= norm;
    }
    output
}

/// O(n^2) evaluation of [`idct`] straight from the definition.
#[allow(clippy::needless_range_loop)]
fn idct_direct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mut output = vec![0.0; n];
    let scale = std::f64::consts::PI / n as f64;
    let norm0 = (1.0 / n as f64).sqrt();
    let norm = (2.0 / n as f64).sqrt();
    for i in 0..n {
        let mut sum = input[0] * norm0;
        for k in 1..n {
            sum += input[k] * norm * (scale * k as f64 * (i as f64 + 0.5)).cos();
        }
        output[i] = sum;
    }
    output
}

/// O(n^2) evaluation of [`dst`] straight from the definition.
#[allow(clippy::needless_range_loop)]
fn dst_direct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mut output = vec![0.0; n];
    let scale = std::f64::consts::PI / (n + 1) as f64;
    let norm = (2.0 / (n + 1) as f64).sqrt();
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += input[i] * (scale * (i as f64 + 1.0) * (k as f64 + 1.0)).sin();
        }
        output[k] = sum * norm;
    }
    output
}

/// Inverse Discrete Sine Transform (DST-I is self-inverse with orthogonal normalization).
pub fn idst(input: &[f64]) -> Vec<f64> {
    // DST-I is its own inverse with orthogonal normalization
    dst(input)
}
