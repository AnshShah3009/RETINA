//! Regression tests for the trailing-block handling of `convolve_row_1d`
//! (`crates/hal/src/cpu/simd.rs`), the SIMD row kernel used by every
//! separable filter in `cv-imgproc` (sobel, scharr, gaussian blur, ...).
//!
//! Defect under test: the scalar fallback loop used to start one whole SIMD
//! block *past* the end of the row (`scalar_start == width`), so whenever
//! `width % 8 != 0` the rightmost `width % 8` pixels of `dst` were never
//! written. Because the caller (`crates/imgproc/src/convolve.rs`) zero-initialises
//! its scratch buffer, those pixels stayed `0.0`, which corrupted the final
//! image column for every odd width.

use cv_hal::cpu::simd::convolve_row_1d;

/// Reference (pure scalar) implementation of the same row convolution.
/// `src` is the *padded* row; `width` is the number of output pixels.
fn convolve_row_1d_reference(src: &[f32], width: usize, kernel: &[f32]) -> Vec<f32> {
    (0..width)
        .map(|x| {
            let mut sum = 0.0f32;
            for (k, w) in kernel.iter().enumerate() {
                let idx = x + k;
                if idx < src.len() {
                    sum += src[idx] * *w;
                }
            }
            sum
        })
        .collect()
}

/// Build a padded row of `width + 2 * radius` samples from a ramp `0..width`,
/// mirroring what `separable_convolve_into_ctx` does per row.
fn padded_ramp(width: usize, radius: usize) -> Vec<f32> {
    (0..width + 2 * radius)
        .map(|i| (i as isize - radius as isize).clamp(0, width as isize - 1) as f32)
        .collect()
}

/// A smooth, strictly positive kernel so every output pixel depends on data
/// and a `0.0` output is unambiguously wrong.
fn kernel3() -> Vec<f32> {
    vec![0.25, 0.5, 0.25]
}

fn assert_parity(width: usize, kernel: &[f32], radius: usize, label: &str) {
    let src = padded_ramp(width, radius);
    let mut dst = vec![f32::NAN; width]; // poison: every element must be written
    convolve_row_1d(&src, &mut dst, kernel, radius);
    let expected = convolve_row_1d_reference(&src, width, kernel);

    assert_eq!(dst.len(), expected.len());
    for (x, (got, want)) in dst.iter().zip(expected.iter()).enumerate() {
        assert!(
            !got.is_nan(),
            "[{label}] width={width}: dst[{x}] was never written (still poisoned/0)"
        );
        assert!(
            (got - want).abs() < 1e-5,
            "[{label}] width={width}: dst[{x}] = {got}, expected {want}\n  got:      {dst:?}\n  expected: {expected:?}"
        );
    }
}

#[test]
fn convolve_row_1d_tail_written_for_all_widths() {
    let kernel = kernel3();
    let radius = 1;
    // The bug is specifically about `width % 8 != 0`; cover multiples of 8
    // (SIMD-only path) and every remainder class.
    for width in [
        1, 2, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41, 64, 65, 72, 73,
    ] {
        assert_parity(width, &kernel, radius, "ramp/r3");
    }
}

#[test]
fn convolve_row_1d_tail_written_for_wide_kernels() {
    // radius 2 and 3 kernels: the padded src length changes, exercising the
    // same loop structure with different k_len.
    for k_len in [3usize, 5, 7] {
        let radius = k_len / 2;
        let kernel: Vec<f32> = (0..k_len)
            .map(|k| {
                let d = (k as f32 - radius as f32).abs();
                (-d * d).exp()
            })
            .collect();
        let sum: f32 = kernel.iter().sum();
        let kernel: Vec<f32> = kernel.iter().map(|v| v / sum).collect();
        for width in [8, 9, 12, 15, 16, 17, 23, 24, 25, 33] {
            assert_parity(width, &kernel, radius, &format!("gauss/k{k_len}"));
        }
    }
}

#[test]
fn convolve_row_1d_tail_written_for_random_rows() {
    // Deterministic pseudo-random rows: catches asymmetric/off-by-N errors that
    // a monotone ramp can hide.
    let kernel = kernel3();
    let radius = 1;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 251) as f32
    };
    for width in [8, 9, 15, 16, 17, 25] {
        let src: Vec<f32> = (0..width + 2 * radius).map(|_| next()).collect();
        let mut dst = vec![-12345.0f32; width];
        convolve_row_1d(&src, &mut dst, &kernel, radius);
        let expected = convolve_row_1d_reference(&src, width, &kernel);
        for (x, (got, want)) in dst.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "random width={width}: dst[{x}] = {got}, expected {want}"
            );
        }
    }
}

/// Control: for widths that are exact multiples of the SIMD block size the
/// result must be *bit-identical* to a hand-computed reference, proving the fix
/// did not perturb the SIMD path.
#[test]
fn convolve_row_1d_multiple_of_8_path_is_unchanged_control() {
    // Identity kernel on a ramp => dst == the visible (unpadded) ramp.
    let kernel = vec![0.0, 1.0, 0.0];
    for width in [8usize, 16, 24, 32] {
        let src = padded_ramp(width, 1);
        let mut dst = vec![f32::NAN; width];
        convolve_row_1d(&src, &mut dst, &kernel, 1);
        for (x, value) in dst.iter().enumerate() {
            assert_eq!(
                *value, x as f32,
                "width={width}: SIMD path result changed at column {x}"
            );
        }
    }

    // Blur kernel on width 16, checked against explicit per-pixel arithmetic.
    let kernel = kernel3();
    let src = padded_ramp(16, 1);
    let mut dst = vec![f32::NAN; 16];
    convolve_row_1d(&src, &mut dst, &kernel, 1);
    for x in 0..16 {
        let want = 0.25 * src[x] + 0.5 * src[x + 1] + 0.25 * src[x + 2];
        assert_eq!(
            dst[x], want,
            "width=16: SIMD path result changed at column {x}"
        );
    }
}
