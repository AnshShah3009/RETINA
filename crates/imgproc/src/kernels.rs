//! Shared interpolation kernels used by the resampling code.
//!
//! `resize.rs` and `geometry.rs` previously each defined byte-identical copies
//! of `cubic_kernel` and `lanczos_kernel`; both now use this single module.
//!
//! **Both kernels are OpenCV's choices**, so `Interpolation::Cubic` and
//! `Interpolation::Lanczos` mean what `cv2.resize(INTER_CUBIC)` and
//! `cv2.resize(INTER_LANCZOS4)` mean. This was measured, not assumed: with
//! these kernels the half-pixel resampler reproduces `INTER_CUBIC` output
//! **bit-for-bit**, and `INTER_LANCZOS4` to ±1 LSB (the residue is cv2's
//! 1/2048 fixed-point coefficient tables).

/// Interpolating cubic kernel with Keys parameter **a = -0.75** - OpenCV's
/// `interpolateCubic` (resize.cpp), *not* Catmull-Rom (a = -0.5). The previous
/// a = -0.5 form was the entire remaining divergence from `cv2.resize`'s
/// `INTER_CUBIC`: interior up to 3 LSB and 13 LSB at borders, all of it the
/// kernel, none of it arithmetic or borders.
pub(crate) fn cubic_kernel(d: f32) -> f32 {
    const A: f32 = -0.75;
    let x = d.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

/// Lanczos window (**a = 4**, eight taps) - OpenCV's `interpolateLanczos4`
/// (`INTER_LANCZOS4`), not the more common a = 3. A 3-lobe window is what the
/// name "Lanczos" usually means; OpenCV's is 4-lobe, and the parameter is part
/// of the parity contract.
pub(crate) fn lanczos_kernel(d: f32) -> f32 {
    const A: f32 = 4.0;
    if d == 0.0 {
        1.0
    } else if d.abs() >= A {
        0.0
    } else {
        let p = std::f32::consts::PI * d;
        (p.sin() / p) * ((p / A).sin() / (p / A))
    }
}
