//! Shared interpolation kernels used by the resampling code.
//!
//! `resize.rs` and `geometry.rs` previously each defined byte-identical copies
//! of `cubic_kernel` and `lanczos_kernel`; both now use this single module.

/// Interpolating Catmull-Rom kernel (Keys, a = -0.5).
pub(crate) fn cubic_kernel(d: f32) -> f32 {
    let a = d.abs();
    if a <= 1.0 {
        1.5 * a * a * a - 2.5 * a * a + 1.0
    } else if a < 2.0 {
        -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0
    } else {
        0.0
    }
}

/// Lanczos window (a = 3).
pub(crate) fn lanczos_kernel(d: f32) -> f32 {
    const A: f32 = 3.0;
    if d == 0.0 {
        1.0
    } else if d.abs() >= A {
        0.0
    } else {
        let p = std::f32::consts::PI * d;
        (p.sin() / p) * ((p / A).sin() / (p / A))
    }
}
