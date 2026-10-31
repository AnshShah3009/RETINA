//! `butter_bandpass` against `scipy.signal.butter(N, [lo, hi], btype="bandpass", fs=fs)`.
//!
//! The passband gain was normalised to 1 at the arithmetic centre
//! `(lo + hi) / 2` instead of at the bilinear image of the prewarped geometric
//! centre, `2 atan(sqrt(tan(π lo / fs) tan(π hi / fs)))`, where the analog
//! prototype has unit gain. Every `b` coefficient was off by the same factor.

use cv_signal::signal::butter_bandpass;

fn close(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-12, "{name}[{i}] = {g}, SciPy {w}");
    }
}

#[test]
fn second_order_bandpass_matches_scipy() {
    let (b, a) = butter_bandpass(2, 100.0, 300.0, 1000.0);
    close("b", &b, &BP_B);
    close("a", &a, &BP_A);
}

#[test]
fn third_order_wide_bandpass_matches_scipy() {
    let (b, a) = butter_bandpass(3, 50.0, 400.0, 1000.0);
    close("b", &b, &BP3_B);
    close("a", &a, &BP3_A);
}

// ── SciPy reference coefficients ──
const BP_B: [f64; 5] = [
    0.2065720838261479,
    0.0,
    -0.4131441676522958,
    0.0,
    0.2065720838261479,
];
const BP_A: [f64; 5] = [
    1.0,
    -0.905078920874777,
    0.597907856327917,
    -0.2907367917810567,
    0.19581571265583306,
];
const BP3_B: [f64; 7] = [
    0.37445269259015945,
    0.0,
    -1.1233580777704784,
    0.0,
    1.1233580777704784,
    0.0,
    -0.37445269259015945,
];
const BP3_A: [f64; 7] = [
    1.0,
    -0.6333600709851303,
    -0.9990030426950665,
    0.3049157098036046,
    0.6741771163177895,
    -0.09739760265563574,
    -0.13776130125989283,
];
