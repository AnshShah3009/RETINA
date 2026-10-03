//! Emit signal-processing values for `parity/parity_signal.py` to recompute.
//!
//! Prints only; asserts nothing. The comparison lives on the Python side so that
//! SciPy remains an independent reference rather than something this file agrees
//! with by construction.
//!
//! Run: `cargo run -q -p cv-signal --example sp_parity`

use cv_signal::signal::{butter, filtfilt};

/// A deterministic band-limited input: two sinusoids plus a DC offset.
///
/// Chosen so the spectrum is known and the filter has something real to reject —
/// a constant or white noise would exercise only the edge cases.
fn input() -> Vec<f64> {
    let n = 400;
    let fs = 1000.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            127.0
                + 100.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
                + 40.0 * (2.0 * std::f64::consts::PI * 17.0 * t).cos()
        })
        .collect()
}

fn main() {
    let fs = 1000.0;
    let x = input();

    // Three configurations: the second-order case where `butter` gives an
    // analytic form, and two fourth-order cases where it recurses.
    for &(order, cutoff) in &[(2usize, 50.0f64), (4, 50.0), (4, 120.0)] {
        let (b, a) = butter(order, cutoff, fs);
        println!("#B {order} {cutoff} {}", b.len());
        for v in &b {
            println!("#BB {v:.17e}");
        }
        for v in &a {
            println!("#BA {v:.17e}");
        }
        for v in &filtfilt(&b, &a, &x) {
            println!("#Y {v:.17e}");
        }
    }
}
