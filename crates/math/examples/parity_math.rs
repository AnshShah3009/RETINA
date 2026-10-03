//! Numerical-parity number generator: special functions & statistics
//! (`cv-math`) vs SciPy. Companion reference side: `parity/parity_math.py`.
//!
//! Every line is `<name> <value:.17e>`; there is no framing ambiguity because
//! every record is a single scalar with a fixed name.

use cv_math::special::{bessel_j0, bessel_j1, erf, erfc, gamma};
use cv_math::stats::{mean, percentile, std_dev, variance};

/// Deterministic sweep of `erf`/`erfc` arguments. Dense where the interesting
/// structure is: `erf` transitions through its whole range over |x| < 4, and
/// `erfc` is only above the f64 noise floor up to about x = 27.
fn erf_grid() -> Vec<f64> {
    let mut v = Vec::new();
    // Fine grid over the transition, including exact 0 and tiny values.
    for i in -80..=80 {
        v.push(i as f64 * 0.05);
    }
    // Zero and a log-spaced sweep into the tail.
    v.push(0.0);
    for i in 1..=40 {
        v.push(i as f64 * 0.5);
    }
    for i in 1..=12 {
        v.push(5.0 + i as f64);
    }
    v
}

/// `erf`/`erfc` only; `erf`'s tail is past 1.0 where f64 cannot resolve more.
fn erfc_grid() -> Vec<f64> {
    let mut v = vec![0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
    for i in 1..=20 {
        v.push(0.25 * i as f64);
    }
    for i in 1..=60 {
        v.push(0.5 * i as f64);
    }
    for i in 1..=20 {
        v.push(25.0 + i as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// `gamma` over the region where f64 represents it without overflow, plus the
/// reflection branch on (-1, 0).
fn gamma_grid() -> Vec<f64> {
    let mut v = Vec::new();
    for i in 1..=120 {
        v.push(i as f64 * 0.05);
    }
    for i in 1..=40 {
        v.push(0.5 * i as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// `j0`/`j1`: dense across both branch regions of the implementation
/// (rational fit below 8, asymptotic form at and above 8).
fn bessel_grid() -> Vec<f64> {
    let mut v = Vec::new();
    for i in 0..=800 {
        v.push(i as f64 * 0.01);
    }
    for i in 0..=400 {
        v.push(7.8 + i as f64 * 0.01);
    }
    v
}

/// Deterministic data set for the statistics comparison: a sinusoid plus a
/// small integer-valued perturbation, so the values are exactly representable
/// in f64 on both sides (no RNG, no round-trip disagreement possible).
fn stat_data() -> Vec<f64> {
    (0..200)
        .map(|i| {
            let t = i as f64;
            100.0 + 30.0 * (0.05 * t).sin() + (i % 7) as f64 - 3.0
        })
        .collect()
}

fn main() {
    for &x in &erf_grid() {
        println!("erf {:.17e} {:.17e}", x, erf(x));
    }
    for &x in &erfc_grid() {
        println!("erfc {:.17e} {:.17e}", x, erfc(x));
    }
    for &x in &gamma_grid() {
        let g = gamma(x);
        // Gamma is exact-zero-adjacent only at poles; guard the printout so a
        // NaN cannot be silently read as agreement.
        println!("gamma {:.17e} {:.17e}", x, g);
    }
    for &x in &bessel_grid() {
        println!("bessel_j0 {:.17e} {:.17e}", x, bessel_j0(x));
        println!("bessel_j1 {:.17e} {:.17e}", x, bessel_j1(x));
    }

    // Statistics: mean/var/std over a deterministic sample, plus percentiles
    // at the classic interpolation-defining quantiles.
    let d = stat_data();
    println!("stat_mean {:.17e}", mean(&d));
    println!("stat_var {:.17e}", variance(&d));
    println!("stat_std {:.17e}", std_dev(&d));
    for p in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0] {
        println!("stat_pct {:.17e} {:.17e}", p, percentile(&d, p));
    }
}
