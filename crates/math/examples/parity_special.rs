//! Numerical-parity number generator for the SciPy special-function surface
//! (`cv-math::special`) vs `scipy.special`. Companion reference side:
//! `parity/parity_special.py`.
//!
//! **This file prints and asserts nothing**, by design. Every defect it used to
//! hide was found by an *independent* reference recomputing the same quantity
//! with SciPy; a Rust-side assertion can only compare the Rust side against a
//! constant someone typed in. The assertions live in
//! `crates/math/tests/special_fixes.rs` and in the Python harness.
//!
//! Every line is `<fn> <args...> <value:.17e>`, so the Python side can key on
//! the function name and re-evaluate with SciPy without any framing ambiguity.
//! Records whose value is NaN are printed as `nan`, which the Python side
//! compares as NaN-to-NaN rather than silently reading as agreement.

use cv_math::special::{
    bessel_i0, bessel_j0, bessel_j1, bessel_jn, bessel_k0, bessel_y0, bessel_y1, bessel_yn, beta,
    double_factorial, erf, erfc, erfi, expi, expn, factorial, gamma, log_beta, log_gamma,
    spherical_jn, spherical_yn,
};

/// Dense where the interesting structure is: `erf` traverses its whole range
/// over |x| < 6 and `erfc` is above the f64 noise floor out to about x = 26.5
/// (the true `erfc(27)` is 5.24e-319, subnormal). Includes exact zero, tiny
/// magnitudes where the old A&S `erf` offset dominated, both sides of the 0.5
/// `erfc` branch point, and the oddness pairs `(+x, -x)`.
fn erf_grid() -> Vec<f64> {
    let mut v = Vec::new();
    for i in -120..=120 {
        v.push(i as f64 * 0.05); // [-6, 6] dense through the transition
    }
    v.push(0.0);
    v.push(-0.0);
    for e in -30..=1 {
        v.push(10f64.powi(e)); // 1e-30 .. 10, log-spaced into the tiny region
    }
    for i in 0..=400 {
        v.push(i as f64 * 0.01); // [0, 4]
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// `erf`'s oddness pairs only, so the Python side can check oddness directly
/// without re-deriving the grid.
fn oddness_grid() -> Vec<f64> {
    let mut v: Vec<f64> = (0..=200).map(|i| i as f64 * 0.03).collect();
    v.push(1e-12);
    v.push(1e-8);
    v.push(1e-4);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// `erfc` on its own grid, dense through the `1 - erf` / continued-fraction
/// branch point at 0.5 and out through the subnormal tail.
fn erfc_grid() -> Vec<f64> {
    let mut v = vec![0.0, -0.0, -1e-12, -0.5, -1.0, -5.0, -26.0];
    for i in 1..=200 {
        v.push(i as f64 * 0.0025); // [0, 0.5]
    }
    for i in 0..=530 {
        v.push(0.5 * i as f64); // [0, 26.5]
    }
    for i in 1..=20 {
        v.push(25.0 + i as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// Non-finite and limiting arguments, printed explicitly so the Python side
/// compares `erf(inf)`, `erfc(-inf)` etc. against SciPy rather than leaving
/// them to a grid that never reaches them.
fn special_values() -> Vec<f64> {
    vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.0,
        -0.0,
        1.0,
        -1.0,
    ]
}

/// `gamma`/`log_gamma` over the region f64 represents without overflow, plus
/// the negative-reflection intervals (-1,0), (-2,-1), (-3,-2), (-4,-3), and
/// the non-positive-integer poles.
fn gamma_grid() -> Vec<f64> {
    let mut v = Vec::new();
    for i in 1..=60 {
        v.push(i as f64 * 0.05); // (0, 3]
    }
    for i in 1..=80 {
        v.push(i as f64 * 0.5); // (0, 40]
    }
    v.push(-0.5);
    v.push(-0.25);
    v.push(-0.75);
    v.push(-1.5);
    v.push(-2.5);
    v.push(-3.5);
    v.push(-4.5);
    v.push(2.5);
    v.push(171.0);
    v.push(172.0);
    v.push(170.5);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// Non-positive integers: the poles of Gamma. Printed separately so a Python
/// `isinf`/`isnan` check can confirm the Rust side's answer is *distinguishable
/// from a valid finite value* rather than merely "large".
fn gamma_poles() -> Vec<f64> {
    vec![0.0, -0.0, -1.0, -2.0, -3.0, -10.0]
}

/// `beta`/`log_beta`: ordinary values, the large-but-representable corner that
/// overflows the naive `gamma(x)*gamma(y)` product, and the non-positive
/// arguments where Beta has poles or a sign change.
fn beta_pairs() -> Vec<(f64, f64)> {
    vec![
        (0.5, 0.5),
        (2.0, 3.0),
        (0.3, 0.7),
        (10.0, 10.0),
        (0.1, 0.1),
        (1.0, 1.0),
        (50.0, 50.0),
        (150.0, 150.0),
        (200.0, 200.0),
        (0.5, 1.0),
        (0.25, 0.25),
        (3.5, 2.5),
        (20.0, 30.0),
    ]
}

/// Bessel arguments: dense across both branch regions of each implementation
/// (rational fit below 8, asymptotic at and above 8) plus the order range.
fn bessel_grid() -> Vec<f64> {
    let mut v = Vec::new();
    for i in 0..=900 {
        v.push(i as f64 * 0.01); // [0, 9]
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// Integer orders, including 0, 1, and orders far larger than x (the regime
/// where upward recurrence is unstable and Miller's backward algorithm must
/// take over).
fn bessel_orders() -> Vec<i32> {
    (0..=25).collect()
}

fn print_special(name: &str, args: &[f64], v: f64) {
    let mut line = name.to_string();
    for a in args {
        line.push_str(&format!(" {a:.17e}"));
    }
    println!("{line} {v:.17e}");
}

fn main() {
    // ---- erf -------------------------------------------------------------
    for &x in &erf_grid() {
        print_special("erf", &[x], erf(x));
    }
    for &x in &oddness_grid() {
        print_special("erf_pos", &[x], erf(x));
        print_special("erf_neg", &[x], erf(-x));
    }
    for &x in &special_values() {
        print_special("erf_inf", &[x], erf(x));
    }

    // ---- erfc ------------------------------------------------------------
    for &x in &erfc_grid() {
        print_special("erfc", &[x], erfc(x));
    }
    for &x in &special_values() {
        print_special("erfc_inf", &[x], erfc(x));
    }

    // ---- erfi ------------------------------------------------------------
    for &x in &[
        0.0, 1e-8, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 6.0, 6.5, 7.0, 10.0,
    ] {
        print_special("erfi", &[x], erfi(x));
        print_special("erfi_neg", &[x], erfi(-x));
    }

    // ---- gamma / log_gamma ----------------------------------------------
    for &x in &gamma_grid() {
        print_special("gamma", &[x], gamma(x));
        print_special("log_gamma", &[x], log_gamma(x));
    }
    for &x in &gamma_poles() {
        print_special("gamma_pole", &[x], gamma(x));
        print_special("log_gamma_pole", &[x], log_gamma(x));
    }

    // ---- beta / log_beta -------------------------------------------------
    for &(x, y) in &beta_pairs() {
        print_special("beta", &[x, y], beta(x, y));
        print_special("log_beta", &[x, y], log_beta(x, y));
    }

    // ---- factorial / double_factorial -----------------------------------
    for n in [0u64, 1, 2, 5, 10, 20, 50, 100, 170, 171, 172, 200, 300] {
        print_special("factorial", &[n as f64], factorial(n));
    }
    for n in [0u64, 1, 2, 3, 4, 5, 10, 20, 100, 170, 171, 200, 300] {
        print_special("double_factorial", &[n as f64], double_factorial(n));
    }

    // ---- Bessel J / Y ----------------------------------------------------
    for &x in &bessel_grid() {
        print_special("bessel_j0", &[x], bessel_j0(x));
        print_special("bessel_j1", &[x], bessel_j1(x));
        print_special("bessel_y0", &[x], bessel_y0(x));
        print_special("bessel_y1", &[x], bessel_y1(x));
        print_special("bessel_i0", &[x], bessel_i0(x));
        print_special("bessel_k0", &[x], bessel_k0(x));
    }
    for &n in &bessel_orders() {
        for &x in &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            print_special("bessel_jn", &[n as f64, x], bessel_jn(n, x));
            print_special("bessel_yn", &[n as f64, x], bessel_yn(n, x));
            print_special("spherical_jn", &[n as f64, x], spherical_jn(n, x));
            print_special("spherical_yn", &[n as f64, x], spherical_yn(n, x));
        }
    }
    // Parity of the order-dependent functions: J_n(-x) = (-1)^n J_n(x) and
    // Y_n(-x) are both undefined for real x < 0 (Y is on the branch cut), but
    // j_n(-x) = (-1)^n j_n(x) must hold exactly.
    for &n in &[0.0, 1.0, 2.0, 3.0, 10.0, 20.0] {
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            let ni = n as i32;
            print_special("bessel_jn_neg", &[n, x], bessel_jn(ni, -x));
            print_special("spherical_jn_neg", &[n, x], spherical_jn(ni, -x));
            print_special("bessel_jn_negorder", &[n, x], bessel_jn(-ni, x));
        }
    }

    // ---- expn / expi -----------------------------------------------------
    for &n in &[0, 1, 2, 3, 5] {
        for &x in &[0.0, 1e-8, 0.1, 0.5, 1.0, 2.0, 5.0, 12.0, 25.0, 50.0] {
            print_special("expn", &[n as f64, x], expn(n, x));
        }
    }
    for &x in &[
        -10.0, -5.0, -1.0, -0.5, 0.0, 1e-8, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0,
    ] {
        print_special("expi", &[x], expi(x));
    }
}
