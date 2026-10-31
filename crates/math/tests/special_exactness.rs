//! Exactness properties of the special functions - the class of defect the
//! Abramowitz & Stegun `erf` had: a function can be within its accuracy bound
//! everywhere and still be *silently wrong* at the points where mathematics
//! demands an exact value. `erf(0)` was `1e-9` instead of `0`, which also made
//! oddness false; the rational fits here have the same hazard.
//!
//! Each assertion below is a mathematical identity, not a tolerance: if one
//! fails, the implementation is wrong at that point, however small the error.

#![forbid(unsafe_code)]

use cv_math::special::{bessel_j0, bessel_j1, factorial, gamma};

#[test]
fn j0_and_j1_are_exact_at_zero() {
    // J0(0) = 1 and J1(0) = 0 are boundary values, not approximations: window
    // functions built from J0 have unity peak by this identity.
    assert_eq!(bessel_j0(0.0), 1.0, "J0(0) must be exactly 1");
    assert_eq!(bessel_j1(0.0), 0.0, "J1(0) must be exactly 0");
    assert_eq!(
        bessel_j1(-0.0).to_bits(),
        (-0.0f64).to_bits(),
        "J1 is odd, at 0 too"
    );
}

#[test]
fn j0_is_even_and_j1_is_odd_exactly() {
    for x in [0.3, 1.0, 2.5, 7.9, 8.1, 15.0, 100.0] {
        assert_eq!(
            bessel_j0(-x).to_bits(),
            bessel_j0(x).to_bits(),
            "J0(-x) must be bit-identical to J0(x) at x={x}"
        );
        assert_eq!(
            bessel_j1(-x).to_bits(),
            (-bessel_j1(x)).to_bits(),
            "J1(-x) must be bit-identical to -J1(x) at x={x}"
        );
    }
}

#[test]
fn factorial_is_exact_representable_integers() {
    // n! is exactly representable in f64 for n <= 22 (22! = 2^19 * 2144253445),
    // and for those the product path is exact by construction. Routing 20..=22
    // through `gamma` would hand back an approximation of a value f64 can hold
    // exactly - the same waste of exactness as `erf(0) != 0`.
    let mut acc = 1.0f64;
    for n in 0..=22u64 {
        if n >= 2 {
            acc *= n as f64;
        }
        assert_eq!(
            factorial(n),
            acc,
            "factorial({n}) must be the exact product, not a Gamma approximation"
        );
    }
}

#[test]
fn gamma_is_exact_at_small_positive_integers() {
    // Gamma(n) = (n-1)! for positive integers; f64 holds these exactly up to
    // 23!, so `gamma` may not approximate them through exp(log_gamma).
    for (x, want) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 2.0),
        (4.0, 6.0),
        (5.0, 24.0),
        (11.0, 3628800.0),
    ] {
        assert_eq!(
            gamma(x),
            want,
            "gamma({x}) must be exactly {want} - an integer-valued point of the \
             factorial function, not an approximation"
        );
    }
}
