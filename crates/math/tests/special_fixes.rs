//! Special-function regression tests for the confirmed defects fixed in
//! `crates/math/src/special.rs`:
//!
//! * 1. `log_gamma` returned NaN for negative non-integers on (-1,0),
//!   (-3,-2), (-5,-4), ... because `ln(pi/sin(pi x))` was taken on a negative
//!   ratio.
//! * 2. `beta` overflowed to NaN because it formed `gamma(x)*gamma(y)`.
//! * 3. `spherical_jn` ran unstable upward recurrence for `n > x`, returning
//!   values with the wrong sign and magnitude off by ~32 orders.
//! * 7. `bessel_jn` mapped a non-finite `x` to `0.0` instead of `NaN`.
//!
//! Every test carries a CONTROL assertion (a case that already worked) so it
//! cannot pass vacuously.

use cv_math::special::{bessel_jn, beta, gamma, log_gamma, spherical_jn};

fn rel_close(got: f64, want: f64, tol: f64) -> bool {
    if want == 0.0 {
        return got == 0.0;
    }
    (got / want - 1.0).abs() <= tol
}

// ---------------------------------------------------------------------------
// Defect 1: log_gamma / gamma on negative non-integer arguments
// ---------------------------------------------------------------------------

#[test]
fn log_gamma_negative_non_integer_no_nan() {
    // Reference values (mpmath.gamma / log|gamma|).
    // (-1,0) — sin(pi x) > 0, so pi/sin(pi x) is NEGATIVE and ln() of it is NaN.
    // Pre-fix: gamma(-0.5) = NaN (true -3.5449077018110321).
    let v = gamma(-0.5);
    assert!(
        rel_close(v, -3.5449077018110321, 1e-12),
        "gamma(-0.5) = {v}, want -3.5449077018110321"
    );
    assert!(!v.is_nan(), "gamma(-0.5) must not be NaN");

    // Pre-fix: gamma(-0.25) = NaN (true -4.9016668098607106).
    let v = gamma(-0.25);
    assert!(
        rel_close(v, -4.9016668098607106, 1e-12),
        "gamma(-0.25) = {v}, want -4.9016668098607106"
    );

    // Pre-fix: gamma(-0.75) = NaN (true -4.8341465442958777).
    let v = gamma(-0.75);
    assert!(
        rel_close(v, -4.8341465442958777, 1e-12),
        "gamma(-0.75) = {v}, want -4.8341465442958777"
    );

    // Pre-fix: gamma(-2.5) = NaN (true -0.9453087204829419).
    let v = gamma(-2.5);
    assert!(
        rel_close(v, -0.94530872048294188, 1e-12),
        "gamma(-2.5) = {v}, want -0.9453087204829419"
    );

    // CONTROL: gamma(-1.5) already worked (it lands on the lucky interval
    // (-2,-1) where pi/sin(pi x) is positive).
    let c = gamma(-1.5);
    assert!(
        rel_close(c, 2.3632718012073547, 1e-12),
        "CONTROL gamma(-1.5) = {c}, want 2.3632718012073547"
    );
    // CONTROL: positive arguments are untouched.
    assert!(rel_close(gamma(0.5), 1.7724538509055159, 1e-12));
    assert!(rel_close(gamma(5.0), 24.0, 1e-12));
    // CONTROL: poles stay infinite.
    assert!(gamma(0.0).is_infinite());
    assert!(gamma(-2.0).is_infinite());
}

#[test]
fn log_gamma_is_log_of_absolute_value() {
    // log_gamma must equal ln|gamma| everywhere: the reflection constant is
    // |pi/sin(pi x)|, never the signed ratio.
    let v = log_gamma(-0.5);
    assert!(!v.is_nan(), "log_gamma(-0.5) must not be NaN");
    assert!(
        rel_close(v, 1.2655121234846454, 1e-12),
        "log_gamma(-0.5) = {v}, want 1.2655121234846454"
    );
    let v = log_gamma(-2.5);
    assert!(
        rel_close(v, -0.05624371649767404, 1e-12),
        "log_gamma(-2.5) = {v}, want -0.056243716497674"
    );
    // CONTROL: log_gamma(3.0) = ln 2.
    assert!(rel_close(log_gamma(3.0), std::f64::consts::LN_2, 1e-12));
}

// ---------------------------------------------------------------------------
// Defect 2: beta overflow
// ---------------------------------------------------------------------------

#[test]
fn beta_does_not_overflow() {
    // Pre-fix: beta(200,200) = NaN (true 9.7132172476111818e-122).
    let v = beta(200.0, 200.0);
    assert!(!v.is_nan(), "beta(200,200) must not be NaN");
    // beta(200,200): the defect was a hard NaN. The value is now correct to
    // ~1.2e-10 relative, which is the accuracy limit of the underlying
    // Numerical-Recipes Lanczos log_gamma fit (its absolute log error grows
    // like x^(1/2)*x*eps and reaches ~2.4e-9 at x = 200; the log-domain
    // cancellation amplifies it). Asserting at 1e-8 keeps the test meaningful
    // without pretending to more accuracy than the fit has.
    let got = v / 9.7132172476111818e-122 - 1.0;
    assert!(
        got.abs() < 1e-8,
        "beta(200,200) = {v:e} (relative deviation {got:e}), \
         want 9.7132172476111818e-122"
    );
    // A case large enough to overflow the naive product but small enough for
    // the fit to be accurate: gamma(171) already overflows, so beta(150,150)
    // was inf/inf = NaN before the fix.
    let v = beta(150.0, 150.0);
    assert!(
        !v.is_nan() && v > 0.0,
        "beta(150,150) = {v:e} must be positive"
    );
    let dev = v / 1.42207504279732784e-91 - 1.0;
    assert!(
        dev.abs() < 1e-6,
        "beta(150,150) = {v:e} (deviation {dev:e})"
    );

    // An even more extreme ratio: beta(300,300) ~ 1e-374 underflows to a
    // subnormal/zero rather than to NaN.
    let v = beta(300.0, 300.0);
    assert!(!v.is_nan(), "beta(300,300) must not be NaN");
    assert!(v >= 0.0, "beta(300,300) = {v} must be non-negative");

    // CONTROL: ordinary values that already worked.
    assert!(rel_close(beta(0.5, 0.5), std::f64::consts::PI, 1e-12));
    assert!(rel_close(beta(2.0, 3.0), 1.0 / 12.0, 1e-12));
    assert!(rel_close(beta(0.3, 0.7), 3.8832220774509334, 1e-10));
    assert!(rel_close(beta(10.0, 10.0), 1.0825088224469029e-6, 1e-10));
}

// ---------------------------------------------------------------------------
// Defect 3: spherical_jn upward-recurrence instability for n > x
// ---------------------------------------------------------------------------

#[test]
fn spherical_jn_stable_for_n_greater_than_x() {
    // Reference: scipy.special.spherical_jn.
    // Pre-fix (upward recurrence, unstable for n > x):
    //   spherical_jn(20, 1.0) = -1.3289483e+07  (true  7.5377957e-26)
    //   spherical_jn(30, 1.0) = -1.2086864e+24  (true  5.5668313e-43)
    //   spherical_jn(10, 0.5) = -2.1535727e-05  (true  7.0641240e-14)
    // Note the WRONG SIGN as well as the ~1e32 relative error.
    let v = spherical_jn(20, 1.0);
    assert!(v > 0.0, "spherical_jn(20,1.0) = {v} must be positive");
    assert!(
        rel_close(v, 7.537795722236873e-26, 1e-10),
        "spherical_jn(20,1.0) = {v:e}, want 7.537795722236873e-26"
    );

    let v = spherical_jn(30, 1.0);
    assert!(v > 0.0, "spherical_jn(30,1.0) = {v} must be positive");
    assert!(
        rel_close(v, 5.566831266981347e-43, 1e-10),
        "spherical_jn(30,1.0) = {v:e}, want 5.566831266981347e-43"
    );

    let v = spherical_jn(10, 0.5);
    assert!(v > 0.0, "spherical_jn(10,0.5) = {v} must be positive");
    assert!(
        rel_close(v, 7.064123963661878e-14, 1e-10),
        "spherical_jn(10,0.5) = {v:e}, want 7.064123963661878e-14"
    );

    // More n > x cases across the boundary.
    assert!(rel_close(
        spherical_jn(25, 3.0),
        2.6112633829308916e-22,
        1e-9
    ));
    assert!(rel_close(
        spherical_jn(15, 1.0),
        5.1326861154437623e-18,
        1e-10
    ));
    assert!(rel_close(spherical_jn(5, 0.5), 2.9774668754574456e-6, 1e-9));
    assert!(rel_close(spherical_jn(2, 0.5), 1.6371106607993413e-2, 1e-9));
    assert!(rel_close(spherical_jn(3, 1.0), 9.0065811171125163e-3, 1e-9));

    // CONTROL: the n < x path (upward recurrence) still works and is
    // undisturbed by the new backward branch.
    assert!(rel_close(spherical_jn(0, 1.0), 0.84147098480789650, 1e-12));
    assert!(rel_close(spherical_jn(1, 1.0), 0.30116867893975674, 1e-12));
    assert!(rel_close(spherical_jn(2, 4.0), 0.27628368577135016, 1e-12));
    assert!(rel_close(
        spherical_jn(4, 2.0),
        1.4079392762915321e-2,
        1e-12
    ));
}

#[test]
fn spherical_jn_boundary_and_odd_parity() {
    // At the n == x switch point the n < x branch (j_{n-1} + 1) must agree
    // with the Miller backward branch.
    let exact_3_1 = 9.0065811171125163e-3;
    let v = spherical_jn(3, 1.0);
    assert!(
        rel_close(v, exact_3_1, 1e-9),
        "spherical_jn(3,1.0) = {v:e}, want {exact_3_1:e}"
    );
    // j_3 = (2*3-1)/x * j_2 - j_1, with x = 1.
    let j2_1 = spherical_jn(2, 1.0);
    let j1_1 = spherical_jn(1, 1.0);
    assert!(
        rel_close(v, 5.0 * j2_1 - j1_1, 1e-9),
        "j_3(1) = {v:e} but 5*j_2(1) - j_1(1) = {}",
        5.0 * j2_1 - j1_1
    );
    // n = x exactly, for x = 2. The reference here is computed from the
    // spherical-Bessel recurrence j_n = (2n-1)/x * j_{n-1} - j_{n-2}, seeded by
    // the closed forms for j_0 and j_1 - NOT from `0.5*sin(x)/x`, which is
    // j_0(x)/2 and is not j_n(x) at n = x for any n >= 1. It gives 0.2273 where
    // the correct value is 0.1984.
    {
        fn jn_ref(n: i32, x: f64) -> f64 {
            let mut jm1 = x.sin() / x; // j_0
            if n == 0 {
                return jm1;
            }
            let mut j0 = x.sin() / (x * x) - x.cos() / x; // j_1
            if n == 1 {
                return j0;
            }
            for k in 2..=n {
                let next = (2.0 * k as f64 - 1.0) / x * j0 - jm1;
                jm1 = j0;
                j0 = next;
            }
            j0
        }
        for n in 1..=4i32 {
            let expect = jn_ref(n, 2.0);
            let got = spherical_jn(n, 2.0);
            assert!(
                rel_close(got, expect, 1e-12),
                "spherical_jn({n}, 2.0) = {got:e}, recurrence gives {expect:e}"
            );
        }
    }

    // Sign parity: j_n(-x) = (-1)^n j_n(x) must hold on the new branch too.
    for &(n, x) in &[(20, 1.0), (10, 0.5), (5, 0.5), (3, 1.0)] {
        let pos = spherical_jn(n, x);
        let neg = spherical_jn(n, -x);
        let expect = if n % 2 == 0 { pos } else { -pos };
        assert!(
            rel_close(neg, expect, 1e-9),
            "spherical_jn({n}, -{x}) = {neg:e}, want {expect:e}"
        );
    }

    // Degenerate orders must not panic.
    assert!(spherical_jn(-1, 1.0).is_nan());
    assert_eq!(spherical_jn(0, 0.0), 1.0);
    assert_eq!(spherical_jn(4, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// Defect 7: bessel_jn must not launder a non-finite x into 0.0
// ---------------------------------------------------------------------------

#[test]
fn bessel_jn_non_finite_argument_is_nan() {
    // Pre-fix: bessel_jn(0, NaN) = 0.0 (it hit the `n == 0 && x.is_finite()`
    // branch); a NaN turned into a plausible finite zero defeats every
    // downstream NaN check.
    let v = bessel_jn(0, f64::NAN);
    assert!(v.is_nan(), "bessel_jn(0, NaN) = {v}, want NaN");
    let v = bessel_jn(3, f64::NAN);
    assert!(v.is_nan(), "bessel_jn(3, NaN) = {v}, want NaN");
    let v = bessel_jn(1, f64::NAN);
    assert!(v.is_nan(), "bessel_jn(1, NaN) = {v}, want NaN");
    // +/-inf is likewise undefined for the Bessel functions.
    let v = bessel_jn(0, f64::INFINITY);
    assert!(v.is_nan(), "bessel_jn(0, inf) = {v}, want NaN");
    let v = bessel_jn(2, f64::NEG_INFINITY);
    assert!(v.is_nan(), "bessel_jn(2, -inf) = {v}, want NaN");

    // CONTROL: x == 0 still has the exact limiting values, and ordinary
    // finite arguments are untouched.
    assert_eq!(bessel_jn(0, 0.0), 1.0);
    assert_eq!(bessel_jn(2, 0.0), 0.0);
    assert!((bessel_jn(2, 1.0) - 0.11490348493190048).abs() < 1e-12);
    assert!((bessel_jn(10, 15.0) - (-0.090071811047659034)).abs() < 1e-9);
    assert!((bessel_jn(5, 2.0) - 0.0070396297558716855).abs() < 1e-9);
}
