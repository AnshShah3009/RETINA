//! `bessel_i0` used the Numerical Recipes polynomial fits, whose accuracy was
//! the fit's and not round-off's: **4.73e-07 relative at `x = 100`**, and no
//! better than 3.1e-08 anywhere the parity harness looked (its grid stops at
//! `x = 9`). The documented bound for this function in
//! `parity/parity_special.py` is `1e-7` relative, so the old code was five
//! times outside its own contract past `x ~ 50`.
//!
//! It mattered to a caller. `cv-signal`'s Kaiser window is
//! `I0(beta*sqrt(1-a^2)) / I0(beta)`, so the error in `I0` is the error in the
//! window's taper: `kaiser_window(5, 100.0)` returned `9.313144427956708e-43`
//! at its edges against SciPy's `9.313140024446352e-43`.
//!
//! Reference values are `mpmath.besseli(0, x)` at 50 digits (and
//! `scipy.special.i0` for the exact f64 roundings), not SciPy's approximation.

use cv_math::special::bessel_i0;
use std::f64::consts::PI;

/// Relative error of `got` against `want`, safe when either is zero.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// The values the old polynomial fit got wrong by up to 4.7e-07. Every one of
/// these is an ordinary argument: `x = 100` is what a Kaiser window with
/// `beta = 100` asks for, `x = 50` and `x = 400` bracket it, and `x = 0.74` is
/// the worst point of the small-argument fit.
///
/// Reference: `mpmath.besseli(0, x)` at 50 digits. The `1e-15` tolerance is
/// round-off on these magnitudes (a few ulps), not a fitted number: the
/// pre-fix code misses it by eight orders of magnitude at `x = 100`.
#[test]
fn bessel_i0_matches_the_reference_to_round_off() {
    let cases: [(f64, f64); 12] = [
        // x, I0(x) to 17 significant figures
        (0.5, 1.0634833707413235),
        (0.74, 1.1416572858355174),
        (1.0, 1.2660658777520083),
        (3.7499999, 9.1189450828430802),
        (3.75, 9.1189458608445667),
        (3.7500001, 9.1189466388461236),
        (9.0, 1093.5883545113747),
        (10.0, 2815.7166284662545),
        (20.0, 4.3558282559553533e7),
        (50.0, 2.9325537838493363e20),
        (100.0, 1.0737517071310738e42),
        (400.0, 1.0418584503521463e172),
    ];
    for (x, want) in cases {
        let got = bessel_i0(x);
        assert!(
            rel(got, want) < 1e-15,
            "I0({x}) = {got:e}, want {want:e} (relative error {:e}); \
             the polynomial fit was 4.7e-07 off at x = 100",
            rel(got, want)
        );
    }
}

/// `I_0` is even, so the two branches must agree at the same `|x|` — and the
/// old large-argument branch was the less accurate of the two, which made
/// `I0(x)` and `I0(-x)` disagree by the fit's error rather than by zero.
#[test]
fn bessel_i0_is_even_to_round_off() {
    for x in [0.25f64, 1.0, 3.75, 9.0, 18.0, 20.0, 100.0, 400.0, 709.0] {
        let pos = bessel_i0(x);
        let neg = bessel_i0(-x);
        assert_eq!(pos, neg, "I0({x}) = {pos:e} but I0(-{x}) = {neg:e}");
    }
}

/// The branch crossover is at `|x| = 18`, and a step there would show up as a
/// jump in the value. Each side is checked against the exact value **at its own
/// argument**, so this measures the two representations against the truth
/// rather than against each other.
#[test]
fn bessel_i0_is_continuous_across_the_branch_crossover() {
    let x = 18.0f64;
    // mpmath.besseli(0, ·) at 50 digits, at the exact doubles used below.
    let below_x = x * (1.0 - 1e-13);
    let above_x = x * (1.0 + 1e-13);
    let cases = [
        (below_x, 6.2184124207701179e6),
        (x, 6.2184124207810029e6),
        (above_x, 6.2184124207918665e6),
    ];
    for (arg, want) in cases {
        let got = bessel_i0(arg);
        assert!(
            rel(got, want) < 1e-15,
            "I0({arg}) = {got:e}, want {want:e} (relative error {:e})",
            rel(got, want)
        );
    }
    // A step at the branch would be far larger than either side's round-off.
    assert!(
        rel(bessel_i0(below_x), bessel_i0(above_x)) < 1e-11,
        "step at the crossover: {:e} below against {:e} above",
        bessel_i0(below_x),
        bessel_i0(above_x)
    );
}

/// `I_0` is monotone increasing on `[0, inf)`, and the Kaiser window relies on
/// it: the taper is `I0(beta*sqrt(1-a^2))`, so a non-monotone `I0` makes the
/// window ripple instead of falling away from its centre.
#[test]
fn bessel_i0_is_monotone_increasing() {
    let mut prev = bessel_i0(0.0);
    let mut x = 0.0f64;
    while x < 720.0 {
        x += if x < 10.0 { 0.001 } else { 0.1 };
        let v = bessel_i0(x);
        assert!(v >= prev, "I0 fell between {x} and the previous sample");
        prev = v;
    }
}

/// The domain edges, which the even/odd handling must not disturb. `I_0` is
/// `+inf` past `x ~ 709.78`, `NaN` at `+-inf` (`scipy.special.i0(inf)` is
/// NaN) and `NaN` in, `NaN` out.
///
/// The `f64::MAX` row is a regression of its own: the first version of this
/// fix computed `sqrt(2*pi*x)`, which overflows to `+inf` there and returned
/// `NaN` where SciPy returns `+inf`.
#[test]
fn bessel_i0_domain_edges() {
    assert_eq!(bessel_i0(0.0), 1.0, "I0(0) = 1 exactly");
    assert_eq!(bessel_i0(-0.0), 1.0, "I0(-0.0) = 1 exactly");
    assert!(bessel_i0(f64::NAN).is_nan());
    assert!(
        bessel_i0(f64::INFINITY).is_nan(),
        "scipy.special.i0(inf) is NaN"
    );
    assert!(bessel_i0(f64::NEG_INFINITY).is_nan());
    assert_eq!(bessel_i0(f64::MAX), f64::INFINITY);
    assert_eq!(bessel_i0(-f64::MAX), f64::INFINITY);

    // The overflow boundary itself: finite at 709.7, infinite at 710, matching
    // where SciPy flips too.
    assert!(bessel_i0(709.7).is_finite(), "I0(709.7) must stay finite");
    assert_eq!(bessel_i0(710.0), f64::INFINITY);

    // Nothing may come back negative anywhere on the domain, including the far
    // tail where `exp` dominates `sqrt`.
    for x in [1e-300f64, 1e-8, 0.5, 3.0, 17.0, 100.0, 700.0] {
        assert!(bessel_i0(x) > 0.0, "I0({x}) = {}", bessel_i0(x));
    }
}

/// A structural check rather than a value check: `I0(x)*sqrt(x)/e^x` tends to
/// `1/sqrt(2*pi)` with a **known** leading correction. The asymptotic series is
/// `1/sqrt(2*pi) * (1 + 1/(8x) + 9/(128x^2) + ...)`, so
/// `(I0(x)*sqrt(x)/e^x - 1/sqrt(2*pi)) * 8x -> 1`. Both the approach from above
/// and that `8x` scaling are what a wrong coefficient breaks first, and they are
/// independent of which representation happens to be in use.
#[test]
fn bessel_i0_asymptotic_prefactor_and_its_first_correction() {
    let limit = 1.0 / (2.0 * PI).sqrt();
    let mut prev_gap = f64::INFINITY;
    for x in [50.0f64, 100.0, 200.0, 400.0, 700.0] {
        let scaled = (bessel_i0(x) * x.sqrt()) / x.exp();
        let gap = scaled - limit;
        assert!(
            gap > 0.0,
            "I0({x})*sqrt(x)/e^{x} = {scaled:e} is not just above {limit:e}"
        );
        // The correction is +1/(8x) relative, so the gap in absolute terms
        // shrinks like 1/x and `gap * 8x / limit` tends to 1.
        let normalised = gap * 8.0 * x / limit;
        assert!(
            (normalised - 1.0).abs() < 0.02,
            "at x = {x}: (I0*sqrt(x)/e^x - 1/sqrt(2pi))*8x/limit = {normalised:e}, \
             want 1 (the series' own 9/(128x^2) term is the 2% here)"
        );
        assert!(gap < prev_gap, "the asymptotic gap grew at x = {x}");
        prev_gap = gap;
    }
}
