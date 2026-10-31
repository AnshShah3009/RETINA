//! `DistortionF32::remove` must invert `DistortionF32::apply`.
//!
//! This is the `f32` twin of `distortion_roundtrip.rs`. The `f32` inversion was
//! a fixed ten iterations of `xd += x - apply(xd)` with no convergence test.
//! That iteration's contraction factor is `1 - A'(r)`, which exceeds 1 once `r`
//! is past about 0.65, so it diverged rather than converging slowly. Measured
//! with `k1 = 0.5, k2 = 0.2, k3 = 0.05`:
//!
//! | radius | forward | recovered (before) | error (before) |
//! | ---: | ---: | ---: | ---: |
//! | 0.5 | 0.569141 | 0.500022 | 0.000022 |
//! | 0.6 | 0.724952 | 0.603138 | 0.003138 |
//! | 0.8 | 1.132022 | 1.132336 | **0.332336** |
//! | 0.9 | 1.406513 | **NaN** | - |
//! | 1.0 | 1.750000 | **NaN** | - |
//!
//! Why the suite missed it: the pre-existing round-trip case used `(0.3, 0.4)`,
//! i.e. r = 0.5, the one radius where ten iterations still land inside 2e-5.
//!
//! The `f32` half of the problem is why it stayed hidden longer than the `f64`
//! half. In `f32` the runaway saturates to `+/-inf` within those ten steps and
//! `inf - finite` is NaN, so the result was not just inaccurate - it was
//! uninterpretable, and the NaN propagated through `PinholeModelF32::unproject`
//! and everything built on it with nothing raised.
//!
//! Tolerances here are `f32`-scaled, not copied from the `f64` file. `f32`
//! resolves ~1.2e-7 relative, so:
//!   - round-trip assertions use `2e-4`; the actual worst case measured by a
//!     dense sweep of the whole unit disc at a 0.01 grid, over five coefficient
//!     sets (strong radial, tangential-only, combined, ordinary, barrel), is
//!     **8.8e-6**. So `2e-4` is ~23x headroom over measured reality and still
//!     ~1600x tighter than the 0.332 error being fixed - the tests cannot pass
//!     vacuously, and they are not so tight that they would catch f32 rounding.
//!   - the CONTROL below pins the measurement against independently computed
//!     `f64` ground truth, so the tolerance cannot absorb a systematically wrong
//!     answer.

use cv_core::DistortionF32;

/// Strong but entirely ordinary coefficients - the same set used for the `f64`
/// twin so the before/after numbers are directly comparable.
fn radial_distortion() -> DistortionF32 {
    DistortionF32 {
        k1: 0.5,
        k2: 0.2,
        k3: 0.05,
        p1: 0.0,
        p2: 0.0,
    }
}

/// Radius used by the tests, as `f64`, for the CONTROL comparisons.
fn radius_of(x: f32, y: f32) -> f64 {
    (x as f64 * x as f64 + y as f64 * y as f64).sqrt()
}

/// `remove(apply(x)) == x` for radii across the range a real image produces.
///
/// The old code passed this at r = 0.5 only. The point of the sweep is the
/// radii at or past 0.65, where the fixed-point iteration turned divergent.
#[test]
fn remove_inverts_apply_across_the_usable_range() {
    let d = radial_distortion();
    let tol = 2e-4f32;

    // (x, y) pairs spanning r = 0 through r = 1.0, off-axis as well as on-axis:
    // an axis-aligned-only sweep would not exercise `cos_t`/`sin_t` recovery.
    let points = [
        (0.0f32, 0.0f32),
        (0.1, 0.0),
        (0.0, -0.3),
        (0.3, 0.4),   // r = 0.5, the radius the old suite stopped at
        (-0.6, 0.8),  // r = 1.0
        (0.8, 0.0),   // r = 0.8, past the divergence threshold
        (0.0, 0.9),   // r = 0.9, where the old code returned NaN
        (0.5, 0.7),   // r = 0.860, off-axis past the threshold
        (-0.4, -0.9), // r = 0.985, off-axis, negative quadrant
    ];

    for (x, y) in points {
        let (xd, yd) = d.apply(x, y);
        let (xr, yr) = d.remove(xd, yd);

        assert!(
            xr.is_finite() && yr.is_finite(),
            "remove(apply(({x}, {y}))) = ({xr}, {yr}); input was ({xd}, {yd}). \
             A non-finite result here is the divergence the bisection replaced."
        );
        assert!(
            (xr - x).abs() < tol && (yr - y).abs() < tol,
            "remove(apply(({x}, {y}))) = ({xr}, {yr}), expected ({x}, {y}); \
             err = ({}, {})",
            (xr - x).abs(),
            (yr - y).abs()
        );
    }
}

/// CONTROL: pins the measurement so the tolerance above cannot pass vacuously.
///
/// For each sample radius the expected `f32` input is taken from the `f64`
/// model - independent ground truth, computed with `Distortion::apply`, not
/// from `DistortionF32::apply`. If the forward map or the inverse were
/// systematically off, this fails; it cannot be satisfied by two consistent
/// wrong implementations of the same wrong model.
///
/// It also asserts the specific regression values from the defect report, so a
/// future change cannot quietly redefine what "correct" means here.
#[test]
fn control_ground_truth_scores_zero() {
    let d32 = radial_distortion();
    let d64 = cv_core::Distortion::new(0.5, 0.2, 0.0, 0.0, 0.05);

    // (true radius, f64 forward radius) taken from the f64 model, plus the
    // radius the buggy f32 iteration returned. The third column is the bug's
    // output, asserted to differ from the truth by a wide margin.
    let cases = [
        (0.5f32, 0.569141f64, 0.500022f64),
        (0.6, 0.724952, 0.603138),
        (0.8, 1.132022, 1.132336),
        (0.9, 1.406513, f64::NAN),
        (1.0, 1.750000, f64::NAN),
    ];

    for (r_true, r_fwd_64, r_buggy) in cases {
        // The f64 forward map must reproduce the tabulated value.
        let (fx, fy) = d64.apply(r_true as f64, 0.0);
        assert!(
            (fx.hypot(fy) - r_fwd_64).abs() < 1e-6,
            "CONTROL: f64 apply({r_true}) gave radius {}, table says {r_fwd_64}",
            fx.hypot(fy)
        );

        // The f32 forward map must agree with f64 to f32 precision.
        let (dx, dy) = d32.apply(r_true, 0.0);
        let r_fwd_32 = (dx as f64 * dx as f64 + dy as f64 * dy as f64).sqrt();
        assert!(
            (r_fwd_32 - r_fwd_64).abs() < 1e-6,
            "CONTROL: f32 forward radius {r_fwd_32} disagrees with f64 {r_fwd_64}"
        );

        // The fixed point must score ~0 against the true radius.
        let (ux, uy) = d32.remove(dx, dy);
        let r_recovered = (ux as f64 * ux as f64 + uy as f64 * uy as f64).sqrt();
        let err = (r_recovered - r_true as f64).abs();
        assert!(
            err < 2e-6,
            "CONTROL: remove(apply({r_true})) recovered radius {r_recovered}, \
             expected {r_true} (err {err}); table's pre-fix value was {r_buggy}"
        );

        // And the pre-fix value must genuinely have been wrong, so this test
        // cannot be satisfied by a tolerance loose enough to accept it.
        assert!(
            r_buggy.is_nan() || (r_buggy - r_true as f64).abs() >= 2e-6,
            "CONTROL: the tabulated pre-fix radius {r_buggy} for r={r_true} is not \
             actually a failure, so this test proves nothing"
        );
    }
}

/// A diverging input must return a FINITE value, never NaN.
///
/// r = 0.9 and r = 1.0 both produced `(NaN, NaN)` before. `NaN` is the
/// dangerous outcome because it compares false against every tolerance and so
/// escapes a purely `assert!((a-b).abs() < tol)` style check - and it then
/// poisons every downstream f32 unprojection.
#[test]
fn diverging_input_returns_finite_not_nan() {
    let d = radial_distortion();

    for &r in &[0.9f32, 1.0, 1.2, 2.0] {
        let (xd, yd) = d.apply(r, 0.0);
        let (ux, uy) = d.remove(xd, yd);
        assert!(
            ux.is_finite() && uy.is_finite(),
            "remove(({xd}, {yd})) for true r={r} returned ({ux}, {uy}); \
             NaN here is the original defect"
        );
        // Even where inversion may legitimately fail, `remove` must degrade to
        // its input rather than to NaN - that is the whole point of routing it
        // through `remove_checked`.
        assert!(
            ux.abs() < 10.0 && uy.abs() < 10.0,
            "remove(({xd}, {yd})) for true r={r} returned ({ux}, {uy}), \
             which is not a sane fallback"
        );
    }

    // Explicitly pin the two tabulated rows.
    let (dx, dy) = d.apply(0.9, 0.0);
    let (ux, uy) = d.remove(dx, dy);
    assert!(
        ux.is_finite() && uy.is_finite(),
        "r = 0.9 returned ({ux}, {uy})"
    );
    assert!(
        (ux as f64 - 0.9).abs() < 1e-5 && (uy as f64).abs() < 1e-5,
        "r = 0.9 recovered ({ux}, {uy}), expected (0.9, 0)"
    );
}

/// The inverse must actually reproduce the forward map to `f32` precision, at
/// radii where the old code was silently returning a wrong finite answer.
#[test]
fn inverse_residual_is_at_f32_precision() {
    let d = radial_distortion();
    for &r in &[0.5f32, 0.7, 0.8, 0.9, 1.0] {
        let (xd, yd) = d.apply(r, 0.0);
        let (ux, uy) = d.remove(xd, yd);
        let (rx, ry) = d.apply(ux, uy);
        let residual = ((rx - xd) as f64).hypot((ry - yd) as f64);
        assert!(
            residual < 1e-5,
            "at r={r}: residual {residual}; the inverse does not undo the forward map"
        );
    }
}

/// Zero distortion is the identity and must be exact, not merely close.
///
/// This is the case the `f64` twin short-circuits, and the short circuit is
/// what makes it exact: ~26 bisection steps would otherwise only reproduce the
/// input to within a few ULP. It is also the commonest case in production -
/// every rectified pipeline runs an all-zero model.
#[test]
fn identity_distortion_is_exact() {
    let d = DistortionF32::none();
    for (x, y) in [
        (0.0f32, 0.0f32),
        (0.5, 0.5),
        (-0.3, 0.7),
        (1.0, -1.0),
        (12.5, -33.25),
    ] {
        let (ax, ay) = d.apply(x, y);
        assert_eq!(
            (ax, ay),
            (x, y),
            "apply must be the identity for zero coefficients"
        );

        let (rx, ry) = d.remove(x, y);
        assert_eq!(
            (rx, ry),
            (x, y),
            "remove must be exactly the identity for zero coefficients, not merely close"
        );
        assert_eq!(
            d.remove_checked(x, y),
            Some((x, y)),
            "remove_checked must be exactly the identity for zero coefficients"
        );
    }
}

/// Coefficients with `p1`/`p2` AND enough radial power that `A'(r) > 1`, so the
/// pre-fix fixed-point iteration diverges on the radial direction.
///
/// This combination matters and is easy to get wrong when writing these tests.
/// `p1`/`p2` alone do NOT expose the regression: with no radial terms `A'(r)` is
/// exactly 1, the fixed-point contraction factor is 0, and the old ten-step loop
/// converges in one step - so a tangential-only case passes against the buggy
/// code and proves nothing about the fix. The regression only appears once the
/// radial term makes the iteration unstable. So this helper keeps the tangential
/// terms (to exercise that code path) while keeping the coefficients large
/// enough to reintroduce the divergence.
fn radial_and_tangential_distortion() -> DistortionF32 {
    DistortionF32::new(0.35, 0.08, 0.0009, -0.0006, 0.01)
}

/// Tangential (p1/p2) terms must be undone too.
///
/// With `k1 = k2 = k3 = 0` the radius is unchanged by `apply`, so a bisection
/// that only handled the radial part would look perfect here while getting the
/// angle wrong. The fixed-point tail is what carries these terms, so it needs
/// its own case. See `radial_and_tangential_distortion` for why this case is
/// written with radial terms present rather than tangential-only.
#[test]
fn tangential_terms_round_trip() {
    let d = radial_and_tangential_distortion();
    assert_ne!(
        (d.p1, d.p2),
        (0.0, 0.0),
        "this case is only meaningful with tangential terms present"
    );

    let tol = 2e-4f32;
    for (x, y) in [
        (0.3f32, 0.4f32),
        (0.5, 0.5),
        (-0.2, 0.6),
        (0.0, 0.7),
        (0.6, 0.0),
        (-0.8, -0.3),
    ] {
        let (xd, yd) = d.apply(x, y);
        let (xr, yr) = d.remove(xd, yd);
        assert!(
            xr.is_finite() && yr.is_finite(),
            "tangential remove(apply(({x}, {y}))) = ({xr}, {yr})"
        );
        assert!(
            (xr - x).abs() < tol && (yr - y).abs() < tol,
            "tangential remove(apply(({x}, {y}))) = ({xr}, {yr}), expected ({x}, {y}); \
             err = ({}, {})",
            (xr - x).abs(),
            (yr - y).abs()
        );

        // Confirm the tangential terms really are what is under test: with
        // p1 = p2 = 0 the radial part alone would pass, so this assertion is
        // what keeps the case honest about which code path it exercises.
        let radial_only = DistortionF32::new(d.k1, d.k2, 0.0, 0.0, d.k3);
        let (ax, ay) = radial_only.apply(x, y);
        let (bx, by) = d.apply(x, y);
        assert!(
            (ax - bx).abs() > 1e-7 || (ay - by).abs() > 1e-7,
            "CONTROL: the tangential coefficients changed nothing at ({x}, {y}); \
             this test is vacuous for that point"
        );
    }
}

/// The tangential terms must survive being added on top of a radial model that
/// alone is already invertible.
///
/// With tangential-only coefficients the distorted radius differs from
/// `A(r)`, because `p1`/`p2` change the length of the distorted point as well
/// as its angle. An inverse that bisects on the pure radial polynomial `A(r)`
/// therefore targets the wrong radius. This case pins that the tangential
/// contribution to the radius is handled exactly rather than ignored.
#[test]
fn tangential_terms_shift_the_radius_and_are_accounted_for() {
    let d = radial_and_tangential_distortion();
    let radial_only = DistortionF32::new(d.k1, d.k2, 0.0, 0.0, d.k3);

    for (x, y) in [(0.3f32, 0.4f32), (0.5, 0.5), (-0.45, 0.3), (0.55, -0.4)] {
        let (with_t, _) = d.apply(x, y);
        let (without_t, _) = radial_only.apply(x, y);

        let r_with = radius_of(with_t, 0.0);
        let r_without = radius_of(without_t, 0.0);
        assert!(
            (r_with - r_without).abs() > 1e-6,
            "CONTROL: at ({x}, {y}) the tangential terms did not move the distorted \
             radius ({r_with} vs {r_without}); this test is vacuous for that point"
        );

        // And despite that radius shift, the inverse must land back on the
        // original point - i.e. it accounted for the shift instead of bisecting
        // on the radial-only radius.
        let (xd, yd) = d.apply(x, y);
        let (xr, yr) = d.remove(xd, yd);
        assert!(
            (xr - x).abs() < 2e-4 && (yr - y).abs() < 2e-4,
            "at ({x}, {y}): remove gave ({xr}, {yr}); the tangential radius shift \
             was not accounted for"
        );
    }
}

/// Radial and tangential terms together, which is what a real calibration
/// produces and what the fixed-point tail has to converge against.
#[test]
fn radial_and_tangential_together_round_trip() {
    let d = radial_and_tangential_distortion();
    let tol = 2e-4f32;
    for (x, y) in [
        (0.0f32, 0.0f32),
        (0.25, 0.15),
        (0.3, 0.4),
        (-0.45, 0.3),
        (0.55, -0.4),
    ] {
        let (xd, yd) = d.apply(x, y);
        let (xr, yr) = d.remove(xd, yd);
        assert!(
            xr.is_finite() && yr.is_finite(),
            "remove(apply(({x}, {y}))) = ({xr}, {yr})"
        );
        assert!(
            (xr - x).abs() < tol && (yr - y).abs() < tol,
            "remove(apply(({x}, {y}))) = ({xr}, {yr}), expected ({x}, {y}); \
             err = ({}, {})",
            (xr - x).abs(),
            (yr - y).abs()
        );
    }
}

/// `remove_checked` must distinguish "no solution" from "zero correction".
///
/// It is the `f32` mirror of the `f64` contract, so it has to be pinned here.
/// `remove` deliberately collapses the two: callers that care must use the
/// checked form.
#[test]
fn remove_checked_reports_failure_and_identity_correctly() {
    let d = radial_distortion();

    // A solvable point yields Some, and Some is not confused with the identity.
    let (xd, yd) = d.apply(0.8, 0.0);
    let checked = d.remove_checked(xd, yd);
    assert!(
        checked.is_some(),
        "r = 0.8 is solvable and must not report failure"
    );
    let (ux, _uy) = checked.expect("just asserted Some");
    assert!(
        (ux - 0.8f32).abs() < 2e-4,
        "remove_checked gave {ux} for a true radius of 0.8"
    );

    // Non-finite input has no answer, and must be rejected rather than
    // propagated.
    assert_eq!(d.remove_checked(f32::NAN, 0.0), None);
    assert_eq!(d.remove_checked(0.0, f32::INFINITY), None);

    // `remove` degrades to its input instead of emitting NaN, which is the
    // behaviour the f32 path needs to stay non-poisoning.
    let (fx, fy) = d.remove(f32::NAN, 0.0);
    assert!(fx.is_nan() && fy == 0.0, "remove fell back to ({fx}, {fy})");

    // Zero coefficients: Some, and exactly the input - "no correction needed",
    // not "could not correct".
    assert_eq!(
        DistortionF32::none().remove_checked(0.3, 0.4),
        Some((0.3, 0.4))
    );

    // The origin is exactly fixed regardless of coefficients.
    assert_eq!(d.remove_checked(0.0, 0.0), Some((0.0, 0.0)));
}

/// The radius, not the coordinate, is what must be recovered.
///
/// Guards against an "inverse" that happens to land near the right answer for
/// axis-aligned points only by fixing up the radius and losing the angle.
#[test]
fn angle_is_preserved_not_just_the_radius() {
    let d = radial_distortion();
    for (x, y) in [
        (0.3f32, 0.4f32),
        (-0.3, 0.4),
        (0.4, -0.3),
        (0.5f32, 0.5f32),
        (-0.45, -0.45),
    ] {
        let (xd, yd) = d.apply(x, y);
        let (xr, yr) = d.remove(xd, yd);
        let t_in = (y as f64).atan2(x as f64);
        let t_out = (yr as f64).atan2(xr as f64);
        assert!(
            (t_in - t_out).abs() < 1e-5,
            "angle changed from {t_in} to {t_out} for ({x}, {y})"
        );
        assert!(
            (radius_of(xr, yr) - radius_of(x, y)).abs() < 2e-4,
            "radius {} != {} for ({x}, {y})",
            radius_of(xr, yr),
            radius_of(x, y)
        );
    }
}
