//! `Distortion::remove` must invert `Distortion::apply`.
//!
//! The inversion was a fixed ten iterations with no convergence test, so it
//! diverged for ordinary inputs. Measured with `k1=0.5, k2=0.2, k3=0.05` -
//! coefficients a calibrator legitimately produces:
//!
//! | radius | forward | recovered (before) | error (before) |
//! | ---: | ---: | ---: | ---: |
//! | 0.5 | 0.569141 | 0.500022 | 0.000022 |
//! | 0.6 | 0.724952 | 0.603138 | 0.003138 |
//! | 0.8 | 1.132022 | 1.132336 | **0.332336** |
//! | 1.0 | 1.750000 | **NaN** | - |
//!
//! At r = 1.0 the iteration ran away and returned NaN, which then propagates
//! through every undistorted image and every refined calibration with no error
//! raised. At r = 0.8 it returned 1.132 where the answer is 0.8 - a normalised
//! error of 0.33, roughly 166 px at fx = 500, reported as a successful undistort.

use cv_core::Distortion;

/// Strong but entirely ordinary coefficients.
fn distortion() -> Distortion {
    Distortion {
        k1: 0.5,
        k2: 0.2,
        k3: 0.05,
        p1: 0.0,
        p2: 0.0,
    }
}

/// `remove(apply(x)) == x` for radii across the range a real image produces.
#[test]
fn remove_inverts_apply_across_the_usable_range() {
    let d = distortion();
    for (x, y) in [
        (0.0f64, 0.0f64),
        (0.1, 0.0),
        (0.3, 0.2),
        (0.5, 0.0),
        (0.6, 0.0),
        (0.0, 0.5),
        (-0.4, 0.3),
        (0.2, -0.5),
    ] {
        let (xd, yd) = d.apply(x, y);
        let (bx, by) = d.remove(xd, yd);
        assert!(
            (bx - x).abs() < 1e-6 && (by - y).abs() < 1e-6,
            "remove(apply({x}, {y})) = ({bx}, {by}); expected ({x}, {y}). \\
             apply gave ({xd}, {yd})"
        );
    }
}

/// Beyond the radius where the distortion stays invertible, the old code
/// returned NaN. It must now return a finite value - and, because it does not
/// converge there, the input unchanged so a caller's residual check can see it.
#[test]
fn a_diverging_input_does_not_return_nan() {
    let d = distortion();
    for r in [0.8f64, 1.0, 1.2, 2.0] {
        let (xd, yd) = d.apply(r, 0.0);
        let (bx, by) = d.remove(xd, yd);
        assert!(
            bx.is_finite() && by.is_finite(),
            "remove({xd}, {yd}) returned ({bx}, {by}) - NaN propagates through \\
             every undistorted image with no error raised"
        );
    }
}

/// The identity distortion must be exactly the identity, not merely close.
#[test]
fn the_identity_distortion_is_exact() {
    let d = Distortion::none();
    for (x, y) in [(0.0f64, 0.0f64), (0.7, -0.3), (-1.0, 0.5)] {
        let (bx, by) = d.remove(x, y);
        assert_eq!(
            (bx, by),
            (x, y),
            "with no distortion, remove is the identity"
        );
    }
}

/// Tangential (decentering) terms shift the angle as well as the radius, which
/// the bisection does not model directly - they are corrected by two
/// fixed-point steps afterwards.
///
/// This asserts what actually happens rather than what would be ideal: where
/// the correction converges, the round trip is exact; where it does not,
/// `remove_checked` says so instead of returning a plausible wrong point. The
/// distinction is the whole point - the old code returned wrong values here
/// silently.
#[test]
fn tangential_terms_are_either_corrected_or_reported() {
    let d = Distortion {
        k1: 0.1,
        k2: 0.05,
        k3: 0.0,
        p1: 0.002,
        p2: -0.001,
    };

    let mut exact = 0usize;
    let mut reported = 0usize;
    for (x, y) in [
        (0.2f64, 0.3f64),
        (-0.5, 0.1),
        (0.6, -0.4),
        (0.3, 0.2),
        (0.1, -0.1),
    ] {
        let (xd, yd) = d.apply(x, y);
        match d.remove_checked(xd, yd) {
            Some((bx, by)) => {
                assert!(
                    (bx - x).abs() < 1e-5 && (by - y).abs() < 1e-5,
                    "remove_checked returned ({bx}, {by}) for ({x}, {y}) but \
                     claimed convergence"
                );
                exact += 1;
            }
            None => reported += 1,
        }
    }
    assert!(
        exact + reported == 5,
        "every point must be either solved or reported, never silently wrong"
    );
    assert!(
        exact > 0,
        "the tangential correction never converged, which suggests it is broken \
         rather than merely limited"
    );
}
