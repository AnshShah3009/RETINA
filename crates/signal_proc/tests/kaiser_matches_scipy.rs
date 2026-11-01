//! `cv-signal`'s Kaiser window consumes `cv-math`'s `bessel_i0`, and it was
//! inheriting that function's error *as its own taper error*.
//!
//! The window is `I0(beta*sqrt(1-a^2)) / I0(beta)`. Both `I0` calls were the
//! Numerical Recipes polynomial fit, which is only good to 4.7e-07 relative
//! around `x = 100` — so `kaiser_window(5, 100.0)` came back with edges
//! `9.313144427956708e-43` where SciPy's `get_window(('kaiser', 100.0), 5)` is
//! `9.313140024446352e-43`. That is a 4.7e-07 relative error on a value near
//! the underflow cliff, which is exactly where a spectral estimator cares: the
//! taper *is* the leakage control, and a few hundred parts per billion in the
//! first lobe is the sidelobe floor.
//!
//! The fix belongs in `cv-math` (`bessel_i0` now evaluates the series and the
//! asymptotically-truncated expansion instead of a fit; see
//! `crates/math/tests/bessel_i0_accuracy.rs`). This file pins the contract from
//! the side that was consuming it, so the boundary cannot silently drift again.
//!
//! Reference arrays: `scipy.signal.windows.kaiser(n, beta, sym=True)`
//! (SciPy 1.17.1), stored at full f64 precision.

use cv_signal::signal::kaiser_window;

/// The window is a ratio of two `I0` values, each accurate to a few ulps, so
/// the ratio carries a couple of ulps more. `1e-13` is that round-off with
/// room to spare; the defect this file pins was **4.7e-07**, four and a half
/// orders of magnitude larger, so the bound discriminates cleanly.
const TOL: f64 = 1e-13;

/// Relative comparison that survives the subnormal end of the window.
fn close(name: &str, got: &[f64], want: &[f64], rel_tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = if *w == 0.0 {
            g.abs()
        } else {
            ((g - w) / w).abs()
        };
        assert!(
            err <= rel_tol,
            "{name}[{i}] = {g:e}, SciPy {w:e} (relative error {err:e} > {rel_tol:e})"
        );
    }
}

/// The case the defect showed up in: `beta = 100` puts the window's argument at
/// `I0(100)`, the worst point of the old fit.
#[test]
fn kaiser_beta_100_matches_scipy() {
    let want5 = [
        9.313_140_024_446_352e-43,
        1.6325887570331584e-6,
        1.0,
        1.6325887570331584e-6,
        9.313_140_024_446_352e-43,
    ];
    close("kaiser(5, 100)", &kaiser_window(5, 100.0), &want5, TOL);

    let want11 = [
        9.313_140_024_446_352e-43,
        5.489_236_493_376_977e-18,
        2.305_168_306_504_981e-9,
        2.4733108794131877e-4,
        1.3396472639814228e-1,
        1.0,
        1.3396472639814228e-1,
        2.4733108794131877e-4,
        2.305_168_306_504_981e-9,
        5.489_236_493_376_977e-18,
        9.313_140_024_446_352e-43,
    ];
    close("kaiser(11, 100)", &kaiser_window(11, 100.0), &want11, TOL);

    let want101 = [
        9.313_140_024_446_352e-43,
        3.6790351205568297e-35,
        1.020_111_111_389_243e-31,
        1.0,
        1.020_111_111_389_243e-31,
        3.6790351205568297e-35,
        9.313_140_024_446_352e-43,
    ];
    let got101: Vec<f64> = {
        let w = kaiser_window(101, 100.0);
        vec![w[0], w[1], w[2], w[50], w[98], w[99], w[100]]
    };
    close("kaiser(101, 100)", &got101, &want101, TOL);
}

/// `beta = 20` exercises the small-argument branch of `I0` at the centre and the
/// large one at the edges, so a regression in either shows here.
#[test]
fn kaiser_beta_20_matches_scipy() {
    let got = kaiser_window(16, 20.0);
    // scipy.signal.windows.kaiser(16, 20.0)
    let want = [
        2.295_774_629_389_451e-8,
        6.329_270_359_646_157e-5,
        2.016_197_558_174_979e-3,
        2.051_151_182_782_682e-2,
        1.0549770483746701e-1,
        3.2825222610412064e-1,
        6.745_290_111_726_915e-1,
        9.575_611_884_256_873e-1,
        9.575_611_884_256_873e-1,
        6.745_290_111_726_915e-1,
        3.2825222610412064e-1,
        1.0549770483746701e-1,
        2.051_151_182_782_682e-2,
        2.016_197_558_174_979e-3,
        6.329_270_359_646_157e-5,
        2.295_774_629_389_451e-8,
    ];
    close("kaiser(16, 20)", &got, &want, TOL);
}

/// The window's defining properties, both of which the old `I0` violated by the
/// fit's error rather than by round-off:
///
/// * the peak is exactly 1 (`I0(0)/I0(beta)` must be `1.0`, not `1 + eps`);
/// * the taper is symmetric about that peak, which needs `I0` to return the same
///   value for the mirrored arguments.
#[test]
fn kaiser_peak_is_exactly_one_and_the_taper_is_symmetric() {
    for &(n, beta) in &[
        (5usize, 8.0f64),
        (16, 20.0),
        (64, 14.0),
        (101, 100.0),
        (33, 3.0),
    ] {
        let w = kaiser_window(n, beta);
        if n % 2 == 1 {
            assert_eq!(w[n / 2], 1.0, "kaiser({n}, {beta}) peak");
        } else {
            assert_eq!(w[n / 2], w[n / 2 - 1], "kaiser({n}, {beta}) twin peaks");
        }
        for i in 0..n / 2 {
            // Mirrored indices reach the same `I0` argument up to a couple of
            // ulps of `2*i/(n-1) - 1`, so this is round-off rather than exactly
            // zero — measured worst 2.8e-14 over these five windows, against
            // the 1.3e-13 the old `I0` fit produced on its own.
            let a = w[i];
            let b = w[n - 1 - i];
            let err = if b == 0.0 {
                a.abs()
            } else {
                ((a - b) / b).abs()
            };
            assert!(
                err < 1e-13,
                "kaiser({n}, {beta}) asymmetric at {i} vs {}: {a:e} against {b:e}",
                n - 1 - i
            );
        }
        // Monotone away from the centre: a ripple means `I0` is not monotone.
        let half = if n % 2 == 1 { n / 2 + 1 } else { n / 2 };
        for i in 1..half {
            assert!(
                w[i] >= w[i - 1],
                "kaiser({n}, {beta}) falls towards the centre at {i}"
            );
        }
    }
}

/// `beta = 0` is the rectangular window, and it must stay *exactly* 1: the
/// numerator and denominator are the same `I0(0)`, so any branch asymmetry in
/// `bessel_i0` would show as a value other than 1.
#[test]
fn kaiser_beta_0_is_exactly_rectangular() {
    for n in [2usize, 3, 17, 64] {
        for v in kaiser_window(n, 0.0) {
            assert_eq!(v, 1.0, "kaiser({n}, 0) is not exactly rectangular");
        }
    }
}
