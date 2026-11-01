//! Pins the `wavedec` / `waverec` coefficient order (which the doc comments had
//! backwards) and the odd-length round trip.

use cv_signal::signal::*;

#[test]
fn wavedec_returns_a_coarse_to_fine_pyramid_in_waverec_order() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
    let coeffs = wavedec(&x, Wavelet::Haar, 3);
    assert_eq!(coeffs.len(), 4, "3 detail levels + 1 approximation");

    // Coarsest first: the approximation shrinks by 2 per level, the details
    // grow back up towards the signal length.
    let lengths: Vec<usize> = coeffs.iter().map(|c| c.len()).collect();
    assert_eq!(lengths, vec![4, 4, 8, 16], "coarse-to-fine pyramid");

    // And that order is what reconstructs the signal...
    let back = waverec(&coeffs, Wavelet::Haar);
    assert_eq!(back.len(), x.len());
    for (a, b) in x.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-10, "round trip: {a} vs {b}");
    }

    // ...and each entry is the detail *of its own level*: entry 1 is the
    // last level's detail, entry 3 is the first level's. (The doc comment used
    // to claim the opposite order, [detail_n, ..., detail_1, approx_n].)
    let (a1, d1) = dwt(&x, Wavelet::Haar);
    let (a2, d2) = dwt(&a1, Wavelet::Haar);
    let (a3, d3) = dwt(&a2, Wavelet::Haar);
    assert_eq!(coeffs[0], a3, "coeffs[0] is approx_level_3");
    assert_eq!(coeffs[1], d3, "coeffs[1] is detail_level_3");
    assert_eq!(coeffs[2], d2, "coeffs[2] is detail_level_2");
    assert_eq!(coeffs[3], d1, "coeffs[3] is detail_level_1");
}

#[test]
fn single_level_round_trip_is_exact_for_even_lengths() {
    for n in [4usize, 8, 16, 64] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() + 0.1).collect();
        for wave in [Wavelet::Haar, Wavelet::Db2, Wavelet::Db4] {
            let (a, d) = dwt(&x, wave);
            assert_eq!(a.len() + d.len(), n);
            let r = idwt(&a, &d, wave);
            assert_eq!(r.len(), n);
            let worst = x
                .iter()
                .zip(r.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-10, "n={n}: max error {worst}");
        }
    }
}

fn assert_round_trip(x: &[f64], wave: Wavelet, levels: usize) {
    let (a, d) = dwt(x, wave);
    let back = idwt(&a, &d, wave);
    assert_eq!(back.len(), x.len(), "{wave:?} single-level length");
    let worst = x
        .iter()
        .zip(back.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-8,
        "{wave:?} single-level max error {worst} on len {}",
        x.len()
    );

    let coeffs = wavedec(x, wave, levels);
    let rec = waverec(&coeffs, wave);
    assert_eq!(rec.len(), x.len(), "{wave:?} waverec length");
    let worst = x
        .iter()
        .zip(rec.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-8,
        "{wave:?} waverec max error {worst} on len {}",
        x.len()
    );
}

/// Odd-length signals must survive `dwt` / `idwt` and `wavedec` / `waverec`.
/// `[1, 2, 3, 4, 5]` used to come back as length 4, and a length-15 Haar
/// decomposition at 2 levels came back as length 12.
#[test]
fn odd_length_signals_round_trip() {
    let five = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for wave in [Wavelet::Haar, Wavelet::Db2, Wavelet::Db4] {
        assert_round_trip(&five, wave, 1);
    }

    let fifteen: Vec<f64> = (0..15).map(|i| i as f64).collect();
    assert_round_trip(&fifteen, Wavelet::Haar, 2);

    // Odd at every level: 9 → 5 → 3.
    let nine: Vec<f64> = (0..9).map(|i| (i as f64 * 0.37).sin()).collect();
    assert_round_trip(&nine, Wavelet::Db2, 3);

    // The even-length control keeps every sample.
    let even: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let back_even = waverec(&wavedec(&even, Wavelet::Haar, 2), Wavelet::Haar);
    assert_eq!(back_even.len(), 16);
    for (a, b) in even.iter().zip(back_even.iter()) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

/// Even-length Haar coefficients are the bits the current filter bank emits.
/// A change to the even path moves these bits. The second pair is 1 ulp away
/// from the closed form `7/√2` and `1/√2`, because the bank accumulates
/// `sample * coeff` rather than multiplying the integer sum once.
#[test]
fn even_length_haar_coefficients_stay_bit_identical() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let (a, d) = dwt(&x, Wavelet::Haar);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(a.len(), 2);
    assert_eq!(d.len(), 2);
    // The bank accumulates `sample * coeff`. That is bit-identical to `3*s`
    // and `s` for the first pair, and 1 ulp away from `7*s` and `s` for the
    // second (`4*s + 3*s`, `4*s - 3*s`). Pin the accumulated bits.
    let a0 = (2.0 * s) + (1.0 * s);
    let d0 = (2.0 * s) + (1.0 * -s);
    let a1 = (4.0 * s) + (3.0 * s);
    let d1 = (4.0 * s) + (3.0 * -s);
    assert_eq!(a[0].to_bits(), a0.to_bits());
    assert_eq!(a[1].to_bits(), a1.to_bits());
    assert_eq!(d[0].to_bits(), d0.to_bits());
    assert_eq!(d[1].to_bits(), d1.to_bits());
    assert_eq!(a[0].to_bits(), 4_611_959_207_554_411_738);
    assert_eq!(a[1].to_bits(), 4_617_258_938_640_847_956);
    assert_eq!(d[0].to_bits(), 4_604_544_271_217_802_189);
    assert_eq!(d[1].to_bits(), 4_604_544_271_217_802_188);

    let back = idwt(&a, &d, Wavelet::Haar);
    assert_eq!(back.len(), 4);
    for (got, want) in back.iter().zip(x.iter()) {
        assert!((got - want).abs() < 1e-12);
    }
}
