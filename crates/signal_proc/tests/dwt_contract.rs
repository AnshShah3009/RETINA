//! Pins the `wavedec` / `waverec` coefficient order (which the doc comments had
//! backwards) and documents the one remaining length limitation.

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

/// **Known limitation, reported not fixed.** `dwt` emits `n / 2` coefficients
/// (floor), so an odd-length input loses its last sample and `idwt` can only
/// ever rebuild an even-length signal — it has no way to learn the original
/// length from its arguments. `waverec` therefore shortens odd-length signals,
/// and `idwt` silently ignores detail coefficients beyond `approx.len()`.
/// Mirrors `pywt`'s 'periodization' mode only for even lengths. Fixing it needs
/// a signature that carries the reconstruction length.
#[test]
fn odd_length_signals_are_truncated_by_the_wavelet_filterbank() {
    let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
    let coeffs = wavedec(&x, Wavelet::Haar, 2);
    let back = waverec(&coeffs, Wavelet::Haar);
    assert_eq!(back.len(), 12, "15 samples in, {} out", back.len());
    assert!(back.len() < x.len());
    // The even-length control keeps every sample.
    let even: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let back_even = waverec(&wavedec(&even, Wavelet::Haar, 2), Wavelet::Haar);
    assert_eq!(back_even.len(), 16);
}
