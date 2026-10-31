//! Regressions for `filtfilt`'s start-up transient.
//!
//! A zero-phase filter built from a *normalised* lowpass must leave a constant
//! alone. It did not: without steady-state initial conditions the forward
//! pass's transient is mirrored by the reverse pass and lands inside the
//! trimmed (unpadded) region. Measured on 300 samples of the constant 3.25 with
//! `butter(4, 50, 1000)`: max deviation 3.613e-1 at index 297 (3.6113 instead
//! of 3.25), with 6.5e-2 near the start; for a 10-sample constant of 3.25 the
//! error reached 3.249 (the signal came out as 0.00077 at the end).

use cv_signal::signal::*;

const FS: f64 = 1000.0;

#[test]
fn filtfilt_preserves_a_constant() {
    for (order, cutoff) in [(4usize, 50.0f64), (4, 200.0), (2, 100.0)] {
        let (b, a) = butter(order, cutoff, FS);
        for n in [10usize, 40, 300] {
            let x = vec![3.25f64; n];
            let y = filtfilt(&b, &a, &x);
            assert_eq!(y.len(), n);
            let worst = y.iter().map(|v| (v - 3.25).abs()).fold(0.0f64, f64::max);
            assert!(
                worst < 1e-9,
                "butter({order}, {cutoff}, {FS}) on {n} samples of 3.25: max deviation \
                 {worst:.6e} (y[0]={}, y[n-1]={})",
                y[0],
                y[n - 1]
            );
        }
    }
}

#[test]
fn filtfilt_matches_a_reference_implementation() {
    // Reference: scipy 1.17.1 `signal.filtfilt` with
    //   b, a = signal.butter(2, 100.0, fs=1000.0)
    //   x = [0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-3,-2,-1,0,1,2,3]
    // `butter` itself is pinned here as well, so the comparison covers the whole
    // design-then-filter path.
    let b_ref = [0.0674552738890719, 0.1349105477781438, 0.0674552738890719];
    let a_ref = [1.0, -1.1429805025399011, 0.41280159809618877];
    let x = [
        0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -3.0, -2.0, -1.0, 0.0,
        1.0, 2.0, 3.0,
    ];
    let y_ref = [
        0.000787972788,
        1.109931304457,
        2.112596727796,
        2.885751750117,
        3.306744556041,
        3.293104272668,
        2.845792989907,
        2.051526579039,
        1.043789976619,
        -0.034195074915,
        -1.040735161551,
        -1.834406249287,
        -2.285371315289,
        -2.312195920274,
        -1.919722800279,
        -1.193913924037,
        -0.253260768876,
        0.798999242268,
        1.894453092419,
        2.996328179008,
    ];

    let (b, a) = butter(2, 100.0, FS);
    for (i, (got, want)) in b.iter().zip(b_ref.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "b[{i}]: {got} vs scipy {want}");
    }
    for (i, (got, want)) in a.iter().zip(a_ref.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "a[{i}]: {got} vs scipy {want}");
    }

    let y = filtfilt(&b, &a, &x);
    for (i, (got, want)) in y.iter().zip(y_ref.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "filtfilt[{i}]: {got} vs scipy {want} (all: {y:?})"
        );
    }
}

#[test]
fn filtfilt_still_filters_and_stays_zero_phase() {
    // Control: the fix must not turn filtfilt into a pass-through.
    let n = 600;
    let f_keep = 10.0;
    let f_kill = 400.0;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            (2.0 * std::f64::consts::PI * f_keep * t).sin()
                + (2.0 * std::f64::consts::PI * f_kill * t).sin()
        })
        .collect();

    let (b, a) = butter(4, 100.0, FS);
    let y = filtfilt(&b, &a, &x);
    assert_eq!(y.len(), n);

    // The 10 Hz component survives with unit amplitude...
    // (600 samples at 1000 Hz hold six full 10 Hz cycles.)
    let peaks = find_peaks(&y, Some(0.5), None);
    assert!(peaks.len() >= 5, "low frequency lost: {peaks:?}");
    let amp = peaks.iter().map(|&i| y[i].abs()).fold(0.0f64, f64::max);
    assert!(
        (amp - 1.0).abs() < 0.05,
        "10 Hz component amplitude {amp} (interior), expected ~1.0"
    );

    // ...and the 400 Hz component is gone: the filtered signal must track the
    // pure 10 Hz sine, i.e. the high-frequency ripple is small.
    let mut worst_ripple = 0.0f64;
    for i in 50..n - 50 {
        let expected = (2.0 * std::f64::consts::PI * f_keep * i as f64 / FS).sin();
        worst_ripple = worst_ripple.max((y[i] - expected).abs());
    }
    assert!(
        worst_ripple < 0.05,
        "400 Hz not attenuated: max deviation from the 10 Hz sine {worst_ripple}"
    );
}

#[test]
fn decimate_preserves_a_constant() {
    // `decimate` is filtfilt + a strided copy, so it inherited the same edge
    // artifacts.
    let x = vec![5.0f64; 200];
    for factor in [2usize, 4] {
        let y = decimate(&x, factor);
        assert_eq!(y.len(), 200 / factor);
        let worst = y.iter().map(|v| (v - 5.0).abs()).fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "decimate(x, {factor}) deviation {worst:.6e}");
    }
}
