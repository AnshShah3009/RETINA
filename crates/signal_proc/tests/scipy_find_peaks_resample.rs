//! `find_peaks` and `resample` against reference values from SciPy 1.x
//! (`scipy.signal.find_peaks`, `scipy.signal.resample`).

use cv_signal::signal::{find_peaks, resample};

/// Plateau peaks: SciPy reports one peak at the midpoint of a flat top
/// (rounded down). The strict-inequality scan found none of them.
#[test]
fn find_peaks_reports_plateaus_like_scipy() {
    let cases: &[(&[f64], &[usize])] = &[
        (&[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], &[1, 4]),
        (&[0.0, 2.0, 2.0, 2.0, 1.0, 3.0, 3.0, 0.0, 1.0, 1.0], &[2, 5]),
        (&[1.0, 1.0, 1.0], &[]),
        (&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], &[2]),
    ];
    for (data, want) in cases {
        assert_eq!(find_peaks(data, None, None), *want, "{data:?}");
    }
    assert_eq!(
        find_peaks(&[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], Some(1.5), None),
        [4]
    );
    assert_eq!(
        find_peaks(&[0.0, 5.0, 0.0, 4.0, 0.0, 6.0, 0.0], None, Some(3)),
        [1, 5]
    );
}

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-8, "got {got:?}, SciPy {want:?}");
    }
}

/// Downsampling to an even length folds `X[num/2]` and `X[-num/2]` into the
/// output's Nyquist bin; only one was copied.
#[test]
fn resample_matches_scipy() {
    let x = [1.0, 3.0, -2.0, 5.0, 0.5, -1.0, 2.0, 4.0];
    close(
        &resample(&x, 4),
        &[2.59283009, 1.07138348, 1.28216991, 1.30361652],
    );
    close(&resample(&x, 3), &[2.21783009, 1.1342751, 1.33539482]);
    close(
        &resample(&x, 6),
        &[
            2.1875,
            0.49189081,
            2.05698947,
            1.6875,
            -0.77297973,
            3.72409944,
        ],
    );
}
