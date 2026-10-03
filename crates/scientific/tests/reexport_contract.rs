#![forbid(unsafe_code)]
//! `cv-scientific` is a re-export layer over `cv-math`, `cv-geometry2d`,
//! `cv-signal` and `cv-pointcloud`. `lib.rs` is nothing but `pub use`, so the
//! only thing this crate can get wrong on its own is *where* its items come
//! from: a module pointing at the wrong sub-crate, or a sub-crate whose entry
//! point it does not actually carry.
//!
//! These tests reach every documented module through the `cv_scientific` path
//! and check the answer against an oracle computed inside the test (a brute
//! force search, an exact area, a closed-form solution) rather than against a
//! constant copied from the implementation. `functional_tests.rs` covers the
//! geometry and point-cloud paths; this file covers the numerical ones.
use cv_core::Rect;

/// A wrong answer here means the FFT is not the DFT the crate documents: zero
/// padding to a power of two (the spectrum would change length), a dropped
/// scale factor, or an `ifft` that does not invert `fft`.
#[test]
fn reexported_fft_is_the_dft_at_any_length() {
    use cv_scientific::fft::{fft, ifft};

    // Control: power-of-two length, the case every FFT gets right.
    let power_of_two = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let roundtrip = ifft(&fft(&power_of_two));
    assert_eq!(roundtrip.len(), power_of_two.len());
    for (got, want) in roundtrip.iter().zip(power_of_two) {
        assert!(
            (got.re - want).abs() < 1e-12 && got.im.abs() < 1e-12,
            "ifft(fft(x)) must return x, got {} + {}i for {}",
            got.re,
            got.im,
            want
        );
    }

    // The case that separates a real FFT from a zero-padding one: 7 is prime.
    let prime = [1.0, -2.0, 3.0, 0.5, 0.0, -1.0, 4.0];
    let spectrum = fft(&prime);
    assert_eq!(
        spectrum.len(),
        prime.len(),
        "a length-7 input must give a length-7 spectrum"
    );
    // Oracle: the naive DFT, O(n^2), for these seven samples.
    let n = prime.len();
    for k in 0..n {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (j, &x) in prime.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        let got = spectrum[k];
        assert!(
            (got.re - re).abs() < 1e-9 && (got.im - im).abs() < 1e-9,
            "bin {} of a length-7 DFT: got {} + {}i, naive DFT says {} + {}i",
            k,
            got.re,
            got.im,
            re,
            im
        );
    }

    // A delta must transform to a flat spectrum of ones (checks the 1/N scale
    // is applied by `ifft` and not by `fft`).
    let mut delta = [0.0; 12];
    delta[0] = 1.0;
    for (k, c) in fft(&delta).iter().enumerate() {
        assert!(
            (c.re - 1.0).abs() < 1e-12 && c.im.abs() < 1e-12,
            "bin {}",
            k
        );
    }
}

/// A KD-tree is only worth its complexity if it returns the same neighbours as
/// a brute-force scan. A wrong answer here would silently corrupt every
/// downstream consumer (this tree is what `cv_scientific::spatial` exists for).
#[test]
fn reexported_kdtree_matches_brute_force() {
    use cv_scientific::spatial::KDTree;

    // Deterministic, non-degenerate cloud in 3D.
    let points: Vec<Vec<f64>> = (0..257)
        .map(|i| {
            let f = i as f64;
            vec![
                (f * 12.9898).sin() * 97.0,
                (f * 78.233).cos() * 43.0,
                (f * 37.719).sin() * 61.0,
            ]
        })
        .collect();
    let tree = KDTree::new(&points).expect("a non-empty cloud must build");

    let queries: Vec<Vec<f64>> = (0..23)
        .map(|i| {
            let f = i as f64 * 3.7;
            vec![f.sin() * 90.0, f.cos() * 40.0, (f * 1.7).sin() * 60.0]
        })
        .collect();

    for q in &queries {
        let got = tree.query(q, 5);
        let mut want: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d2: f64 = p.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, d2.sqrt())
            })
            .collect();
        want.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("no NaN coordinates"));

        assert_eq!(got.len(), 5, "k = 5 neighbours were requested");
        for (rank, (got, want)) in got.iter().zip(&want[..5]).enumerate() {
            assert_eq!(
                got.0, want.0,
                "neighbour {} of query {:?}: tree chose point {} (d = {}), brute force says {} (d = {})",
                rank, q, got.0, got.1, want.0, want.1
            );
            assert!((got.1 - want.1).abs() < 1e-9);
        }
    }

    // Control: a query exactly on a point returns that point at distance zero.
    let first = tree.query(&points[0], 1);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0, 0);
    assert!(first[0].1.abs() < 1e-12);
}

/// Delaunay triangulation of a square's four corners is two triangles covering
/// exactly the square: an index or orientation slip shows up as the wrong area
/// or a missing face.
#[test]
fn reexported_delaunay_covers_the_hull() {
    use cv_scientific::geometry2d::{delaunay_triangulation, Point2D};

    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(10.0, 0.0),
        Point2D::new(10.0, 10.0),
        Point2D::new(0.0, 10.0),
    ];
    let triangles = delaunay_triangulation(&points);
    assert_eq!(
        triangles.len(),
        2,
        "four convex points triangulate into two faces"
    );

    let area: f64 = triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (&points[t[0]], &points[t[1]], &points[t[2]]);
            ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs() / 2.0
        })
        .sum();
    assert!(
        (area - 100.0).abs() < 1e-9,
        "the two triangles must tile the 10x10 square, covered area = {}",
        area
    );
    for t in &triangles {
        // Three *distinct* indices, each naming a real point. The check that
        // used to be here was
        // `(t[0] as f64 - 10.0 * 0.5).abs() < 100.0 && t[0] != t[1] && t[1] != t[2]`:
        // the first clause is `|i - 5| < 100`, true for every index of this
        // four-point set, and the second never compares `t[0]` with `t[2]`. So a
        // degenerate triangle `[0, 1, 0]`, or an index past the end of `points`,
        // passed an assertion whose message claims it rejects exactly those.
        assert!(
            t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
            "triangle vertices must be three distinct point indices, got {:?}",
            t
        );
        for &index in t {
            assert!(
                index < points.len(),
                "triangle index {} is out of range for {} points",
                index,
                points.len()
            );
        }
    }
}

/// Linear and cubic interpolation must reproduce their knots exactly — an
/// off-by-one in the spline coefficient indexing breaks exactly this.
#[test]
fn reexported_interpolators_reproduce_their_knots() {
    use cv_scientific::interpolate::{interp1d_akima, interp1d_cubic, interp1d_linear};

    let x = [0.0, 1.0, 2.0, 3.5, 5.0];
    let y = [0.0, 1.0, 4.0, -2.0, 7.0];

    // Control: linear interpolation at the knots is the identity.
    assert_eq!(interp1d_linear(&x, &y, &x).unwrap(), y.to_vec());
    // ...and halfway between two knots it is their mean.
    let mid = interp1d_linear(&x, &y, &[0.5, 2.75]).unwrap();
    assert!((mid[0] - 0.5).abs() < 1e-12);
    assert!((mid[1] - 1.0).abs() < 1e-12);

    for (name, got) in [
        ("cubic", interp1d_cubic(&x, &y, &x).unwrap()),
        ("akima", interp1d_akima(&x, &y, &x).unwrap()),
    ] {
        assert_eq!(got.len(), y.len());
        for (k, (got, want)) in got.iter().zip(y).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "{} spline at knot {} returned {}, knot value is {}",
                name,
                k,
                got,
                want
            );
        }
    }
}

/// A conjugate-gradient solve of a known SPD system must return the known
/// solution; a fabricated answer (zeros, or the right-hand side) fails here.
#[test]
fn reexported_sparse_solver_solves_a_known_system() {
    use cv_scientific::sparse::{cg_solve, CsrMatrix};
    use nalgebra::DVector;

    // A = [[4, 1], [1, 3]], b = [1, 2]  =>  x = [1/11, 7/11].
    let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let x = cg_solve(&a, &b, 100, 1e-12).expect("a symmetric positive definite system converges");

    assert!(
        (x[0] - 1.0 / 11.0).abs() < 1e-10 && (x[1] - 7.0 / 11.0).abs() < 1e-10,
        "expected [1/11, 7/11], got [{}, {}]",
        x[0],
        x[1]
    );

    // Control: the residual A x - b must vanish, which a transposed or
    // otherwise wrong matrix would not satisfy.
    let r0 = 4.0 * x[0] + x[1] - 1.0;
    let r1 = x[0] + 3.0 * x[1] - 2.0;
    assert!(
        r0.abs() < 1e-10 && r1.abs() < 1e-10,
        "residual [{}, {}]",
        r0,
        r1
    );
}

/// Detection IoU is a ratio of areas: the three cases a caller checks are
/// identical boxes (1), disjoint boxes (0) and a nested pair (the area ratio).
/// The existing test only covers a partial overlap.
#[test]
fn reexported_iou_covers_the_degenerate_cases() {
    use cv_scientific::geometry::vectorized_iou;

    let one = |r: Rect| vec![r];
    let iou = |a: Rect, b: Rect| vectorized_iou(&one(a), &one(b))[(0, 0)];

    let a = Rect::new(0.0, 0.0, 10.0, 10.0);
    assert!(
        (iou(a, a) - 1.0).abs() < 1e-6,
        "a box against itself is IoU 1, got {}",
        iou(a, a)
    );
    assert_eq!(
        iou(a, Rect::new(100.0, 100.0, 10.0, 10.0)),
        0.0,
        "disjoint boxes share no area"
    );
    assert_eq!(
        iou(a, Rect::new(10.0, 0.0, 10.0, 10.0)),
        0.0,
        "boxes that only touch on an edge share no area"
    );
    // Control for the ratio itself: a 5x5 box inside a 10x10 box.
    let nested = iou(a, Rect::new(0.0, 0.0, 5.0, 5.0));
    assert!((nested - 0.25).abs() < 1e-6, "nested IoU = {}", nested);
}

/// `filtfilt` is zero-phase, so a constant signal stays constant, a tone below
/// the cutoff passes and a tone above it is attenuated. A wrong filter
/// normalisation or a single forward pass fails at least one of the three.
#[test]
fn reexported_filtfilt_preserves_a_constant() {
    use cv_scientific::signal::{butter, filtfilt};

    let (b, a) = butter(2, 100.0, 1000.0);
    let x = vec![3.0f64; 400];
    let y = filtfilt(&b, &a, &x);
    assert_eq!(y.len(), x.len());

    for (i, v) in y.iter().enumerate().skip(50).take(300) {
        assert!(
            (v - 3.0).abs() < 1e-6,
            "sample {} of a filtered constant is {}, expected 3.0",
            i,
            v
        );
    }

    // Amplitude of a tone after filtering, measured over the middle of the
    // signal: `filtfilt`'s reflect padding leaves a transient at both ends
    // (scipy's `filtfilt` returns bit-identical values for this input,
    // including its 0.587 tail), so the middle is the part that carries the
    // filter's response.
    let fs = 1000.0;
    let amplitude = |f: f64| {
        let signal: Vec<f64> = (0..400)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin())
            .collect();
        let out = filtfilt(&b, &a, &signal);
        let middle = &out[100..300];
        let rms = (middle.iter().map(|v| v * v).sum::<f64>() / middle.len() as f64).sqrt();
        rms * 2f64.sqrt()
    };

    let passband = amplitude(10.0);
    assert!(
        (passband - 1.0).abs() < 0.05,
        "a 10 Hz tone through a 100 Hz low-pass must pass, got amplitude {}",
        passband
    );
    let stopband = amplitude(400.0);
    assert!(
        stopband < 0.01,
        "a 400 Hz tone through a 100 Hz low-pass must be attenuated, got amplitude {}",
        stopband
    );
}
