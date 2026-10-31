//! Normal cdf / ppf and test p-values in the tails, against `scipy.stats`.
//!
//! Before the fix:
//! * `NormalDistribution::cdf` was `0.5 * (1 + erf(z))`, which cancels in the
//!   lower tail: `cdf(-10) = 0` (SciPy `7.61985302416047e-24`).
//! * `ppf` was Acklam's rational approximation (documented as A&S 26.2.23)
//!   with no refinement: relative error ~1e-9.
//! * p-values were `2 * (1 - cdf(|t|))`, so anything below ~1e-16 became 0:
//!   `linregress` on a near-perfect line returned `p = 0` (SciPy 1.44e-96).
//! * Degeneracy tests were absolute `1e-30`: Pearson on data of size 1e-200
//!   was "zero variance".

use cv_math::stats::*;

fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        0.0
    } else {
        ((a - b) / b).abs()
    }
}

#[test]
fn normal_cdf_and_sf_tails() {
    let n = NormalDistribution::new(0.0, 1.0);
    // scipy.stats.norm.cdf / .sf
    let table = [
        (-10.0, 7.61985302416047e-24, 1.0),
        (-5.0, 2.866515718791933e-07, 0.9999997133484281),
        (-1.0, 0.15865525393145707, 0.8413447460685429),
        (0.0, 0.5, 0.5),
        (1.5, 0.9331927987311419, 0.06680720126885807),
        (8.0, 0.9999999999999993, 6.22096057427174e-16),
        (10.0, 1.0, 7.61985302416047e-24),
    ];
    // The tail's condition number is ~x², and z = x/√2 is itself rounded, so
    // ~1e-14 relative at |x| = 10 is the attainable accuracy, not a defect.
    for (x, cdf, sf) in table {
        assert!(
            rel(n.cdf(x), cdf) < 5e-14,
            "cdf({x}) = {:e}, SciPy {cdf:e}",
            n.cdf(x)
        );
        assert!(
            rel(n.sf(x), sf) < 5e-14,
            "sf({x}) = {:e}, SciPy {sf:e}",
            n.sf(x)
        );
    }
    assert_eq!(n.cdf(-40.0), 0.0); // underflows in SciPy too
    let m = NormalDistribution::new(2.0, 3.0);
    assert!(rel(m.cdf(-25.0), 1.1285884059538324e-19) < 1e-13);
}

#[test]
fn normal_ppf_is_refined_to_machine_precision() {
    let n = NormalDistribution::new(0.0, 1.0);
    // scipy.stats.norm.ppf
    let table = [
        (1e-300, -37.0470962993612),
        (1e-100, -21.273453560965322),
        (1e-10, -6.361340902404056),
        (1e-3, -3.090232306167813),
        (0.02, -2.053748910631823),
        (0.02425, -1.972961051311885),
        (0.3, -0.5244005127080409),
        (0.7, 0.5244005127080407),
        (0.975, 1.959963984540054),
        (0.999, 3.090232306167813),
        (1.0 - 1e-10, 6.361340889697422),
    ];
    for (p, z) in table {
        assert!(
            rel(n.ppf(p), z) < 4e-15,
            "ppf({p:e}) = {}, SciPy {z}",
            n.ppf(p)
        );
    }
    assert_eq!(n.ppf(0.5), 0.0);
    assert_eq!(n.ppf(0.0), f64::NEG_INFINITY);
    assert_eq!(n.ppf(1.0), f64::INFINITY);
    assert!(n.ppf(f64::NAN).is_nan());
}

#[test]
fn linregress_p_value_is_not_rounded_to_zero() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| 2.0 * v + 1.0 + 0.01 * (1.7 * v).sin())
        .collect();
    let r = linregress(&x, &y).unwrap();
    // scipy.stats.linregress
    assert!(rel(r.slope, 1.9999273899390166) < 1e-12);
    assert!(rel(r.intercept, 1.0009788929899415) < 1e-12);
    assert!(rel(r.r_squared, 0.9999998366113401) < 1e-12);
    assert!(rel(r.std_err, 0.00015277285809039284) < 1e-9);
    assert!(
        rel(r.p_value, 1.4440792375649412e-96) < 1e-6,
        "p = {:e}",
        r.p_value
    );
}

#[test]
fn t_test_p_values_in_the_tail() {
    let d: Vec<f64> = (0..20)
        .map(|i| 10.0 + 0.01 * (i as f64 * 1.3).sin())
        .collect();
    let (t, p) = t_test_1sample(&d, 9.9);
    // scipy.stats.ttest_1samp
    assert!(rel(t, 62.73891879428504) < 1e-10);
    assert!(rel(p, 1.7095648909473782e-23) < 1e-6, "p = {p:e}");

    let a: Vec<f64> = (0..15)
        .map(|i| 1.0 + 0.1 * (i as f64 * 0.9).cos())
        .collect();
    let b: Vec<f64> = (0..12)
        .map(|i| 2.0 + 0.2 * (i as f64 * 1.1).sin())
        .collect();
    let (t, p) = t_test_2sample(&a, &b);
    // scipy.stats.ttest_ind(equal_var=False)
    assert!(rel(t, -21.59059882128932) < 1e-10);
    assert!(rel(p, 4.0128437581021173e-13) < 1e-6, "p = {p:e}");

    let (t, p) = t_test_1sample(&[4.5, 5.1, 4.8, 5.3, 5.0, 4.9, 5.2, 4.7], 5.0);
    assert!(rel(t, -0.662266178532522) < 1e-10);
    assert!(rel(p, 0.5289936281086356) < 1e-10, "p = {p}");
}

#[test]
fn degeneracy_tests_are_relative() {
    let x: Vec<f64> = (1..6).map(|i| 1e-200 * i as f64).collect();
    let y: Vec<f64> = x.iter().rev().map(|v| 3.0 * v + 1e-199).collect();
    // scipy.stats.pearsonr: -1.0
    let r = pearson_correlation(&x, &y).expect("data of size 1e-200 has variance");
    assert!((r + 1.0).abs() < 1e-12, "r = {r}");
    // A constant, whose mean is not exactly representable, is still constant.
    assert!(pearson_correlation(&[0.1, 0.1, 0.1], &[1.0, 2.0, 3.0]).is_err());
    assert!(t_test_1sample(&[0.1, 0.1, 0.1], 0.0).0.is_nan());
    let tiny: Vec<f64> = (0..5).map(|i| 1e-20 * i as f64).collect();
    let r = linregress(&tiny, &tiny).expect("x of size 1e-20 has variance");
    assert!((r.slope - 1.0).abs() < 1e-12);
}
