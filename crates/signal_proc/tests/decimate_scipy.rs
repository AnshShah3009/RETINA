//! `decimate` must match `scipy.signal.decimate(x, q)` with SciPy's defaults:
//! an order-8 Chebyshev type I lowpass with 0.05 dB passband ripple and edge
//! `0.8 / q` (of Nyquist), applied forward and backward, then every `q`-th
//! sample.
//!
//! It used an order-8 *Butterworth* at the same edge. Butterworth is 3 dB down
//! at its cutoff, so after the forward-backward pass the top of the band that
//! decimation is supposed to keep was cut: a tone at 0.18 cycles/sample (below
//! the new Nyquist of 0.25 for `q = 2`) came back with amplitude 0.895, where
//! SciPy keeps 0.994. On a mixed signal the output differed from SciPy's by up
//! to 0.23 on samples of size ~2.
//!
//! References: SciPy 1.17.1.
//!
//! ```python
//! i = np.arange(120.0)
//! x = np.sin(0.3 * i) + 0.5 * np.cos(1.7 * i) + 0.1 * (i % 7)
//! scipy.signal.decimate(x, 2), scipy.signal.decimate(x, 5)
//! ```

use cv_signal::signal::decimate;

fn input() -> Vec<f64> {
    (0..120)
        .map(|i| {
            let t = i as f64;
            (0.3 * t).sin() + 0.5 * (1.7 * t).cos() + 0.1 * (i % 7) as f64
        })
        .collect()
}

fn assert_matches(q: usize, want: &[f64]) {
    let got = decimate(&input(), q);
    assert_eq!(got.len(), want.len(), "q={q}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-9, "q={q}: max |decimate - scipy| = {worst:.3e}");
}

#[test]
fn decimate_by_2_matches_scipy() {
    assert_matches(2, &SCIPY_Q2);
}

#[test]
fn decimate_by_5_matches_scipy() {
    assert_matches(5, &SCIPY_Q5);
}

/// A tone inside the band decimation keeps survives with SciPy's amplitude.
///
/// `np.abs(scipy.signal.decimate(np.sin(2*np.pi*0.18*np.arange(400.0)), 2)[20:-20]).max()`
/// is `0.9941923506280773`.
#[test]
fn passband_tone_keeps_scipy_amplitude() {
    let x: Vec<f64> = (0..400)
        .map(|i| (2.0 * std::f64::consts::PI * 0.18 * i as f64).sin())
        .collect();
    let y = decimate(&x, 2);
    let peak = y[20..y.len() - 20]
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        (peak - 0.9941923506280773).abs() < 1e-9,
        "passband peak {peak}, SciPy 0.9941923506280773"
    );
}

#[rustfmt::skip]
const SCIPY_Q2: [f64; 60] = [
    0.49686205908467096, 0.543944025454267, 1.470966496631445, 1.35088736391257,
    0.7345232564776774, 0.4667528409102811, 0.042821092341306495, -0.6462051518761119,
    -0.8951519623719021, -0.2914452957217042, 0.11969353288601185, 0.3753979810224524,
    1.1029409745979086, 1.5058149975381527, 1.0600094979332289, 0.527056848970457,
    0.2991138039163797, -0.2991914372729022, -0.9130144533970171, -0.6172707773565007,
    -0.020665953643646123, 0.23209518589495953, 0.7110686456886472, 1.4163162321713674,
    1.3639869158912328, 0.7217311159788878, 0.40573001400378805, 0.04653544164273354,
    -0.6905397808158388, -0.8734515803369054, -0.2769738113662437, 0.14975192140897794,
    0.41652914991172363, 1.1103147531049005, 1.5191018937749323, 1.0335947134021986,
    0.4996842816290607, 0.2673752907378187, -0.3273752194682381, -0.9131376519071018,
    -0.6087923361615006, 0.011444892251917652, 0.26449538621720803, 0.7370538582431734,
    1.4288731653160667, 1.3524159495460693, 0.6977236333002232, 0.37112902745760235,
    0.01745795556927279, -0.7051911179695676, -0.8694036738026943, -0.2541125742560036,
    0.18315859507253596, 0.4459471028544297, 1.130834686329572, 1.5217752622294771,
    1.0160789840453541, 0.48093174525831206, 0.23746982429685198, -0.3307179655445801,
];

#[rustfmt::skip]
const SCIPY_Q5: [f64; 24] = [
    0.49826198230908425, 1.2614046475411167, 0.4470934882570687, -0.6847758274844568,
    0.01602199503573307, 1.2358273830655042, 0.7014845560302532, -0.5749240701716747,
    -0.24721218014654792, 1.1069230805143186, 0.9407246939358179, -0.41287912888113176,
    -0.45000032511247573, 0.8925181401959574, 1.1452174624777542, -0.20471902957229968,
    -0.5943701624925845, 0.6444230549596832, 1.2554960751569577, 0.09818732019574736,
    -0.7157507145389393, 0.4398796542732688, 1.2248544175275669, 0.4581892536805343,
];
