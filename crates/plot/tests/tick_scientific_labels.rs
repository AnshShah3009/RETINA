//! Regression: scientific tick labels corrupted their own exponent.
//!
//! `format_tick_value` formats a magnitude outside `1e-4 .. 1e5` with
//! `format!("{:.3e}", v)` and then trims trailing zeros from the *whole*
//! string. That string ends with the exponent, not the mantissa, so a value
//! like `1e10` ("1.000e10") became "1.000e1" - a tick labelled **10** on an
//! axis whose values are around ten billion. The mantissa zeros the trimming
//! was meant to remove were left alone.
//!
//! These tests read the labels a human would read, because the corruption is in
//! the string: the label still parses as a finite number, so every numeric
//! invariant (ticks are finite, ticks are inside the axis range) still passes.

use cv_plot::chart::Figure;
use cv_plot::export::to_svg;

/// Numeric `<text>` labels carrying the given axis class, in emission order.
fn axis_labels(svg: &str, axis_class: &str) -> Vec<String> {
    let needle = format!("class=\"tick {axis_class}\"");
    let mut out = Vec::new();
    for line in svg.lines() {
        if !line.contains(&needle) || !line.contains("<text") {
            continue;
        }
        let after_class = &line[line.find(&needle).unwrap() + needle.len()..];
        if let Some(gt) = after_class.find('>') {
            let tail = &after_class[gt + 1..];
            if let Some(end) = tail.find('<') {
                out.push(tail[..end].trim().to_string());
            }
        }
    }
    out
}

fn svg_for(xs: &[f64], ys: &[f64]) -> String {
    let mut f = Figure::new("scientific ticks");
    f.scatter(xs, ys, "s");
    to_svg(&f)
}

/// A tick on a `1e10`-scale axis must not be labelled as if it were order 1.
///
/// The corrupted exponent made the two largest ticks read as `10` and `12.5`,
/// which is both wrong and no longer increasing along the axis.
#[test]
fn large_ticks_keep_their_exponent() {
    let svg = svg_for(&[0.0, 1e10], &[0.0, 1.0]);
    let labels = axis_labels(&svg, "tick-x");
    assert!(!labels.is_empty(), "no x tick labels in:\n{svg}");

    for lbl in &labels {
        let v: f64 = lbl
            .parse()
            .unwrap_or_else(|_| panic!("tick label {lbl:?} is not a number"));
        if v != 0.0 {
            assert!(
                v.abs() >= 1e9,
                "tick on a 1e10-scale axis is labelled {lbl:?} (parsed as {v}), \
                 as if it were order 1; labels were {labels:?}"
            );
        }
    }

    // Monotonicity is a second, independent symptom: trimming the exponent
    // turned the two largest ticks into 10 and 12.5, below the 7.5e9 tick
    // before them.
    let parsed: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
    for pair in parsed.windows(2) {
        assert!(
            pair[1] > pair[0],
            "x tick labels are not increasing: {parsed:?}"
        );
    }
}

/// The same corruption hit negative exponents: `1e-10` ("1.000e-10") was
/// trimmed to "1.000e-1", labelling a tiny tick as `0.1`.
#[test]
fn small_ticks_keep_their_exponent() {
    let svg = svg_for(&[0.0, 1e-10], &[0.0, 1.0]);
    let labels = axis_labels(&svg, "tick-x");
    assert!(!labels.is_empty(), "no x tick labels in:\n{svg}");

    for lbl in &labels {
        let v: f64 = lbl
            .parse()
            .unwrap_or_else(|_| panic!("tick label {lbl:?} is not a number"));
        if v != 0.0 {
            assert!(
                v.abs() <= 1e-9,
                "tick on a 1e-10-scale axis is labelled {lbl:?} (parsed as {v}); \
                 labels were {labels:?}"
            );
        }
    }
}
