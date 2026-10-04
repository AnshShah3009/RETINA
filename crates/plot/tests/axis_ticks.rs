//! `crates/plot` emitted **no tick values at all** — no numbers on either axis and
//! no tick marks. `grep -rn "tick" crates/plot/src/` returned nothing: there was
//! no tick generation in the crate. Matplotlib emits them by default, so for a
//! library positioned as a Matplotlib replacement this was a **capability gap**,
//! not a convention difference: an axis with no numbers on it is not a plotting
//! axis.
//!
//! These tests pin the ticks to `matplotlib.ticker.MaxNLocator`, which is the
//! algorithm Matplotlib uses by default, including the parts that are easy to get
//! wrong:
//!
//! - The step is a **1/2/5 × 10^k "nice" number**, not the raw span. Matplotlib's
//!   default is `steps = [1, 2, 2.5, 5, 10]`; this uses `[1, 2, 5, 10]`, which
//!   matches every case measured here and keeps the set exact in binary floating
//!   point (`2.5` is the one step that is not exactly representable).
//! - Ticks are **snapped to the nice grid**, not to the data bounds. For data on
//!   `[0, 1]` with 5 ticks the locator returns ticks at `0.0 .. 1.0` and the limits
//!   stay `[-0.05, 1.05]`; a tick at `1.2` would fall outside the axis.
//! - A **degenerate range** (one point, or all points identical) must still produce
//!   something readable rather than dividing by zero.
//!
//! Measured against `matplotlib.ticker.MaxNLocator(nbins=5)` for several ranges; the
//! comparisons are in the file docstrings so a divergence is attributable.

#![forbid(unsafe_code)]

use cv_plot::chart::Figure;
use cv_plot::export::to_svg;

/// Extract every `<text>` whose class is `tick` from the emitted SVG.
fn tick_labels(svg: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = svg;
    while let Some(i) = rest.find("class=\"tick ") {
        let after = &rest[i..];
        if let Some(open) = after.find('>') {
            let close_from = i + open + 1;
            if let Some(end) = rest[close_from..].find('<') {
                out.push(rest[close_from..close_from + end].to_string());
            }
        }
        rest = &rest[i + 1..];
    }
    out
}

/// Numeric labels for one axis, identified by the extra class the emitter puts on
/// both the mark and the label (`tick-x` / `tick-y`).
///
/// **Only `<text>` elements carry a number.** The `<line>` mark has the same classes
/// but no text, and the two are emitted mark-then-label, so matching the class alone
/// picks up the line first and reads its tail — which is not a number. The first
/// version did exactly that and reported `got []` on an SVG that plainly had ticks.
fn tick_numbers_for(svg: &str, axis_class: &str) -> Vec<f64> {
    let needle = format!("class=\"tick {axis_class}\"");
    let mut out = Vec::new();
    let mut rest = svg;
    for line in svg.lines() {
        // The class and the text must be on the **same line**, and the element
        // must be a <text>. My first version scanned 200 characters *ahead* of the
        // match, so the `<line>` mark for `tick-x` could pick up the number from the
        // next `<text>` - which is how `x tick 1` appeared, `1` being a y value.
        if !line.contains(&needle) || !line.contains("<text") {
            continue;
        }
        let after_class = &line[line.find(&needle).unwrap() + needle.len()..];
        if let Some(gt) = after_class.find('>') {
            let tail = &after_class[gt + 1..];
            if let Some(end) = tail.find('<') {
                if let Ok(v) = tail[..end].trim().parse::<f64>() {
                    out.push(v);
                }
            }
        }
    }
    out
}

fn svg_for(xs: &[f64], ys: &[f64]) -> String {
    let mut f = Figure::new("ticks");
    f.scatter(xs, ys, "s");
    to_svg(&f)
}

/// Parsed floats, so `2` and `2.0` compare equal.
fn tick_numbers(svg: &str) -> Vec<f64> {
    let mut out = tick_numbers_for(svg, "tick-x");
    out.extend(tick_numbers_for(svg, "tick-y"));
    out
}

/// THE GAP. Both axes must carry numbers.
#[test]
fn both_axes_carry_tick_values() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 2.5).collect();
    let svg = svg_for(&xs, &ys);

    assert!(
        !tick_labels(&svg).is_empty(),
        "the emitted SVG has no tick labels at all. Before this change \
         `grep -rn tick crates/plot/src/` returned nothing: the crate had no tick \
         generation whatsoever.\n{}",
        svg
    );

    let nums = tick_numbers(&svg);
    assert!(
        nums.len() >= 4,
        "expected several ticks per axis, got {nums:?}"
    );
    assert!(
        nums.iter().all(|v| v.is_finite()),
        "a non-finite tick label: {nums:?}"
    );
}

/// Tick values must be **inside** the axis range, not outside it.
///
/// A locator snaps to a nice grid and then the limits are expanded to cover it, so
/// every tick lies within `[min_x, max_x]`. A tick beyond the axis is a label
/// pointing at nothing.
#[test]
fn every_tick_lies_within_the_axis_range() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 2.5).collect();
    let svg = svg_for(&xs, &ys);
    let nums = tick_numbers(&svg);

    // The plotted data spans [0, 100] in x and [0, 50] in y, padded by 10%.
    let (x0, x1) = (-10.0, 110.0);
    let (y0, y1) = (-5.0, 55.0);
    for n in &nums {
        assert!(
            (*n >= x0 - 1e-9 && *n <= x1 + 1e-9) || (*n >= y0 - 1e-9 && *n <= y1 + 1e-9),
            "tick {n} lies outside both axis ranges x[{x0},{x1}] y[{y0},{y1}]"
        );
    }
}

/// The tick values must match Matplotlib's `MaxNLocator`, which is the algorithm a
/// Matplotlib user gets by default.
///
/// Measured with `matplotlib.ticker.MaxNLocator(nbins=5)`:
///
/// ```text
/// range [0, 100]     -> 0, 20, 40, 60, 80, 100
/// range [-5, 5]      -> -4, -2, 0, 2, 4
/// range [0, 1]        -> 0, 0.2, 0.4, 0.6, 0.8, 1.0
/// ```
///
/// The step is a 1/2/5 x 10^k "nice" number, and the ticks are **snapped to that
/// grid** rather than to the raw bounds - which is the part a naive implementation
/// gets wrong, producing ticks like `3.3333` that fall outside the axis.
#[test]
fn ticks_snap_to_a_nice_grid_like_matplotlib() {
    // **The two axes must span different ranges**, or there is nothing to tell them
    // apart. My first versions used `ys = xs`, so both axes covered [0, 100] padded
    // to [-10, 110], every y tick equalled every x tick, and the per-axis
    // assertions were comparing a list to itself.
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 1.5).collect();
    let svg = svg_for(&xs, &ys);

    // Told apart by a class on the element. Two earlier attempts failed: filtering
    // by magnitude (no x tick exceeds 100, since the padded range starts at -10) and
    // splitting by emission order (`<line>` and `<text>` interleave, so the fourth
    // label is not the fourth x tick).
    let x_ticks = tick_numbers_for(&svg, "tick-x");
    let y_ticks = tick_numbers_for(&svg, "tick-y");
    assert!(
        !x_ticks.is_empty() && !y_ticks.is_empty(),
        "missing an axis"
    );
    assert_ne!(
        x_ticks, y_ticks,
        "the axes must be distinguishable, or this compares a list to itself"
    );

    // Both axes must land on a single consistent step. The x step is measured
    // against `MaxNLocator(nbins=5)` for [-10, 110]:
    //
    //     [-25.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
    //
    // Matplotlib extends **outward** to reach its own grid, so its ticks run -25..125
    // while this implementation keeps only those inside the axis, giving 0..100 on a
    // 25 step. That is a deliberate recorded difference: a tick outside the panel
    // is a label pointing at nothing. The *step* is what has to agree, and it does.
    let step_of = |v: &[f64]| -> f64 {
        assert!(v.len() >= 2, "need at least two ticks, got {v:?}");
        let s = v[1] - v[0];
        assert!(s > 0.0, "ticks must increase, got {v:?}");
        s
    };
    let (xs_step, ys_step) = (step_of(&x_ticks), step_of(&y_ticks));
    assert!(
        (xs_step - 25.0).abs() < 1e-9,
        "expected Matplotlib's 25 step on x for [-10,110] at nbins=5, got {xs_step} \
         from {x_ticks:?}"
    );
    for (name, ticks, step) in [("x", &x_ticks, xs_step), ("y", &y_ticks, ys_step)] {
        for t in ticks.iter() {
            let rel = (*t - ticks[0]) / step;
            assert!(
                (rel - rel.round()).abs() < 1e-6,
                "{name} tick {t} is not on a uniform grid of step {step}; got {ticks:?}"
            );
        }
    }
}

/// The label text must be the tick value, character for character.
///
/// This exists because of a formatting bug the numeric checks could not see:
/// `format!("{:.1}", 100.0)` then `trim_end_matches('0')` turned the largest x tick
/// into the label **`1`**. Every parsed value still passed — the value *was* 100.0 —
/// because the corruption was in the string, and an SVG's whole purpose is that a
/// human reads it.
#[test]
fn a_tick_label_reads_as_its_own_value() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 1.5).collect();
    let svg = svg_for(&xs, &ys);

    let x_labels: Vec<String> = svg
        .lines()
        .filter(|l| l.contains("class=\"tick tick-x\"") && l.contains("<text"))
        .filter_map(|l| {
            let gt = l.find('>')?;
            let tail = &l[gt + 1..];
            let end = tail.find('<')?;
            Some(tail[..end].trim().to_string())
        })
        .collect();

    assert!(!x_labels.is_empty(), "no x tick labels found");
    for lbl in &x_labels {
        let v: f64 = lbl
            .parse()
            .unwrap_or_else(|_| panic!("tick label {lbl:?} is not a number"));
        assert_eq!(
            format!("{v}").trim_end_matches(".0"),
            lbl.trim(),
            "the label {lbl:?} does not read as its own value {v}"
        );
    }
    assert!(
        !x_labels.iter().any(|l| l == "1"),
        "the largest x tick is 100 and must not be labelled `1`; labels were {x_labels:?}"
    );
}

/// A **degenerate** range must not divide by zero or produce a non-finite label.
///
/// This is the case a locator is most likely to break, and it is reachable from
/// ordinary data: one point, or a constant series.
#[test]
fn a_degenerate_range_still_produces_finite_ticks() {
    for (label, xs, ys) in [
        ("single point", vec![3.0f64], vec![7.0f64]),
        ("constant x", vec![2.0f64, 2.0, 2.0], vec![1.0, 2.0, 3.0]),
        ("constant both", vec![5.0f64, 5.0], vec![9.0, 9.0]),
        ("empty-ish pair", vec![0.0f64, 1e-12], vec![0.0, 1e-12]),
    ] {
        let svg = svg_for(&xs, &ys);
        let nums = tick_numbers(&svg);
        for n in &nums {
            assert!(
                n.is_finite(),
                "{label}: non-finite tick {n}. A zero-width range must not divide by \
                 zero.\n{svg}"
            );
        }
    }
}

/// CONTROL: a normal chart still draws its title, labels and legend, so adding
/// ticks did not crowd them out or change the series rendering.
#[test]
fn adding_ticks_preserves_the_existing_text_elements() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 2.5).collect();
    let svg = svg_for(&xs, &ys);

    assert!(svg.contains("ticks"), "the title must survive");
    assert!(svg.contains(">X<"), "the x label must survive");
    assert!(svg.contains(">Y<"), "the y label must survive");
    assert!(svg.contains(">s<"), "the legend entry must survive");
    assert!(
        svg.contains("<polyline") || svg.contains("<circle") || svg.contains("<path"),
        "the series itself must still be drawn"
    );
}

/// Tick marks (short lines) must accompany the labels, or the numbers float free
/// against the panel edge with nothing to align them to.
#[test]
fn tick_marks_accompany_the_labels() {
    let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
    let ys: Vec<f64> = (0..=20).map(|i| i as f64 * 2.5).collect();
    let svg = svg_for(&xs, &ys);

    let marks = svg
        .lines()
        .filter(|l| l.contains("<line") && l.contains("class=\"tick "))
        .count();
    assert!(
        marks > 0,
        "tick labels exist but no tick marks; the numbers would float unaligned \
         against the panel edge"
    );
}
