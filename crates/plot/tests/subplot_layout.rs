//! Subplot layout, styling and degenerate-input coverage for the 2D renderer.
//!
//! Everything here is checked against numbers derived from the input data, not
//! from the emitted markup: the panel rectangles are computed by hand from the
//! documented layout rules (`60/30/50/50` outer margins, `45/20/28/38` cell
//! insets, 10% padding with a floor of 1.0 on the raw span) so a wrong answer
//! means the projection is wrong, not that the markup was reformatted.
//!
//! The tests deliberately exercise only the public API that predates the
//! subplot fix, so the file can be compiled against the unfixed crate to show
//! that it fails there.

use cv_plot::{Figure, PlotError, PlotType, Series, Style, SubPlot, COLORS};

/// Unique-enough path for a save that is expected to succeed; the process id
/// and a counter keep concurrent tests off each other's files.
fn unique_tmp(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "cv_plot_subplot_{}_{}_{}.tmp",
        std::process::id(),
        tag,
        n
    ))
}

// ---------------------------------------------------------------- SVG reading

/// `(open_tag, offset_just_past_the_closing_angle_bracket)` for every `marker`.
fn tag_spans<'a>(svg: &'a str, marker: &str) -> Vec<(&'a str, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = svg[i..].find(marker) {
        let start = i + rel;
        let gt = start + svg[start..].find('>').expect("unterminated tag");
        spans.push((&svg[start..=gt], gt + 1));
        i = gt + 1;
    }
    spans
}

fn tags_with<'a>(svg: &'a str, marker: &str) -> Vec<&'a str> {
    tag_spans(svg, marker).into_iter().map(|(t, _)| t).collect()
}

fn attr(tag: &str, name: &str) -> String {
    let key = format!("{}=\"", name);
    let start = tag
        .find(&key)
        .unwrap_or_else(|| panic!("attribute `{name}` missing from tag: {tag}"))
        + key.len();
    let end = start + tag[start..].find('"').expect("unterminated attribute");
    tag[start..end].to_string()
}

fn attr_f64(tag: &str, name: &str) -> f64 {
    attr(tag, name)
        .parse()
        .unwrap_or_else(|_| panic!("attribute `{name}` is not numeric in tag: {tag}"))
}

/// `(x, y)` pairs of a polyline's `points` attribute.
fn points(tag: &str) -> Vec<(f64, f64)> {
    attr(tag, "points")
        .split_whitespace()
        .map(|p| {
            let (x, y) = p.split_once(',').expect("point is not `x,y`");
            (
                x.parse().expect("non-numeric x"),
                y.parse().expect("non-numeric y"),
            )
        })
        .collect()
}

fn texts_of_class(svg: &str, class: &str) -> Vec<String> {
    tag_spans(svg, "<text")
        .into_iter()
        .filter(|(t, _)| t.contains(&format!("class=\"{class}\"")))
        .map(|(_, after)| {
            let close = svg[after..]
                .find("</text>")
                .unwrap_or_else(|| panic!("<text> was never closed in: {svg}"))
                + after;
            svg[after..close].trim().to_string()
        })
        .collect()
}

fn fills_of(svg: &str, marker: &str) -> Vec<String> {
    tags_with(svg, marker)
        .iter()
        .map(|t| attr(t, "fill"))
        .collect()
}

fn strokes_of(svg: &str, marker: &str) -> Vec<String> {
    tags_with(svg, marker)
        .iter()
        .map(|t| attr(t, "stroke"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 0.05,
        "{what}: expected {expected}, got {actual}"
    );
}

// ------------------------------------------------------- the layout itself

/// `subplot(rows, cols, index)` must select the panel `index` names and the
/// figure must be drawn as that grid, each panel scaled to its own data.
///
/// Two 2x2 panels are filled with data whose ranges differ by two orders of
/// magnitude. A correct renderer gives both panels the same screen span; the
/// defect gave panel 0 a span of 2.9 px of a 710 px axis because every panel
/// was drawn into a single axes against one global data range, and it put both
/// series into panel 3 regardless of `index`.
#[test]
fn subplot_index_selects_the_panel_and_every_panel_scales_to_its_own_data() {
    let mut fig = Figure::new("two panels")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.subplot(2, 2, 0)
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "small");
    fig.subplot(2, 2, 1)
        .add_series(&[100.0, 200.0], &[1000.0, 2000.0], "large");

    let svg = cv_plot::to_svg(&fig);

    // The series must have landed in the panels `index` asked for.
    assert_eq!(
        fig.subplots[0].series.len(),
        1,
        "panel 0 takes the first series"
    );
    assert_eq!(fig.subplots[1].series.len(), 1, "panel 1 takes the second");
    assert!(
        fig.subplots[2].series.is_empty() && fig.subplots[3].series.is_empty(),
        "subplot(2, 2, 0)/(2, 2, 1) must not write into panels 2 and 3"
    );

    let polylines = tags_with(&svg, "<polyline");
    assert_eq!(polylines.len(), 2, "one polyline per panel: {svg}");

    // Hand-computed geometry for an 800x600 figure, 2x2:
    //   interior   = x 60, y 50, w 710, h 500
    //   cell       = 355 x 250
    //   panel 0    = x 105, y 78, w 290, h 184   (cell + 45/28, minus 20/38)
    //   panel 1    = x 460, y 78, w 290, h 184
    // Data 0..1 is padded to -0.1..1.1 (span 1.2); 100..200 to 90..210 (120);
    // 1000..2000 to 900..2100 (1200).
    let p_small = points(polylines[0]);
    let p_large = points(polylines[1]);
    assert_close(p_small[0].0, 105.0 + (0.1 / 1.2) * 290.0, "panel 0 x of 0");
    assert_close(p_small[1].0, 105.0 + (1.1 / 1.2) * 290.0, "panel 0 x of 1");
    assert_close(p_small[0].1, 262.0 - (0.1 / 1.2) * 184.0, "panel 0 y of 0");
    assert_close(p_small[1].1, 262.0 - (1.1 / 1.2) * 184.0, "panel 0 y of 1");
    assert_close(
        p_large[0].0,
        460.0 + (10.0 / 120.0) * 290.0,
        "panel 1 x of 100",
    );
    assert_close(
        p_large[1].0,
        460.0 + (110.0 / 120.0) * 290.0,
        "panel 1 x of 200",
    );
    assert_close(
        p_large[0].1,
        262.0 - (100.0 / 1200.0) * 184.0,
        "panel 1 y of 1000",
    );
    assert_close(
        p_large[1].1,
        262.0 - (1100.0 / 1200.0) * 184.0,
        "panel 1 y of 2000",
    );

    // The point of the whole exercise: both panels use their own range, so the
    // same two-point series has the same screen extent in each. Against one
    // global range the small panel would be 240x smaller.
    assert_close(
        p_small[1].0 - p_small[0].0,
        p_large[1].0 - p_large[0].0,
        "panels must share the same span despite different data ranges",
    );
    assert!(
        p_small[1].0 - p_small[0].0 > 200.0,
        "panel 0 spans only {:.1} px - it is being scaled against another panel's data",
        p_small[1].0 - p_small[0].0
    );

    // Each panel draws its own axis box: 2 lines per panel, 4 panels. The two
    // left edges and the two baselines are shared between the columns/rows.
    let axis_lines: Vec<&str> = tags_with(&svg, "<line")
        .into_iter()
        .filter(|t| t.contains("stroke=\"black\""))
        .collect();
    assert_eq!(axis_lines.len(), 8, "two axis lines per panel: {svg}");
    let mut verticals: Vec<f64> = axis_lines
        .iter()
        .filter(|t| attr_f64(t, "x1") == attr_f64(t, "x2"))
        .map(|t| attr_f64(t, "x1"))
        .collect();
    verticals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    verticals.dedup();
    assert_eq!(verticals, vec![105.0, 460.0], "panel left edges: {svg}");
    let mut horizontals: Vec<f64> = axis_lines
        .iter()
        .filter(|t| attr_f64(t, "y1") == attr_f64(t, "y2"))
        .map(|t| attr_f64(t, "y1"))
        .collect();
    horizontals.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    horizontals.dedup();
    assert_eq!(
        horizontals,
        vec![262.0, 512.0],
        "panel baselines, one row per cell height: {svg}"
    );
}

/// A 2x2 grid's four panels must occupy four different cells: the second row
/// starts below the first. A renderer that ignores the grid draws one box.
#[test]
fn second_row_of_a_grid_sits_below_the_first() {
    let mut fig = Figure::new("grid")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    for i in 0..4 {
        fig.subplot(2, 2, i)
            .add_series(&[0.0, 1.0], &[0.0, 1.0], "s");
    }
    let svg = cv_plot::to_svg(&fig);
    let polylines = tags_with(&svg, "<polyline");
    assert_eq!(polylines.len(), 4, "one polyline per panel: {svg}");

    let tops: Vec<f64> = polylines.iter().map(|t| points(t)[1].1).collect();
    // Cell height is 250 and the panel is inset 28 from the top of its cell,
    // so row 1 sits (250 - 28 + 28) - (28) = 250 px lower than row 0.
    assert_close(
        tops[2] - tops[0],
        250.0,
        "row 1 must be one cell below row 0",
    );
    assert_close(tops[3] - tops[1], 250.0, "row 1 column 1 likewise");
    assert_close(tops[1] - tops[0], 0.0, "panels in one row share a baseline");
}

/// Degenerate grid arguments must not panic, allocate without bound, or leave
/// the panel cursor pointing outside `subplots`.
#[test]
fn degenerate_grid_arguments_are_clamped_not_fatal() {
    // Zero rows/columns: treated as a 1x1 grid rather than an empty one.
    let mut fig = Figure::new("zero");
    fig.subplot(0, 0, 0)
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "s");
    assert_eq!(
        fig.subplots.len(),
        1,
        "a 0x0 grid must still hold one panel"
    );
    assert_eq!(fig.subplots[0].series.len(), 1, "and stay usable");

    // An index past the end selects the last panel instead of panicking.
    let mut fig = Figure::new("past the end");
    fig.subplot(1, 2, 99).add_series(&[0.0], &[0.0], "s");
    assert_eq!(fig.subplots.len(), 2);
    assert_eq!(
        fig.subplots[1].series.len(),
        1,
        "index 99 clamps to the last panel"
    );

    // A product that overflows `usize` must not wrap into a wrong grid (debug
    // builds panic on the overflow) nor try to allocate it.
    let mut fig = Figure::new("huge");
    fig.subplot(usize::MAX, 2, 0);
    assert!(
        fig.subplots.len() <= 4096,
        "subplot(usize::MAX, 2, 0) grew the panel list to {}",
        fig.subplots.len()
    );
    fig.add_series(&[0.0, 1.0], &[0.0, 1.0], "s");
    assert_eq!(
        fig.subplots[0].series.len(),
        1,
        "the first panel is still usable"
    );

    // Control: a normal grid is still exactly rows*cols panels.
    let mut fig = Figure::new("normal");
    fig.subplot(3, 4, 0);
    assert_eq!(fig.subplots.len(), 12, "3x4 must yield 12 panels");

    // A caller who empties the public `subplots` vector must get a no-op
    // builder and a still-well-formed renderer, not a panic or an index into
    // nothing.
    let mut empty = Figure::new("cleared");
    empty.subplots.clear();
    empty
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "nowhere")
        .scatter(&[0.0], &[0.0], "nowhere")
        .title("nowhere")
        .labels("x", "y");
    assert!(empty.subplots.is_empty(), "nothing may be added back");
    let svg = cv_plot::to_svg(&empty);
    assert!(svg.starts_with("<svg ") && svg.ends_with("</svg>"));
    let path = unique_tmp("cleared");
    assert!(
        matches!(
            cv_plot::save_svg(&empty, path.to_str().unwrap()),
            Err(PlotError::InvalidData(_))
        ),
        "a figure with no panels has no plottable points and must be rejected"
    );
    assert!(!path.exists());
}

// ------------------------------------------------------------ colour/legend

/// Default-styled series must come out in different palette colours, and the
/// legend swatch must be the colour the series is actually drawn in.
///
/// The `Series` constructors all use `Style::default()`, whose colour field was
/// pre-filled with `COLORS[0]`, so the exporter's auto-colour branch was dead
/// code: every series in the figure was the same blue.
#[test]
fn default_styles_cycle_the_palette_and_the_legend_matches_the_drawn_colour() {
    let mut fig = Figure::new("cycle")
        .size(800.0, 600.0)
        .grid(false)
        .legend(true);
    fig.add_series(&[0.0, 1.0], &[0.0, 1.0], "first");
    fig.add_series(&[0.0, 1.0], &[1.0, 0.0], "second");
    fig.scatter(&[0.5], &[0.5], "third");

    let svg = cv_plot::to_svg(&fig);
    let strokes = strokes_of(&svg, "<polyline");
    let circles = fills_of(&svg, "<circle");
    assert_eq!(
        strokes,
        vec![COLORS[0].to_string(), COLORS[1].to_string()],
        "two default line series must get two palette colours: {svg}"
    );
    assert_eq!(
        circles,
        vec![COLORS[2].to_string(); 1],
        "the scatter series must keep the colour its index earned"
    );

    // The legend rects carry the same colours, in the same order.
    let swatches: Vec<String> = tags_with(&svg, "<rect")
        .iter()
        .map(|t| attr(t, "fill"))
        .collect();
    assert_eq!(
        swatches,
        vec![
            COLORS[0].to_string(),
            COLORS[1].to_string(),
            COLORS[2].to_string()
        ],
        "legend swatches must match the series colours: {svg}"
    );
    assert_eq!(
        texts_of_class(&svg, "legend"),
        vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string()
        ],
        "labels in series order: {svg}"
    );
}

/// Control for the palette test: an explicit colour is used verbatim and is not
/// replaced by the cycle.
#[test]
fn an_explicit_style_colour_overrides_the_palette() {
    let mut fig = Figure::new("explicit")
        .size(800.0, 600.0)
        .grid(false)
        .legend(false);
    fig.subplots[0].series.push(Series {
        x: vec![0.0, 1.0],
        y: vec![0.0, 1.0],
        label: "pinned".into(),
        plot_type: PlotType::Scatter,
        style: Style::new("#ff0000"),
    });
    let svg = cv_plot::to_svg(&fig);
    assert_eq!(
        fills_of(&svg, "<circle"),
        vec!["#ff0000".to_string(); 2],
        "an explicit colour must survive: {svg}"
    );
}

// ----------------------------------------------------------- non-finite data

/// A non-finite point must be skipped, and the line must break there rather
/// than join across the gap or emit `NaN` as a coordinate.
#[test]
fn non_finite_points_are_skipped_and_break_the_line() {
    let mut fig = Figure::new("gap")
        .size(800.0, 600.0)
        .grid(false)
        .legend(false);
    fig.add_series(
        &[1.0, 2.0, f64::NAN, 4.0, 5.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        "gap",
    );

    let svg = cv_plot::to_svg(&fig);
    assert!(
        !svg.contains("NaN") && !svg.contains("inf"),
        "non-finite data leaked into the SVG: {svg}"
    );
    let polylines = tags_with(&svg, "<polyline");
    assert_eq!(
        polylines.len(),
        2,
        "the line must break at the missing point, not join across it: {svg}"
    );
    assert_eq!(points(polylines[0]).len(), 2, "run before the gap");
    assert_eq!(points(polylines[1]).len(), 2, "run after the gap");

    // Control: the same series without the NaN is one polyline of five points.
    let mut whole = Figure::new("whole")
        .size(800.0, 600.0)
        .grid(false)
        .legend(false);
    whole.add_series(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        "whole",
    );
    let svg = cv_plot::to_svg(&whole);
    let polylines = tags_with(&svg, "<polyline");
    assert_eq!(polylines.len(), 1, "a finite series is one polyline: {svg}");
    assert_eq!(points(polylines[0]).len(), 5);
}

/// A NaN must not poison the axis range: the finite values still set it.
///
/// (`f64::min`/`f64::max` return the non-NaN operand, so this is a property of
/// the language, not of the crate - but the defect was that the NaN vertex was
/// still mapped and emitted as `NaN`.)
#[test]
fn a_leading_nan_does_not_shift_the_bounds() {
    let mut with_nan = Figure::new("nan")
        .size(800.0, 600.0)
        .grid(false)
        .legend(false);
    with_nan.scatter(&[f64::NAN, 1.0, 4.0], &[f64::NAN, 10.0, 40.0], "s");
    let mut clean = Figure::new("clean")
        .size(800.0, 600.0)
        .grid(false)
        .legend(false);
    clean.scatter(&[1.0, 4.0], &[10.0, 40.0], "s");

    let a = cv_plot::to_svg(&with_nan);
    let b = cv_plot::to_svg(&clean);
    assert_eq!(
        tags_with(&a, "<circle"),
        tags_with(&b, "<circle"),
        "the finite points must be placed exactly as if the NaN were not there"
    );
    assert!(
        !a.contains("NaN"),
        "the NaN point must be dropped, not drawn at a NaN coordinate: {a}"
    );
}

/// A figure whose only data is non-finite has no plottable point, so saving it
/// must fail rather than write a file full of `NaN` coordinates.
#[test]
fn an_all_nan_figure_is_rejected_and_leaves_no_file() {
    let mut fig = Figure::new("all nan").grid(false).legend(false);
    fig.scatter(&[f64::NAN, f64::INFINITY], &[0.0, 1.0], "s");

    let path = unique_tmp("all_nan");
    assert!(
        matches!(
            cv_plot::save_svg(&fig, path.to_str().unwrap()),
            Err(PlotError::InvalidData(_))
        ),
        "an all-non-finite figure must be rejected"
    );
    assert!(
        !path.exists(),
        "a rejected save left a file at {}",
        path.display()
    );
    let svg = cv_plot::to_svg(&fig);
    assert!(
        !svg.contains("NaN"),
        "to_svg invented a NaN coordinate: {svg}"
    );
}

// --------------------------------------------------------------- figure size

/// A size that cannot be drawn must be reported, not silently saved.
///
/// `size(0.0, 0.0)` emitted `width="0"`, which the SVG spec defines as "do not
/// render", and a negative size emitted an invalid header; both saved `Ok`.
#[test]
fn non_positive_or_non_finite_figure_size_is_rejected() {
    for (w, h) in [
        (0.0, 0.0),
        (-800.0, 600.0),
        (800.0, -600.0),
        (f64::NAN, 600.0),
        (800.0, f64::INFINITY),
    ] {
        let mut fig = Figure::new("bad size").size(w, h).grid(false).legend(false);
        fig.scatter(&[0.0, 1.0], &[0.0, 1.0], "s");
        let path = unique_tmp("bad_size");
        assert!(
            matches!(
                cv_plot::save_svg(&fig, path.to_str().unwrap()),
                Err(PlotError::InvalidData(_))
            ),
            "size({w}, {h}) must be rejected"
        );
        assert!(
            !path.exists(),
            "size({w}, {h}) was rejected but still left a file"
        );
        assert!(
            matches!(
                cv_plot::save_html(&fig, path.to_str().unwrap()),
                Err(PlotError::InvalidData(_))
            ),
            "size({w}, {h}) must be rejected by save_html too"
        );
        assert!(!path.exists());
    }

    // Control: an ordinary size still saves.
    let mut ok = Figure::new("fine")
        .size(640.0, 480.0)
        .grid(false)
        .legend(false);
    ok.scatter(&[0.0, 1.0], &[0.0, 1.0], "s");
    let path = unique_tmp("good_size");
    cv_plot::save_svg(&ok, path.to_str().unwrap()).expect("a positive size must still save");
    assert!(path.exists());
    std::fs::remove_file(&path).expect("remove the file this test wrote");
}

// ---------------------------------------------------------------- ragged data

/// Series whose `x` and `y` differ in length must be reported.
///
/// Every drawing arm walks the pair with `zip`, so the extra values were
/// dropped without a word and the figure described less data than it held.
#[test]
fn a_ragged_series_is_rejected_before_anything_is_written() {
    let mut fig = Figure::new("ragged").grid(false).legend(false);
    fig.scatter(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0], "ragged");

    let path = unique_tmp("ragged");
    match cv_plot::save_svg(&fig, path.to_str().unwrap()) {
        Err(PlotError::InvalidData(msg)) => {
            assert!(
                msg.contains("4") && msg.contains("2"),
                "the error must name both lengths, got: {msg}"
            );
        }
        other => panic!("expected InvalidData for a ragged series, got {other:?}"),
    }
    assert!(!path.exists(), "a rejected ragged save left a file behind");

    // Control: equal lengths save, and every pair is drawn.
    let mut ok = Figure::new("ok").grid(false).legend(false);
    ok.scatter(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0], "ok");
    assert_eq!(tags_with(&cv_plot::to_svg(&ok), "<circle").len(), 4);
}

// ------------------------------------------------------ unrendered plot types

/// `Histogram` and `Heatmap` are declared plot types that no drawing arm
/// handles: the old renderer emitted an empty plot area and saved it `Ok`.
#[test]
fn unimplemented_plot_types_are_reported_instead_of_silently_dropped() {
    let series = |plot_type: PlotType, label: &str| Series {
        x: vec![1.0, 2.0, 3.0],
        y: vec![1.0, 2.0, 3.0],
        label: label.into(),
        plot_type,
        style: Style::default(),
    };

    for plot_type in [PlotType::Histogram, PlotType::Heatmap] {
        let mut fig = Figure::new("unimplemented").grid(false).legend(false);
        fig.subplots[0] = SubPlot {
            series: vec![series(plot_type.clone(), "unimplemented")],
            ..SubPlot::default()
        };

        let path = unique_tmp("unimplemented");
        match cv_plot::save_svg(&fig, path.to_str().unwrap()) {
            Err(PlotError::InvalidData(msg)) => assert!(
                msg.contains("unimplemented"),
                "the error must name the series, got: {msg}"
            ),
            other => panic!("expected InvalidData for a {plot_type:?} series, got {other:?}"),
        }
        assert!(!path.exists(), "a rejected save left a file behind");

        // `to_svg` cannot fail, so it says so in the output instead of drawing
        // an empty plot area that looks like zero-valued data.
        let svg = cv_plot::to_svg(&fig);
        assert!(
            svg.contains("is not rendered"),
            "the SVG must admit the series was not drawn: {svg}"
        );
        assert!(
            tags_with(&svg, "<polyline").is_empty(),
            "no geometry may be invented for an unhandled type: {svg}"
        );
    }

    // Control: the same data as a Line renders normally.
    let mut ok = Figure::new("line").grid(false).legend(false);
    ok.subplots[0] = SubPlot {
        series: vec![series(PlotType::Line, "line")],
        ..SubPlot::default()
    };
    let path = unique_tmp("line");
    cv_plot::save_svg(&ok, path.to_str().unwrap()).expect("a Line series must still save");
    assert!(tags_with(&cv_plot::to_svg(&ok), "<polyline").len() == 1);
    std::fs::remove_file(&path).expect("remove the file this test wrote");
}

// ------------------------------------------------------------------ styling

/// `Style::marker` must actually select the marker shape.
#[test]
fn marker_style_selects_the_marker_shape() {
    let draw = |style: Style| {
        let mut fig = Figure::new("m")
            .size(800.0, 600.0)
            .grid(false)
            .legend(false);
        fig.subplots[0].series.push(Series {
            x: vec![1.0, 2.0],
            y: vec![1.0, 2.0],
            label: "m".into(),
            plot_type: PlotType::Scatter,
            style,
        });
        cv_plot::to_svg(&fig)
    };

    let squares = draw(Style::default().marker("s"));
    assert_eq!(
        tags_with(&squares, "<rect").len(),
        2,
        "marker \"s\" must draw squares: {squares}"
    );
    assert!(tags_with(&squares, "<circle").is_empty());

    let triangles = draw(Style::default().marker("^"));
    assert_eq!(tags_with(&triangles, "<polygon").len(), 2, "marker \"^\"");
    let crosses = draw(Style::default().marker("x"));
    // Two crossing lines per point, plus the two axis lines of the panel.
    // Filter by the **tick class**, not by "has a black stroke". The old filter
    // was `!t.contains("stroke=\"black\"")`, which happened to exclude the panel's
    // two axis lines - and, when tick marks were added, did not exclude those,
    // because a tick `<line>` carries no `stroke` attribute at all. Adding ticks
    // therefore broke this test.
    //
    // Being explicit about what is being excluded is the fix: a marker line has a
    // stroke and no tick class; a tick has the class and no stroke; a panel axis line
    // has neither in the form matched here.
    let cross_lines = tags_with(&crosses, "<line")
        .into_iter()
        .filter(|t| !t.contains("class=\"tick"))
        .count();
    // Exclude both the panel's axis box (black stroke, no tick class) and the tick
    // marks (tick class, no stroke). The original filter excluded only the first,
    // which worked by accident until ticks existed.
    let cross_lines = tags_with(&crosses, "<line")
        .into_iter()
        .filter(|t| !t.contains("class=\"tick") && !t.contains("stroke=\"black\""))
        .count();
    assert_eq!(
        cross_lines, 4,
        "marker \"x\" is two lines per point: {crosses}"
    );

    // Control 1: the default marker is still a circle of marker_size/2.
    let circles = draw(Style::default());
    let cs = tags_with(&circles, "<circle");
    assert_eq!(cs.len(), 2, "the default marker is a circle: {circles}");
    assert_close(attr_f64(cs[0], "r"), 3.0, "default radius");

    // Control 2: an unknown marker falls back to the circle rather than
    // drawing nothing.
    let unknown = draw(Style::default().marker("!"));
    assert_eq!(
        tags_with(&unknown, "<circle").len(),
        2,
        "unknown marker fallback"
    );
}

/// `Style::alpha` must reach the output, and full opacity must not add an
/// attribute that was never asked for.
#[test]
fn alpha_below_one_is_applied_and_full_alpha_is_omitted() {
    let draw = |style: Style| {
        let mut fig = Figure::new("a")
            .size(800.0, 600.0)
            .grid(false)
            .legend(false);
        fig.subplots[0].series.push(Series {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            label: "a".into(),
            plot_type: PlotType::Line,
            style,
        });
        cv_plot::to_svg(&fig)
    };

    let faded = draw(Style::default().alpha(0.25));
    let line = tags_with(&faded, "<polyline")[0];
    assert_eq!(
        attr(line, "opacity"),
        "0.25",
        "line alpha must be emitted: {faded}"
    );

    // Control: the default style keeps the bare polyline.
    let solid = draw(Style::default());
    let line = tags_with(&solid, "<polyline")[0];
    assert!(
        !line.contains("opacity"),
        "full opacity must not add an attribute: {solid}"
    );
}

// -------------------------------------------------------------- panel titles

/// A panel title must be drawn - above the panel it names when the figure has
/// several panels, and in the title slot when a single panel has no figure
/// title of its own.
#[test]
fn panel_titles_are_drawn_for_every_panel() {
    let mut fig = Figure::new("fig title")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.subplot(1, 2, 0)
        .title("left")
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "a");
    fig.subplot(1, 2, 1)
        .title("right")
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "b");

    let svg = cv_plot::to_svg(&fig);
    let mut titles = texts_of_class(&svg, "panel-title");
    titles.sort();
    assert_eq!(
        titles,
        vec!["left".to_string(), "right".to_string()],
        "both panel titles must be drawn: {svg}"
    );
    assert_eq!(
        texts_of_class(&svg, "title"),
        vec!["fig title".to_string()],
        "the figure title goes above the grid: {svg}"
    );

    // A figure created without a title falls back to its single panel's title:
    // `Figure::title` used to write into a field no renderer read.
    let mut fallback = Figure::new("").size(800.0, 600.0).legend(false).grid(false);
    fallback
        .title("only panel")
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "s");
    let svg = cv_plot::to_svg(&fallback);
    assert!(
        svg.contains("only panel"),
        "the single panel's title must still reach the output: {svg}"
    );

    // Control: an untitled figure emits no title text at all.
    let mut quiet = Figure::new("").size(800.0, 600.0).legend(false).grid(false);
    quiet.add_series(&[0.0, 1.0], &[0.0, 1.0], "s");
    let svg = cv_plot::to_svg(&quiet);
    assert!(texts_of_class(&svg, "title").is_empty());
    assert!(texts_of_class(&svg, "panel-title").is_empty());
}

// ----------------------------------------------------------------- single panel

/// Control for the whole file: a one-panel figure must keep the original
/// full-figure axes geometry, and a panel with no data must not invent any.
#[test]
fn a_single_panel_keeps_the_full_figure_axes() {
    let mut fig = Figure::new("solo")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.scatter(&[0.0, 10.0], &[0.0, 100.0], "s");
    let svg = cv_plot::to_svg(&fig);
    let circles = tags_with(&svg, "<circle");
    assert_eq!(circles.len(), 2, "one circle per point: {svg}");
    // Same numbers as `svg_geometry.rs` computes for a single-plot figure:
    // the plot area is still 60..770 x 50..550.
    assert_close(
        attr_f64(circles[0], "cx"),
        60.0 + (1.0 / 12.0) * 710.0,
        "cx",
    );
    assert_close(
        attr_f64(circles[0], "cy"),
        550.0 - (10.0 / 120.0) * 500.0,
        "cy",
    );
    let axis = tags_with(&svg, "<line")
        .into_iter()
        .filter(|t| t.contains("stroke=\"black\""))
        .collect::<Vec<_>>();
    assert_eq!(axis.len(), 2, "one axes for one panel: {svg}");
}

/// An empty panel inside a populated figure must draw nothing - no invented
/// range, no zero-valued points.
#[test]
fn an_empty_panel_in_a_grid_draws_no_geometry_of_its_own() {
    let mut fig = Figure::new("mixed")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.subplot(1, 2, 0)
        .add_series(&[0.0, 1.0], &[0.0, 1.0], "a");
    fig.subplot(1, 2, 1); // selected but never given data

    let svg = cv_plot::to_svg(&fig);
    assert_eq!(
        tags_with(&svg, "<polyline").len(),
        1,
        "only the populated panel draws geometry: {svg}"
    );
    assert!(!svg.contains("NaN"), "empty panel leaked NaN: {svg}");

    // The figure as a whole is still savable: one panel has data.
    let path = unique_tmp("mixed");
    cv_plot::save_svg(&fig, path.to_str().unwrap())
        .expect("a figure with one populated panel must save");
    std::fs::remove_file(&path).expect("remove the file this test wrote");
}
