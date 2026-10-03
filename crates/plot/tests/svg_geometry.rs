//! Hand-verifiable geometry checks on the SVG that `cv_plot` emits.
//!
//! The plot area is inset by fixed margins (`left = 60`, `right = 30`,
//! `top = 50`, `bottom = 50` for the defaults), and data are mapped onto it as
//!
//! ```text
//! px = margin_left  + (x - min_x) / (max_x - min_x) * plot_width
//! py = height - margin_bottom - (y - min_y) / (max_y - min_y) * plot_height
//! ```
//!
//! with the data bounds padded by `10%` of `max(range, 1)`. Every expected
//! number below is computed by hand from those rules, so a wrong answer means
//! the projection itself is wrong (flipped axis, wrong margin, padding applied
//! to the wrong range, off-by-one in the padding), not just a cosmetic
//! difference in the emitted markup.

use cv_plot::{Figure, PlotError};

/// The margins `to_svg` reserves, and the resulting plot area, for an 800x600
/// figure.
const ML: f64 = 60.0;
const MR: f64 = 30.0;
const MT: f64 = 50.0;
const MB: f64 = 50.0;

/// A `(open_tag, offset_just_past_the_closing_angle_bracket)` pair for every
/// occurrence of `marker` in `svg`.
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

/// Extract every opening tag in `svg` that starts with `marker`.
fn tags_with<'a>(svg: &'a str, marker: &str) -> Vec<&'a str> {
    tag_spans(svg, marker).into_iter().map(|(t, _)| t).collect()
}

/// Read a numeric attribute out of a tag, e.g. `attr_f64(tag, "cx")`.
fn attr_f64(tag: &str, name: &str) -> f64 {
    let key = format!("{}=\"", name);
    let start = tag
        .find(&key)
        .unwrap_or_else(|| panic!("attribute `{name}` missing from tag: {tag}"))
        + key.len();
    let end = start + tag[start..].find('"').expect("unterminated attribute");
    tag[start..end]
        .parse()
        .unwrap_or_else(|_| panic!("attribute `{name}` is not numeric in tag: {tag}"))
}

/// The text content of every element carrying `class="<class>"`.
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

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 0.05,
        "{what}: expected {expected}, got {actual}"
    );
}

/// A scatter plot over a known range must place its markers at the exact
/// screen coordinates the projection rule predicts.
///
/// Wrong answer: the first marker not at x = 60 + (1/12) * 710, or the two
/// markers swapped (x increasing rightwards but y increasing downwards), or
/// both markers collapsed onto the same point, or the padding applied to the
/// smaller of the two ranges.
#[test]
fn scatter_maps_data_extrema_onto_the_padded_axes() {
    let mut fig = Figure::new("extremes")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.scatter(&[0.0, 10.0], &[0.0, 100.0], "s");

    let svg = fig.to_svg_for_test();
    let circles = tags_with(&svg, "<circle");
    assert_eq!(circles.len(), 2, "one circle per scatter point: {svg}");

    let plot_w = 800.0 - ML - MR;
    let plot_h = 600.0 - MT - MB;

    // x: data span 0..10 -> padded span -1..11 (width 12).
    assert_close(
        attr_f64(circles[0], "cx"),
        ML + 1.0 / 12.0 * plot_w,
        "cx of x=0",
    );
    assert_close(
        attr_f64(circles[1], "cx"),
        ML + 11.0 / 12.0 * plot_w,
        "cx of x=10",
    );
    // y: data span 0..100 -> padded span -10..110 (height 120), inverted.
    assert_close(
        attr_f64(circles[0], "cy"),
        600.0 - MB - 10.0 / 120.0 * plot_h,
        "cy of y=0 (must be at the bottom)",
    );
    assert_close(
        attr_f64(circles[1], "cy"),
        600.0 - MB - 110.0 / 120.0 * plot_h,
        "cy of y=100 (must be at the top)",
    );

    // The marker radius is the style's marker_size halved.
    assert_close(attr_f64(circles[0], "r"), 3.0, "default marker radius");
}

/// A single data point is the degenerate case of the padding rule: the span is
/// 0, so `max(range, 1) * 0.1` makes it 0.1 and the point lands exactly at the
/// centre of the plot area.
///
/// Wrong answer: `NaN`/`inf` coordinates (if the degenerate bounds were not
/// padded, `0/0` reaches the format string), or a marker anywhere but the
/// centre because the padding used the raw span of 0.
#[test]
fn single_point_lands_at_the_exact_centre_of_the_plot_area() {
    let mut fig = Figure::new("one")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.scatter(&[5.0], &[7.0], "solo");

    let svg = fig.to_svg_for_test();
    let circles = tags_with(&svg, "<circle");
    assert_eq!(circles.len(), 1, "expected exactly one circle: {svg}");
    assert_close(attr_f64(circles[0], "cx"), 415.0, "centre x");
    assert_close(attr_f64(circles[0], "cy"), 300.0, "centre y");
    assert!(
        !svg.contains("NaN") && !svg.contains("inf"),
        "degenerate bounds leaked non-finite coordinates into the SVG: {svg}"
    );
}

/// A one-element line series cannot describe a segment, so it must draw no
/// polyline - but the surrounding SVG must still be well formed.
///
/// Wrong answer: a degenerate single-vertex `<polyline>` (a stray dot in the
/// middle of the plot), a panic on the `len() > 1` guard, or truncated markup.
#[test]
fn single_point_line_series_emits_no_polyline_but_stays_well_formed() {
    let mut fig = Figure::new("degenerate")
        .size(400.0, 300.0)
        .legend(false)
        .grid(false);
    fig.add_series(&[3.0], &[4.0], "lone");

    let svg = fig.to_svg_for_test();
    assert!(
        tags_with(&svg, "<polyline").is_empty(),
        "a 1-point line must not emit a polyline: {svg}"
    );
    assert!(svg.starts_with("<svg "), "header missing: {svg}");
    assert!(svg.ends_with("</svg>"), "footer missing: {svg}");
    assert!(!svg.contains("NaN"), "non-finite coordinate in SVG: {svg}");
}

/// Grid rendering is fully determined by the flag: 6 vertical + 6 horizontal
/// lines, the outermost ones exactly on the plot-area edges.
///
/// Wrong answer: 5 or 10 lines (off-by-one in the loop bounds), grid lines
/// that stop short of the plot area (wrong plot width), or the grid drawn even
/// though `grid(false)` was set.
#[test]
fn grid_toggle_controls_exactly_twelve_gridlines() {
    let mut on = Figure::new("g").size(800.0, 600.0).legend(false).grid(true);
    on.scatter(&[0.0, 1.0], &[0.0, 1.0], "s");

    let svg = on.to_svg_for_test();
    let grid: Vec<&str> = tags_with(&svg, "<line")
        .into_iter()
        .filter(|t| t.contains("class=\"grid\""))
        .collect();
    assert_eq!(grid.len(), 12, "expected 6 vertical + 6 horizontal: {svg}");

    let plot_w = 800.0 - ML - MR;
    let plot_h = 600.0 - MT - MB;
    let vertical: Vec<f64> = grid
        .iter()
        .copied()
        .filter(|t| attr_f64(t, "x1") == attr_f64(t, "x2"))
        .map(|t| attr_f64(t, "x1"))
        .collect();
    let horizontal: Vec<f64> = grid
        .iter()
        .copied()
        .filter(|t| attr_f64(t, "y1") == attr_f64(t, "y2"))
        .map(|t| attr_f64(t, "y1"))
        .collect();

    assert_eq!(vertical.len(), 6, "vertical grid lines: {vertical:?}");
    assert_eq!(horizontal.len(), 6, "horizontal grid lines: {horizontal:?}");
    for (i, x) in vertical.iter().enumerate() {
        assert_close(*x, ML + i as f64 / 5.0 * plot_w, "vertical grid x");
    }
    assert_close(
        *vertical.last().unwrap(),
        800.0 - MR,
        "last vertical grid line must sit on the right edge of the plot area",
    );
    for (i, y) in horizontal.iter().enumerate() {
        assert_close(*y, MT + i as f64 / 5.0 * plot_h, "horizontal grid y");
    }

    let mut off = Figure::new("g")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    off.scatter(&[0.0, 1.0], &[0.0, 1.0], "s");
    let svg = off.to_svg_for_test();
    assert!(
        !svg.contains("class=\"grid\""),
        "grid(false) still emitted grid lines: {svg}"
    );
}

/// The legend must contain exactly one entry per series, in order, and must
/// disappear entirely when disabled.
///
/// Wrong answer: one entry for two series (the colour/index walk skipping a
/// series), a legend drawn despite `legend(false)`, or a label emitted for a
/// subplot that has no series.
#[test]
fn legend_has_one_entry_per_series_and_honours_the_toggle() {
    let mut fig = Figure::new("l").size(800.0, 600.0).legend(true).grid(false);
    fig.add_series(&[0.0, 1.0], &[0.0, 1.0], "first");
    fig.scatter(&[0.0, 1.0], &[1.0, 2.0], "second");

    let svg = fig.to_svg_for_test();
    assert_eq!(
        texts_of_class(&svg, "legend"),
        vec!["first".to_string(), "second".to_string()],
        "legend entries must match the series in order: {svg}"
    );

    let mut quiet = Figure::new("l")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    quiet.add_series(&[0.0, 1.0], &[0.0, 1.0], "first");
    let svg = quiet.to_svg_for_test();
    assert!(
        texts_of_class(&svg, "legend").is_empty(),
        "legend(false) still emitted legend text: {svg}"
    );
}

/// Every bar must produce one rect, `plot_width / n * 0.8` wide, with a height
/// proportional to the value above the padded minimum.
///
/// Wrong answer: fewer rects than bars (a zip truncation), a zero or NaN width
/// (division by a zero-length series), a height measured from the top instead
/// of the padded minimum, or bars whose x does not increase with the data.
#[test]
fn bar_series_draws_one_hand_sized_rect_per_bar() {
    let mut fig = Figure::new("b")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.bar(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0], "bars");

    let svg = fig.to_svg_for_test();
    let rects = tags_with(&svg, "<rect");
    assert_eq!(rects.len(), 4, "one rect per bar: {svg}");

    let plot_w = 800.0 - ML - MR;
    let plot_h = 600.0 - MT - MB;
    assert_close(attr_f64(rects[0], "width"), plot_w / 4.0 * 0.8, "bar width");
    for r in &rects {
        assert_close(attr_f64(r, "width"), plot_w / 4.0 * 0.8, "bar width");
    }

    // y data span 1..4 -> padded 0.7..4.3 (3.6).
    assert_close(
        attr_f64(rects[0], "height"),
        (1.0 - 0.7) / 3.6 * plot_h,
        "height of the shortest bar",
    );
    assert_close(
        attr_f64(rects[3], "height"),
        (4.0 - 0.7) / 3.6 * plot_h,
        "height of the tallest bar",
    );

    let xs: Vec<f64> = rects.iter().map(|r| attr_f64(r, "x")).collect();
    assert!(
        xs.windows(2).all(|w| w[1] > w[0]),
        "bar x positions must increase with the data: {xs:?}"
    );
}

/// A figure with no plottable points must be rejected by `save_svg` without
/// leaving a file behind, while `to_svg` still produces well-formed markup.
///
/// Wrong answer: a file written containing `NaN`/garbage geometry, or `Ok(())`
/// returned for a figure that cannot be drawn.
#[test]
fn empty_figure_is_rejected_by_save_svg_without_creating_a_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("empty.svg");
    let fig = Figure::new("nothing here").legend(false).grid(false);

    match cv_plot::save_svg(&fig, path.to_str().unwrap()) {
        Err(PlotError::InvalidData(_)) => {}
        Err(other) => panic!("expected PlotError::InvalidData, got {other:?}"),
        Ok(()) => panic!("save_svg accepted a figure with no plottable points"),
    }
    assert!(
        !path.exists(),
        "a rejected save must not leave a file behind at {}",
        path.display()
    );

    let svg = cv_plot::to_svg(&fig);
    assert!(svg.starts_with("<svg ") && svg.ends_with("</svg>"));
    assert!(
        tags_with(&svg, "<circle").is_empty() && tags_with(&svg, "<polyline").is_empty(),
        "an empty figure must not invent geometry: {svg}"
    );
}

/// `save_svg` / `save_html` must write exactly the bytes the in-memory
/// renderers produce, and the file APIs must report IO failures instead of
/// silently succeeding.
///
/// Wrong answer: a truncated or re-encoded file, a `save_html` whose body
/// differs from `to_html`, or a missing parent directory reported as success.
#[test]
fn save_writes_exactly_the_rendered_bytes_and_reports_io_errors() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut fig = Figure::new("round trip").size(640.0, 480.0).grid(false);
    fig.scatter(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], "s");

    let svg_path = dir.path().join("p.svg");
    let html_path = dir.path().join("p.html");
    cv_plot::save_svg(&fig, svg_path.to_str().unwrap()).expect("save_svg");
    cv_plot::save_html(&fig, html_path.to_str().unwrap()).expect("save_html");

    assert_eq!(
        std::fs::read_to_string(&svg_path).unwrap(),
        cv_plot::to_svg(&fig),
        "save_svg wrote something other than to_svg()"
    );
    assert_eq!(
        std::fs::read_to_string(&html_path).unwrap(),
        cv_plot::to_html(&fig),
        "save_html wrote something other than to_html()"
    );

    // A missing parent directory must surface as an IO error.
    let missing = dir.path().join("nope").join("p.svg");
    assert!(
        cv_plot::save_svg(&fig, missing.to_str().unwrap()).is_err(),
        "saving into a missing directory reported success"
    );

    // PNG export is documented as unsupported; it must not create a file.
    let png = dir.path().join("p.png");
    assert!(
        matches!(
            cv_plot::save_png(&fig, png.to_str().unwrap()),
            Err(PlotError::Export(_))
        ),
        "save_png must report Export(..), not success or another variant"
    );
    assert!(
        !png.exists(),
        "save_png created a file while reporting failure"
    );
}

/// The figure size must reach the SVG header, every panel must draw its own
/// axis labels, and the legend must cover every panel.
///
/// Wrong answer: the header still saying 800x600 after `size(321, 123)` (the
/// title builder shadowing the size), only the first subplot's labels
/// appearing (every later panel's labels silently dropped), or a legend that
/// misses a panel.
#[test]
fn size_reaches_the_header_and_every_panel_gets_its_own_labels() {
    let mut fig = Figure::new("layout")
        .size(321.0, 123.0)
        .legend(true)
        .grid(false);
    // `Figure::new` already ships one default subplot; overwrite it so the
    // first subplot is the one carrying the labels under test.
    fig.subplots[0] = cv_plot::SubPlot {
        series: vec![cv_plot::Series::new(vec![0.0, 1.0], vec![0.0, 1.0], "sub0")],
        title: String::new(),
        x_label: "alpha".to_string(),
        y_label: "gamma".to_string(),
    };
    fig.subplots.push(cv_plot::SubPlot {
        series: vec![cv_plot::Series::new(vec![0.0, 1.0], vec![1.0, 2.0], "sub1")],
        title: String::new(),
        x_label: "beta".to_string(),
        y_label: "delta".to_string(),
    });

    let svg = fig.to_svg_for_test();
    assert!(
        svg.contains(r#"<svg xmlns="http://www.w3.org/2000/svg" width="321" height="123">"#),
        "figure size missing from the SVG header: {svg}"
    );
    assert_eq!(
        texts_of_class(&svg, "label"),
        vec![
            "alpha".to_string(),
            "gamma".to_string(),
            "beta".to_string(),
            "delta".to_string()
        ],
        "each panel must label its own axes: {svg}"
    );
    assert_eq!(
        texts_of_class(&svg, "legend"),
        vec!["sub0".to_string(), "sub1".to_string()],
        "the legend must cover every subplot: {svg}"
    );

    // `subplot(rows, cols, index)` selects the panel: `index` is 0-based and
    // row-major, so `subplot(2, 2, 0)` makes the FIRST panel current.
    let mut grown = Figure::new("grow");
    grown.subplot(2, 2, 0);
    assert_eq!(
        grown.subplots.len(),
        4,
        "subplot(2, 2) must yield 4 subplots"
    );
    grown.add_series(&[0.0], &[0.0], "added-later");
    assert!(
        grown.subplots[0]
            .series
            .iter()
            .any(|s| s.label == "added-later"),
        "add_series must write into the panel subplot() selected"
    );
    assert!(
        grown.subplots[3].series.is_empty(),
        "add_series must not fall through to the last panel"
    );

    // Control: selecting a later panel still routes there.
    grown
        .subplot(2, 2, 3)
        .add_series(&[1.0], &[1.0], "last-panel");
    assert!(
        grown.subplots[3]
            .series
            .iter()
            .any(|s| s.label == "last-panel"),
        "subplot(2, 2, 3) must make the last panel current"
    );
    assert_eq!(
        grown.subplots[0].series.len(),
        1,
        "the panel selected earlier must not receive later series"
    );
}

/// Local helper so the tests above can render through the same public entry
/// point users call.
trait RenderForTest {
    fn to_svg_for_test(&self) -> String;
}

impl RenderForTest for Figure {
    fn to_svg_for_test(&self) -> String {
        cv_plot::to_svg(self)
    }
}
