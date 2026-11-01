//! Regression: an all-zero bar series collapsed its own y-range.
//!
//! `Bounds::of` makes `y = 0` a sticky autoscale edge for bars, so that bars
//! stand on the axis. When every bar value *is* zero, both the minimum and the
//! maximum sticky edge are `0.0`, the padding is applied past neither of them,
//! and the panel's y-range becomes exactly `[0, 0]`. `Bounds::project` then
//! divides by that zero range, so every bar was emitted with `NaN` for its
//! `y` and `height` - and `save_svg` accepted it, because `ensure_plottable`
//! only checks that the input pairs are finite, not that the output is.

use cv_plot::{Figure, PlotError};

fn tags_with<'a>(svg: &'a str, marker: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = svg[i..].find(marker) {
        let start = i + rel;
        let gt = start + svg[start..].find('>').expect("unterminated tag");
        out.push(&svg[start..=gt]);
        i = gt + 1;
    }
    out
}

fn attr_f64(tag: &str, name: &str) -> f64 {
    let key = format!("{name}=\"");
    let start = tag
        .find(&key)
        .unwrap_or_else(|| panic!("attribute `{name}` missing from tag: {tag}"))
        + key.len();
    let end = start + tag[start..].find('"').expect("unterminated attribute");
    tag[start..end]
        .parse()
        .unwrap_or_else(|_| panic!("attribute `{name}` is not numeric in tag: {tag}"))
}

/// A bar chart whose values are all zero is ordinary data (a day with no
/// events, a series of empty buckets). It must draw finite rectangles.
#[test]
fn all_zero_bars_draw_finite_rects() {
    let mut fig = Figure::new("zeros")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.bar(&[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0], "bars");

    let svg = cv_plot::to_svg(&fig);
    assert!(
        !svg.contains("NaN") && !svg.contains("inf"),
        "an all-zero bar series leaked a non-finite coordinate:\n{svg}"
    );

    let rects = tags_with(&svg, "<rect");
    assert_eq!(rects.len(), 3, "one rect per bar:\n{svg}");
    for rect in &rects {
        for name in ["x", "y", "width", "height"] {
            let v = attr_f64(rect, name);
            assert!(v.is_finite(), "bar `{name}` is not finite: {rect}");
        }
    }
}

/// The save path must not write an SVG full of `NaN`s either.
#[test]
fn all_zero_bars_save_a_finite_svg() {
    let mut fig = Figure::new("zeros")
        .size(800.0, 600.0)
        .legend(false)
        .grid(false);
    fig.bar(&[0.0], &[0.0], "bars");

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("zeros.svg");
    match cv_plot::save_svg(&fig, path.to_str().unwrap()) {
        Ok(()) => {
            let body = std::fs::read_to_string(&path).expect("saved SVG");
            assert!(
                !body.contains("NaN"),
                "save_svg wrote NaN coordinates:\n{body}"
            );
        }
        Err(PlotError::InvalidData(_)) => {
            // Rejecting is also acceptable as long as no file was written.
            assert!(!path.exists(), "a rejected save left a file behind");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}
