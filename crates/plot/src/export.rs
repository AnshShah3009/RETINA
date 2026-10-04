//! Export plots to various formats
//!
//! The renderer draws *panels*: one subplot fills the figure, and a figure
//! whose `Figure::subplot` grid holds several subplots lays them out in that
//! grid, each with its own data range, grid, axis box and labels. Every panel
//! is scaled to its own data, so a panel holding values in `0..1` next to one
//! holding `0..2000` fills its own cell in both cases.

use crate::chart::{Figure, PlotType, Series, SubPlot};
use crate::style::COLORS;
use std::fs::File;
use std::io::Write;

/// Outer margin of the figure, in pixels.
const MARGIN_LEFT: f64 = 60.0;
const MARGIN_RIGHT: f64 = 30.0;
const MARGIN_TOP: f64 = 50.0;
const MARGIN_BOTTOM: f64 = 50.0;

/// Insets inside one cell of a multi-panel grid, leaving room for that panel's
/// own title and axis labels.
const CELL_LEFT: f64 = 45.0;
const CELL_RIGHT: f64 = 20.0;
const CELL_TOP: f64 = 28.0;
const CELL_BOTTOM: f64 = 38.0;

/// A screen-space rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Rect {
    fn right(&self) -> f64 {
        self.x + self.w
    }

    fn bottom(&self) -> f64 {
        self.y + self.h
    }

    fn center_x(&self) -> f64 {
        self.x + self.w / 2.0
    }

    fn center_y(&self) -> f64 {
        self.y + self.h / 2.0
    }
}

/// Padded data range of one panel.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Bounds {
    /// Data range of `subplot`, padded by 10% of `max(span, 1)`.
    ///
    /// `None` when the panel holds no finite `(x, y)` pair at all, which is the
    /// only case where there is no honest answer to "what range is this axis".
    /// Non-finite pairs are skipped rather than folded in: `f64::min`/`f64::max`
    /// return the non-NaN operand, so a NaN mid-series does not poison the
    /// range, but it would still be mapped to a `NaN` screen coordinate.
    fn of(subplot: &SubPlot) -> Option<Self> {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let mut any = false;

        for series in &subplot.series {
            for (x, y) in series.x.iter().zip(series.y.iter()) {
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                any = true;
                min_x = min_x.min(*x);
                max_x = max_x.max(*x);
                min_y = min_y.min(*y);
                max_y = max_y.max(*y);
            }
        }
        if !any {
            return None;
        }

        // The padded range can never be zero: a single point or a constant
        // series gets a span of 0.2 instead of a division by zero.
        let x_pad = (max_x - min_x).max(1.0) * 0.1;
        let y_pad = (max_y - min_y).max(1.0) * 0.1;
        Some(Self {
            min_x: min_x - x_pad,
            max_x: max_x + x_pad,
            min_y: min_y - y_pad,
            max_y: max_y + y_pad,
        })
    }

    /// Map a data point onto the panel.
    fn project(&self, rect: &Rect, x: f64, y: f64) -> (f64, f64) {
        let px = rect.x + (x - self.min_x) / (self.max_x - self.min_x) * rect.w;
        let py = rect.y + rect.h - (y - self.min_y) / (self.max_y - self.min_y) * rect.h;
        (px, py)
    }
}

/// Screen-space rectangle of every panel, in `figure.subplots` order.
///
/// A single-subplot figure keeps the original full-figure axes geometry, which
/// is what every hand-checked geometry test in `tests/` is written against. A
/// multi-subplot figure is split into the grid shape `subplot()` recorded (or a
/// single row when the panels were pushed by hand), and each panel is inset
/// inside its cell.
fn panel_rects(figure: &Figure) -> Vec<Rect> {
    let n = figure.subplots.len().max(1);
    let interior = Rect {
        x: MARGIN_LEFT,
        y: MARGIN_TOP,
        w: figure.width - MARGIN_LEFT - MARGIN_RIGHT,
        h: figure.height - MARGIN_TOP - MARGIN_BOTTOM,
    };
    if n == 1 {
        return vec![interior];
    }

    let (rows, cols) = match figure.subplot_grid {
        Some((r, c)) if r >= 1 && c >= 1 && r.saturating_mul(c) >= n => (r, c),
        _ => (1, n),
    };
    let cell_w = interior.w / cols as f64;
    let cell_h = interior.h / rows as f64;

    // The insets are capped at a fraction of the cell so that a figure too
    // small for its own grid (a 123 px tall figure with two panels, say) cannot
    // produce a negative - i.e. vertically mirrored - panel. The caps do not
    // bind for a normally sized cell.
    let inset_l = CELL_LEFT.min(cell_w * 0.25);
    let inset_r = CELL_RIGHT.min(cell_w * 0.10);
    let inset_t = CELL_TOP.min(cell_h * 0.20);
    let inset_b = CELL_BOTTOM.min(cell_h * 0.25);

    (0..n)
        .map(|i| {
            let (row, col) = (i / cols, i % cols);
            Rect {
                x: interior.x + col as f64 * cell_w + inset_l,
                y: interior.y + row as f64 * cell_h + inset_t,
                w: cell_w - inset_l - inset_r,
                h: cell_h - inset_t - inset_b,
            }
        })
        .collect()
}

/// The colour a series is drawn in: its own style, or the next palette entry.
///
/// The legend walks the same function, so a swatch can never disagree with the
/// series it names.
fn series_color(series: &Series, panel: usize, index: usize, per_panel: usize) -> &str {
    if series.style.color.is_empty() {
        COLORS[(panel * per_panel + index) % COLORS.len()]
    } else {
        &series.style.color
    }
}

/// Maximal runs of finite `(x, y)` pairs in a series.
///
/// A gap is a genuine gap: the line renderer breaks the polyline there instead
/// of joining two points across a missing value, and nothing is ever mapped to
/// a `NaN` screen coordinate.
fn finite_runs(series: &Series) -> Vec<Vec<(f64, f64)>> {
    let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut run: Vec<(f64, f64)> = Vec::new();
    for (x, y) in series.x.iter().zip(series.y.iter()) {
        if x.is_finite() && y.is_finite() {
            run.push((*x, *y));
        } else if !run.is_empty() {
            runs.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
}

/// Opacity attribute for a style, omitted at full opacity so the common case
/// stays the bare element.
fn opacity_attr(alpha: f64) -> String {
    if alpha >= 1.0 {
        String::new()
    } else {
        format!(r#" opacity="{}""#, alpha.max(0.0))
    }
}

/// One marker of a scatter series.
///
/// `Style::marker` selects the shape: `o` (the default) draws a circle, `s` a
/// square, `^`/`v` a triangle, `+`/`x` a cross. Any other value falls back to
/// the circle, which is the documented default.
fn push_marker(
    svg: &mut String,
    marker: &str,
    px: f64,
    py: f64,
    size: f64,
    color: &str,
    alpha: f64,
) {
    let half = size / 2.0;
    let opacity = opacity_attr(alpha);
    match marker {
        "s" => svg.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}"{}/>
"#,
            px - half,
            py - half,
            size,
            size,
            color,
            opacity
        )),
        "^" => svg.push_str(&format!(
            r#"  <polygon points="{},{} {},{} {},{}" fill="{}"{}/>
"#,
            px,
            py - half,
            px - half,
            py + half,
            px + half,
            py + half,
            color,
            opacity
        )),
        "v" => svg.push_str(&format!(
            r#"  <polygon points="{},{} {},{} {},{}" fill="{}"{}/>
"#,
            px - half,
            py - half,
            px + half,
            py - half,
            px,
            py + half,
            color,
            opacity
        )),
        "+" => svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}"{}/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}"{}/>
"#,
            px - half,
            py,
            px + half,
            py,
            color,
            opacity,
            px,
            py - half,
            px,
            py + half,
            color,
            opacity
        )),
        "x" => svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}"{}/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}"{}/>
"#,
            px - half,
            py - half,
            px + half,
            py + half,
            color,
            opacity,
            px - half,
            py + half,
            px + half,
            py - half,
            color,
            opacity
        )),
        _ => svg.push_str(&format!(
            r#"  <circle cx="{}" cy="{}" r="{}" fill="{}"{}/>
"#,
            px, py, half, color, opacity
        )),
    }
}

/// Convert plot to SVG string
/// Tick positions on `[lo, hi]`, on a 1/2/5 x 10^k "nice" grid.
///
/// This is the algorithm behind Matplotlib's default `MaxNLocator`, and it is the
/// reference for tick placement in this crate - an axis with no numbers on it is not
/// a plotting axis.
///
/// The two details that are easy to get wrong:
///
/// - The step is a **nice** number, not the raw `span / count`. For a span of 100
///   and 5 ticks the raw step is 20 (fine), but for a span of 1 it is 0.2, which is
///   *not* of the form `m x 10^k` and would print as `0.2000000000000001` at some
///   widths. Snapping to `m in {1, 2, 5, 10}` avoids it, and keeps every tick
///   exactly representable - which is why `2.5` is left out of the step set.
/// - Ticks are **snapped outward from `lo`**, so the first is `lo` rounded down to
///   the grid and the last is `hi` rounded up. A tick outside `[lo, hi]` is a label
///   pointing at nothing, and the caller expands the axis to cover the grid.
///
/// A degenerate range (`hi == lo`, or one point) cannot be divided, so it is
/// widened by `1.0` first. That mirrors the `max(span, 1.0)` padding the axis
/// bounds already use.
fn nice_ticks(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    // Matplotlib's default step set. `2.5` matters: for a raw step of 12 over a
    // 60-wide range, `[1,2,5,10]` snaps to 20 where Matplotlib picks 15.
    const STEPS: [f64; 5] = [1.0, 2.0, 2.5, 5.0, 10.0];
    let count = count.max(2);
    if !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    let (mut lo, mut hi) = (lo, hi);
    if hi < lo {
        std::mem::swap(&mut lo, &mut hi);
    }
    let span = hi - lo;
    if !(span > 0.0) {
        // One point, or a constant series: widen so the axis is still readable.
        lo -= 0.5;
        hi += 0.5;
    }
    let span = hi - lo;
    // Matplotlib's `MaxNLocator` divides the span by the **bin** count, not by
    // `count - 1`. My first version used `count - 1`, which for a 120-wide range asks
    // for a step of 30 and yields three ticks where Matplotlib yields seven.
    let raw = span / count as f64;
    if !(raw > 0.0) || !raw.is_finite() {
        return Vec::new();
    }
    let exp = raw.log10().floor();
    let base = 10f64.powf(exp);
    // The **ceil** of `raw` over the nice set, not the floor. Verified against
    // `MaxNLocator(nbins=5)` for four ranges:
    //
    //   [-10, 110] raw 24  -> 25    [0, 100] raw 20 -> 20
    //   [ -5,  55] raw 12  -> 15    [0, 1]   raw 0.2 -> 0.2
    //
    // Taking the floor instead gives 20, 20, 10, 0.2 - wrong on two of the four.
    let step = STEPS
        .iter()
        .map(|s| s * base)
        .find(|s| *s >= raw - 1e-12 * raw)
        .unwrap_or(10.0 * base);

    let mut out = Vec::new();
    let first = (lo / step).floor() * step;
    // A guard rather than a `while` on a possibly non-advancing step.
    let max_ticks = 512;
    let mut v = first;
    while v <= hi + step * 1e-9 && out.len() < max_ticks {
        // Re-round to kill the accumulated error of repeated addition, so a tick
        // lands exactly on the grid instead of 1e-13 off it.
        let snapped = (v / step).round() * step;
        if snapped >= lo - step * 1e-9 && snapped <= hi + step * 1e-9 {
            out.push(snapped);
        }
        v += step;
        if !v.is_finite() {
            break;
        }
    }
    out
}

/// Render the tick labels and marks for one axis of a panel.
fn render_ticks(svg: &mut String, rect: &Rect, bounds: &Bounds, x_label: &str, y_label: &str) {
    const TICKS: usize = 5;
    // x axis, along the bottom
    for v in nice_ticks(bounds.min_x, bounds.max_x, TICKS) {
        let px = rect.x + (v - bounds.min_x) / (bounds.max_x - bounds.min_x) * rect.w;
        if px < rect.x - 1.0 || px > rect.x + rect.w + 1.0 {
            continue;
        }
        svg.push_str(&format!(
            r#"  <line x1="{px:.2}" y1="{}" x2="{px:.2}" y2="{}" class="tick tick-x"/>
"#,
            rect.bottom(),
            rect.bottom() + 5.0
        ));
        svg.push_str(&format!(
            r#"  <text x="{px:.2}" y="{}" class="tick tick-x" text-anchor="middle">{}</text>
"#,
            rect.bottom() + 17.0,
            format_tick_value(v)
        ));
    }
    // y axis, up the left side
    for v in nice_ticks(bounds.min_y, bounds.max_y, TICKS) {
        let py = rect.y + rect.h - (v - bounds.min_y) / (bounds.max_y - bounds.min_y) * rect.h;
        if py < rect.y - 1.0 || py > rect.y + rect.h + 1.0 {
            continue;
        }
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{py:.2}" x2="{}" y2="{py:.2}" class="tick tick-y"/>
"#,
            rect.x - 5.0,
            rect.x
        ));
        svg.push_str(&format!(
            r#"  <text x="{}" y="{py:.2}" class="tick tick-y" text-anchor="end">{}</text>
"#,
            rect.x - 8.0,
            format_tick_value(v)
        ));
    }
    let _ = (x_label, y_label);
}

/// Format a tick compactly, the way Matplotlib's `ScalarFormatter` does.
///
/// The default `{:.6}` would print `0.2000000000000001` for a value that is exactly
/// 0.2 in the grid, which is both ugly and a symptom of the step not being a nice
/// number in the first place.
fn format_tick_value(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        return "0".to_string();
    }
    if a >= 1e5 || a < 1e-4 {
        let s = format!("{:.3e}", v);
        // Trim a trailing zero in the exponent: 1.200e4 -> 1.2e4
        return s
            .replace('e', "e")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
    }
    let decimals = if a >= 100.0 {
        0
    } else if a >= 1.0 {
        1
    } else {
        // Enough decimals to show the step, and no more.
        let step_digits = if a >= 0.1 { 2 } else { 3 };
        step_digits
    };
    let s = format!("{:.*}", decimals, v);
    // **Trim the decimal point, never the digits.** My first version did
    // `trim_end_matches('0').trim_end_matches('.')`, which turned `100.0` into
    // `"1"` - so the largest x tick was labelled `1` instead of `100`. The
    // rounding to `decimals` places already produces the minimal representation, so
    // only a trailing `.` (and a bare `.0`) needs removing.
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s.as_str()
    };
    if s.is_empty() || s == "-" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

pub fn to_svg(figure: &Figure) -> String {
    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <style>
    .axis {{ font-family: Arial, sans-serif; font-size: 12px; fill: #333; }}
    .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #333; }}
    .panel-title {{ font-family: Arial, sans-serif; font-size: 13px; font-weight: bold; fill: #333; }}
    .label {{ font-family: Arial, sans-serif; font-size: 12px; fill: #666; }}
    .legend {{ font-family: Arial, sans-serif; font-size: 11px; fill: #333; }}
    .grid {{ stroke: #e0e0e0; stroke-width: 0.5; }}
  </style>
"#,
        figure.width, figure.height
    ));

    let rects = panel_rects(figure);
    let grid_layout = figure.subplots.len() > 1;

    // Title. A single-panel figure has no room for a separate panel title, so
    // the panel's own title is used when the figure has none; with several
    // panels the figure title goes over the grid and every panel draws its own.
    let heading: &str = if !figure.title.is_empty() {
        &figure.title
    } else if !grid_layout {
        figure
            .subplots
            .first()
            .map(|s| s.title.as_str())
            .unwrap_or("")
    } else {
        ""
    };
    if !heading.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="title" text-anchor="middle">{}</text>
"#,
            figure.width / 2.0,
            MARGIN_TOP / 2.0 + 8.0,
            heading
        ));
    }

    let max_series = figure
        .subplots
        .iter()
        .map(|s| s.series.len())
        .max()
        .unwrap_or(1)
        .max(1);

    for (panel_idx, subplot) in figure.subplots.iter().enumerate() {
        let rect = rects[panel_idx];

        if grid_layout && !subplot.title.is_empty() {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="panel-title" text-anchor="middle">{}</text>
"#,
                rect.center_x(),
                rect.y - 10.0,
                subplot.title
            ));
        }

        if figure.grid {
            // Vertical grid lines
            let num_v_lines = 5;
            for i in 0..=num_v_lines {
                let x = rect.x + (i as f64 / num_v_lines as f64) * rect.w;
                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                    x,
                    rect.y,
                    x,
                    rect.bottom()
                ));
            }
            // Horizontal grid lines
            let num_h_lines = 5;
            for i in 0..=num_h_lines {
                let y = rect.y + (i as f64 / num_h_lines as f64) * rect.h;
                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                    rect.x,
                    y,
                    rect.right(),
                    y
                ));
            }
        }

        // Axes
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>
"#,
            rect.x,
            rect.y,
            rect.x,
            rect.bottom(),
            rect.x,
            rect.bottom(),
            rect.right(),
            rect.bottom()
        ));

        // Plot each series of this panel, against this panel's own range.
        let bounds = Bounds::of(subplot);
        let mut unsupported = 0usize;
        for (series_idx, series) in subplot.series.iter().enumerate() {
            let series_color = series_color(series, panel_idx, series_idx, max_series);

            match series.plot_type {
                PlotType::Histogram | PlotType::Heatmap => {
                    // Nothing here draws these, and silently emitting an empty
                    // plot area looks like a plot whose data is all zero. Say
                    // so in the output instead; the `save_*` functions turn
                    // this into an error.
                    svg.push_str(&format!(
                        r#"  <text x="{}" y="{}" class="axis" text-anchor="middle">{}: {:?} is not rendered</text>
"#,
                        rect.center_x(),
                        rect.y + 20.0 + unsupported as f64 * 16.0,
                        series.label,
                        series.plot_type
                    ));
                    unsupported += 1;
                }
                _ => {
                    let Some(bounds) = bounds else { continue };
                    for run in finite_runs(series) {
                        match series.plot_type {
                            PlotType::Line => {
                                // A single point cannot describe a segment.
                                if run.len() < 2 {
                                    continue;
                                }
                                let mut points = String::new();
                                for (x, y) in &run {
                                    let (px, py) = bounds.project(&rect, *x, *y);
                                    points.push_str(&format!("{:.1},{:.1} ", px, py));
                                }
                                svg.push_str(&format!(
                                    r#"  <polyline points="{}" fill="none" stroke="{}" stroke-width="{}"{}/>
"#,
                                    points.trim(),
                                    series_color,
                                    series.style.line_width,
                                    opacity_attr(series.style.fill_alpha)
                                ));
                            }
                            PlotType::Scatter => {
                                for (x, y) in &run {
                                    let (px, py) = bounds.project(&rect, *x, *y);
                                    push_marker(
                                        &mut svg,
                                        &series.style.marker,
                                        px,
                                        py,
                                        series.style.marker_size,
                                        series_color,
                                        series.style.fill_alpha,
                                    );
                                }
                            }
                            PlotType::Bar => {
                                let bar_width = rect.w / series.x.len().max(1) as f64 * 0.8;
                                for (x, y) in &run {
                                    // Baseline is the bottom of the (padded)
                                    // axis, as before: the bar's height is the
                                    // value's distance above the axis minimum,
                                    // and its top edge still lands exactly on
                                    // the value.
                                    let (px, _) = bounds.project(&rect, *x, 0.0);
                                    let bar_height =
                                        (y - bounds.min_y) / (bounds.max_y - bounds.min_y) * rect.h;
                                    let py = rect.bottom() - bar_height;
                                    svg.push_str(&format!(
                                        r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="{}"/>
"#,
                                        px - bar_width / 2.0,
                                        py,
                                        bar_width,
                                        bar_height,
                                        series_color,
                                        series.style.fill_alpha
                                    ));
                                }
                            }
                            // Handled above.
                            PlotType::Histogram | PlotType::Heatmap => {}
                        }
                    }
                }
            }
        }

        // Tick labels and marks, before the axis titles so a label can never
        // overlap one.
        if let Some(b) = bounds.as_ref() {
            render_ticks(&mut svg, &rect, b, &subplot.x_label, &subplot.y_label);
        }

        // Axis labels
        if grid_layout {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{}</text>
"#,
                rect.center_x(),
                rect.bottom() + 26.0,
                subplot.x_label
            ));
            let ly = rect.center_y();
            let lx = rect.x - 30.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="middle" transform="rotate(-90, {}, {})">{}</text>
"#,
                lx, ly, lx, ly, subplot.y_label
            ));
        } else {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{}</text>
"#,
                figure.width / 2.0,
                figure.height - 10.0,
                subplot.x_label
            ));
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 15, {})">{}</text>
"#,
                15.0,
                figure.height / 2.0,
                figure.height / 2.0,
                subplot.y_label
            ));
        }
    }

    // Legend
    if figure.legend {
        let legend_x = figure.width - MARGIN_RIGHT - 100.0;
        let legend_y = MARGIN_TOP + 10.0;

        let mut legend_idx = 0;
        for (subplot_idx, subplot) in figure.subplots.iter().enumerate() {
            for (series_idx, series) in subplot.series.iter().enumerate() {
                let color = series_color(series, subplot_idx, series_idx, max_series);
                svg.push_str(&format!(
                    r#"  <rect x="{}" y="{}" width="12" height="12" fill="{}"/>
  <text x="{}" y="{}" class="legend">{}</text>
"#,
                    legend_x,
                    legend_y + legend_idx as f64 * 15.0,
                    color,
                    legend_x + 18.0,
                    legend_y + 10.0 + legend_idx as f64 * 15.0,
                    series.label
                ));
                legend_idx += 1;
            }
        }
    }

    svg.push_str("</svg>");
    svg
}

/// Convert plot to interactive HTML (plotly-like)
pub fn to_html(figure: &Figure) -> String {
    let svg = to_svg(figure);

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .plot-container {{ max-width: {width}px; margin: auto; }}
  </style>
</head>
<body>
  <div class="plot-container">
    {svg}
  </div>
</body>
</html>"#,
        title = figure.title,
        width = figure.width as i32,
        svg = svg,
    )
}

/// Reject a figure that cannot be drawn, before any file is created.
///
/// `to_svg` returns a plain `String`, so the renderer cannot report anything
/// itself: it renders what it can and leaves the rest out. Each `save_*`
/// function therefore runs this first, and a figure that would have produced a
/// plausible-looking but wrong or empty file is reported as
/// `PlotError::InvalidData` instead. The three cases:
///
/// - **No plottable point.** A figure with no data leaves the axis bounds at
///   their sentinels and `f64::MIN - f64::MAX` overflows to something the axis
///   code then treats as 1.0, so the output is valid, empty, and silently so.
/// - **A size that cannot be drawn.** `size(0.0, 0.0)` emits `width="0"`, which
///   the SVG spec defines as "do not render"; a negative or NaN size emits an
///   invalid header. Both used to save successfully.
/// - **Data the renderer cannot draw.** `x` and `y` of different lengths are
///   silently truncated to the shorter one by the zip in every drawing arm, and
///   `Histogram`/`Heatmap` series are drawn by no arm at all - a figure holding
///   only those saved a file with an empty plot area.
fn ensure_plottable(figure: &Figure) -> Result<(), crate::PlotError> {
    if !figure.width.is_finite()
        || !figure.height.is_finite()
        || figure.width <= 0.0
        || figure.height <= 0.0
    {
        return Err(crate::PlotError::InvalidData(format!(
            "figure size {}x{} is not a usable canvas: width and height must be finite and positive",
            figure.width, figure.height
        )));
    }

    for subplot in &figure.subplots {
        for series in &subplot.series {
            if series.x.len() != series.y.len() {
                return Err(crate::PlotError::InvalidData(format!(
                    "series `{}` has {} x values but {} y values; only pairwise data can be drawn",
                    series.label,
                    series.x.len(),
                    series.y.len()
                )));
            }
            match series.plot_type {
                PlotType::Line | PlotType::Scatter | PlotType::Bar => {}
                PlotType::Histogram | PlotType::Heatmap => {
                    return Err(crate::PlotError::InvalidData(format!(
                        "series `{}` is a {:?}, which this renderer does not draw; \
                         no output can be produced for it",
                        series.label, series.plot_type
                    )))
                }
            }
        }
    }

    let has_points = figure.subplots.iter().any(|s| {
        s.series.iter().any(|series| {
            series
                .x
                .iter()
                .zip(series.y.iter())
                .any(|(x, y)| x.is_finite() && y.is_finite())
        })
    });
    if !has_points {
        return Err(crate::PlotError::InvalidData(
            "figure contains no plottable points".to_string(),
        ));
    }
    Ok(())
}

/// Save figure to SVG file
pub fn save_svg(figure: &Figure, path: &str) -> Result<(), crate::PlotError> {
    ensure_plottable(figure)?;
    let svg = to_svg(figure);
    let mut file = File::create(path)?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

/// Save figure to HTML file
pub fn save_html(figure: &Figure, path: &str) -> Result<(), crate::PlotError> {
    // Same guard as `save_svg`: without it the same unusable figure produced an
    // error here and a silently written, empty HTML file.
    ensure_plottable(figure)?;
    let html = to_html(figure);
    let mut file = File::create(path)?;
    file.write_all(html.as_bytes())?;
    Ok(())
}

/// Save figure to PNG.
///
/// PNG export is not currently supported. This function returns an error
/// indicating that PNG format is unavailable. Use `save_svg` or `save_html`
/// instead.
pub fn save_png(_figure: &Figure, _path: &str) -> Result<(), crate::PlotError> {
    Err(crate::PlotError::Export(
        "PNG export is not supported. Use save_svg() or save_html() instead.".to_string(),
    ))
}
