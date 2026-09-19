//! Export plots to various formats

use crate::chart::{Figure, PlotType};
use crate::style::COLORS;
use std::fs::File;
use std::io::Write;

/// Convert plot to SVG string
pub fn to_svg(figure: &Figure) -> String {
    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <style>
    .axis {{ font-family: Arial, sans-serif; font-size: 12px; fill: #333; }}
    .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #333; }}
    .label {{ font-family: Arial, sans-serif; font-size: 12px; fill: #666; }}
    .legend {{ font-family: Arial, sans-serif; font-size: 11px; fill: #333; }}
    .grid {{ stroke: #e0e0e0; stroke-width: 0.5; }}
  </style>
"#,
        figure.width, figure.height
    ));

    // Plot area (leave margins)
    let margin_left = 60.0;
    let margin_right = 30.0;
    let margin_top = 50.0;
    let margin_bottom = 50.0;

    let plot_width = figure.width - margin_left - margin_right;
    let plot_height = figure.height - margin_top - margin_bottom;

    // Title
    if !figure.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="title" text-anchor="middle">{}</text>
"#,
            figure.width / 2.0,
            margin_top / 2.0 + 8.0,
            figure.title
        ));
    }

    // Get data bounds
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for subplot in &figure.subplots {
        for series in &subplot.series {
            for (x, y) in series.x.iter().zip(series.y.iter()) {
                min_x = min_x.min(*x);
                max_x = max_x.max(*x);
                min_y = min_y.min(*y);
                max_y = max_y.max(*y);
            }
        }
    }

    // Add padding to bounds
    let x_range = (max_x - min_x).max(1.0) * 0.1;
    let y_range = (max_y - min_y).max(1.0) * 0.1;
    min_x -= x_range;
    max_x += x_range;
    min_y -= y_range;
    max_y += y_range;

    // Grid
    if figure.grid {
        // Vertical grid lines
        let num_v_lines = 5;
        for i in 0..=num_v_lines {
            let x = margin_left + (i as f64 / num_v_lines as f64) * plot_width;
            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                x,
                margin_top,
                x,
                figure.height - margin_bottom
            ));
        }
        // Horizontal grid lines
        let num_h_lines = 5;
        for i in 0..=num_h_lines {
            let y = margin_top + (i as f64 / num_h_lines as f64) * plot_height;
            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
"#,
                margin_left,
                y,
                figure.width - margin_right,
                y
            ));
        }
    }

    // Axes
    svg.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>
"#,
        margin_left,
        margin_top,
        margin_left,
        figure.height - margin_bottom,
        margin_left,
        figure.height - margin_bottom,
        figure.width - margin_right,
        figure.height - margin_bottom
    ));

    // Plot each series
    let max_series = figure
        .subplots
        .iter()
        .map(|s| s.series.len())
        .max()
        .unwrap_or(1)
        .max(1);
    for (subplot_idx, subplot) in figure.subplots.iter().enumerate() {
        for (series_idx, series) in subplot.series.iter().enumerate() {
            let auto_color = COLORS[(subplot_idx * max_series + series_idx) % COLORS.len()];
            let series_color = if series.style.color.is_empty() {
                auto_color
            } else {
                &series.style.color
            };

            match series.plot_type {
                PlotType::Line => {
                    // Draw polyline
                    if series.x.len() > 1 {
                        let mut points = String::new();
                        for (x, y) in series.x.iter().zip(series.y.iter()) {
                            let px = margin_left + (x - min_x) / (max_x - min_x) * plot_width;
                            let py = figure.height
                                - margin_bottom
                                - (y - min_y) / (max_y - min_y) * plot_height;
                            points.push_str(&format!("{:.1},{:.1} ", px, py));
                        }
                        svg.push_str(&format!(
                            r#"  <polyline points="{}" fill="none" stroke="{}" stroke-width="{}"/>
"#,
                            points.trim(),
                            series_color,
                            series.style.line_width
                        ));
                    }
                }
                PlotType::Scatter => {
                    for (x, y) in series.x.iter().zip(series.y.iter()) {
                        let px = margin_left + (x - min_x) / (max_x - min_x) * plot_width;
                        let py = figure.height
                            - margin_bottom
                            - (y - min_y) / (max_y - min_y) * plot_height;
                        svg.push_str(&format!(
                            r#"  <circle cx="{}" cy="{}" r="{}" fill="{}"/>
"#,
                            px,
                            py,
                            series.style.marker_size / 2.0,
                            series_color
                        ));
                    }
                }
                PlotType::Bar => {
                    let bar_width = plot_width / series.x.len().max(1) as f64 * 0.8;
                    for (x, y) in series.x.iter().zip(series.y.iter()) {
                        let px = margin_left + (x - min_x) / (max_x - min_x) * plot_width;
                        let bar_height = (y - min_y) / (max_y - min_y) * plot_height;
                        let py = figure.height - margin_bottom - bar_height;
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
                _ => {}
            }
        }
    }

    // Axis labels
    if let Some(subplot) = figure.subplots.first() {
        // X label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{}</text>
"#,
            figure.width / 2.0,
            figure.height - 10.0,
            subplot.x_label
        ));

        // Y label (rotated)
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 15, {})">{}</text>
"#,
            15.0,
            figure.height / 2.0,
            figure.height / 2.0,
            subplot.y_label
        ));
    }

    // Legend
    if figure.legend {
        let legend_x = figure.width - margin_right - 100.0;
        let legend_y = margin_top + 10.0;

        let mut legend_idx = 0;
        for (subplot_idx, subplot) in figure.subplots.iter().enumerate() {
            for (series_idx, series) in subplot.series.iter().enumerate() {
                let color = if series.style.color.is_empty() {
                    COLORS[(subplot_idx * max_series + series_idx) % COLORS.len()]
                } else {
                    &series.style.color
                };
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

/// Save figure to SVG file
pub fn save_svg(figure: &Figure, path: &str) -> Result<(), crate::PlotError> {
    let svg = to_svg(figure);
    let mut file = File::create(path)?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

/// Save figure to HTML file
pub fn save_html(figure: &Figure, path: &str) -> Result<(), crate::PlotError> {
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
