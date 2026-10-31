#![forbid(unsafe_code)]
//! Regression tests for defects #2 and #3: `export::save_html` and
//! `Plot3D::save_svg` / `Plot3D::save_html` had no empty-figure guard, so a
//! figure with nothing to plot silently wrote a useless file while the sibling
//! `save_svg` correctly returned `PlotError::InvalidData`.

use cv_plot::chart::{Figure, PlotType, Series, SubPlot};
use cv_plot::three_d::{Plot3D, PointCloud3D};
use cv_plot::{save_html as fig_save_html, PlotError};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Unique temp path per test: tests run concurrently in one process, and this
/// repo has been bitten by shared temp paths before. Process id + counter.
fn unique_tmp(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "cv_plot_empty_guard_{}_{}_{}.out",
        std::process::id(),
        n,
        tag
    ));
    let _ = std::fs::remove_file(&path);
    path
}

fn empty_figure() -> Figure {
    let mut figure = Figure::new("empty");
    figure.subplots = vec![SubPlot::default()];
    figure
}

fn populated_figure() -> Figure {
    let mut figure = Figure::new("populated");
    let mut sub = SubPlot::default();
    let mut series = Series::new(vec![1.0, 2.0, 3.0], vec![1.0, 4.0, 9.0], "y");
    series.plot_type = PlotType::Line;
    sub.series.push(series);
    figure.subplots.push(sub);
    figure
}

/// Defect #2: `export::save_html` must reject an empty figure like `save_svg`.
/// CONTROL in the same test: a figure WITH points must still save.
#[test]
fn save_html_rejects_empty_figure_but_saves_populated() {
    let empty_path = unique_tmp("2d_empty_html");
    let err = fig_save_html(&empty_figure(), empty_path.to_str().unwrap())
        .expect_err("empty figure must be rejected by save_html");
    assert!(
        matches!(err, PlotError::InvalidData(ref m) if m.contains("no plottable points")),
        "expected InvalidData('...no plottable points'), got: {}",
        err
    );
    assert!(
        !empty_path.exists(),
        "no HTML file may be written for a figure with no points"
    );

    // CONTROL
    let ok_path = unique_tmp("2d_ok_html");
    fig_save_html(&populated_figure(), ok_path.to_str().unwrap())
        .expect("populated figure must still save as HTML");
    let body = std::fs::read_to_string(&ok_path).expect("HTML file must exist");
    assert!(body.contains("<svg") && body.contains("</svg>"));
}

/// Defect #3: `Plot3D::save_svg` / `Plot3D::save_html` must reject a plot with
/// no points. CONTROL: a populated Plot3D still saves both formats.
#[test]
fn plot3d_saves_reject_empty_but_save_populated() {
    let empty_svg = unique_tmp("3d_empty_svg");
    let empty_html = unique_tmp("3d_empty_html");
    let empty = Plot3D::new().add_point_cloud(PointCloud3D::new("empty"));

    let err = empty
        .save_svg(empty_svg.to_str().unwrap())
        .expect_err("Plot3D::save_svg must reject a plot with no points");
    assert!(
        matches!(err, PlotError::InvalidData(ref m) if m.contains("no points")),
        "expected InvalidData('...no points'), got: {}",
        err
    );
    assert!(!empty_svg.exists(), "no SVG file may be written");

    let err = empty
        .save_html(empty_html.to_str().unwrap())
        .expect_err("Plot3D::save_html must reject a plot with no points");
    assert!(
        matches!(err, PlotError::InvalidData(ref m) if m.contains("no points")),
        "expected InvalidData('...no points'), got: {}",
        err
    );
    assert!(!empty_html.exists(), "no HTML file may be written");

    // CONTROL
    let mut pc = PointCloud3D::new("populated");
    pc.add_points(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
    let full = Plot3D::new().add_point_cloud(pc);

    let svg_path = unique_tmp("3d_ok_svg");
    full.save_svg(svg_path.to_str().unwrap())
        .expect("populated Plot3D must still save as SVG");
    let svg = std::fs::read_to_string(&svg_path).expect("SVG file must exist");
    assert!(svg.contains("<svg") && svg.contains("<circle"));

    let html_path = unique_tmp("3d_ok_html");
    full.save_html(html_path.to_str().unwrap())
        .expect("populated Plot3D must still save as HTML");
    let html = std::fs::read_to_string(&html_path).expect("HTML file must exist");
    assert!(html.contains("THREE.Points"));

    // Defect #4 regression: `grid(false)` must actually change the SVG now that
    // `show_grid` is wired up, and `axes(false)` must drop the axis lines.
    let grid_off = full.clone().grid(false).to_svg();
    assert_ne!(
        grid_off,
        full.to_svg(),
        "Plot3D::grid(false) must change the rendered SVG (show_grid was dead)"
    );
    let axes_off = full.clone().axes(false).to_svg();
    assert_ne!(
        axes_off,
        full.to_svg(),
        "Plot3D::axes(false) must change the rendered SVG"
    );

    // Defect #4 regression: `camera_angle` must reach `to_html` (previously the
    // camera position was hardcoded and the field was unused there).
    let cam = full.clone().camera_angle(60.0, 120.0).to_html();
    assert!(
        cam.contains("camera.position.set(") && !cam.contains("camera.position.set(5, 5, 5)"),
        "camera_angle must be honoured by to_html"
    );
    assert_ne!(
        cam,
        full.to_html(),
        "camera_angle must change the HTML output"
    );
}
