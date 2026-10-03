#![forbid(unsafe_code)]
//! Plotting and Visualization
//!
//! This crate provides plotting and visualization capabilities equivalent to Python's
//! matplotlib and plotly. It supports:
//!
//! - Line plots, scatter plots, bar charts
//! - Multiple series, subplot grids
//! - Export to SVG and HTML (interactive)
//! - 3D visualization for point clouds
//!
//! ## Quick Start
//!
//! ```no_run
//! use cv_plot::{Plot, PlotType};
//!
//! let mut plot = Plot::new("My Plot");
//! plot.add_series(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 9.0, 16.0], "y = x²");
//! plot.save("plot.svg").unwrap();
//! ```
//!
//! `no_run`, for the same reason as the example in [`three_d`]: doctests execute
//! with the crate root as the working directory, so running this one during
//! `cargo test` rewrote the tracked `crates/plot/plot.svg` on every test run.
//! It still type-checks, which is the part worth testing.
//!
//! ## Plot Types
//!
//! - [`PlotType::Line`] -- Line plot
//! - [`PlotType::Scatter`] -- Scatter plot
//! - [`PlotType::Bar`] -- Bar chart
//!
//! [`PlotType::Histogram`] and [`PlotType::Heatmap`] are declared but nothing
//! draws them. `to_svg` marks such a series as unrendered in its output, and
//! `save_svg`/`save_html` reject a figure that holds only those series instead
//! of writing a file with an empty plot area.
//!
//! ## Subplots
//!
//! [`Figure::subplot`] selects a panel in a `rows` x `cols` grid; every panel
//! is scaled to its own data. See the method's documentation for the index
//! convention.

pub mod chart;
pub mod export;
pub mod style;
pub mod three_d;

pub use chart::{Figure, Plot, PlotType, Series, SubPlot};
pub use export::{save_html, save_png, save_svg, to_html, to_svg};
pub use style::{Color, Legend, Style, COLORS};
pub use three_d::{Plot3D, Point3D, PointCloud3D};

/// Plotting error types
#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Export error: {0}")]
    Export(String),
}
