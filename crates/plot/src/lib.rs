//! Plotting and Visualization
//!
//! This crate provides plotting and visualization capabilities equivalent to Python's
//! matplotlib and plotly. It supports:
//!
//! - Line plots, scatter plots, bar charts, histograms
//! - Multiple series, subplots
//! - Export to SVG, HTML (interactive), PNG
//! - 3D visualization for point clouds
//!
//! ## Quick Start
//!
//! ```rust
//! use cv_plot::{Plot, PlotType};
//!
//! let mut plot = Plot::new("My Plot");
//! plot.add_series(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 9.0, 16.0], "y = x²");
//! plot.save("plot.svg").unwrap();
//! ```
//!
//! ## Plot Types
//!
//! - [`PlotType::Line`] -- Line plot
//! - [`PlotType::Scatter`] -- Scatter plot
//! - [`PlotType::Bar`] -- Bar chart
//! - [`PlotType::Histogram`] -- Histogram
//! - `PlotType::Heatmap` -- 2D heatmap

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
