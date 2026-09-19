//! Chart and Plot types

use crate::style::Style;
use crate::PlotError;

/// Type of plot to create
#[derive(Debug, Clone, Default)]
pub enum PlotType {
    #[default]
    Line,
    Scatter,
    Bar,
    Histogram,
    Heatmap,
}

/// A data series for plotting
#[derive(Debug, Clone)]
pub struct Series {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub label: String,
    pub plot_type: PlotType,
    pub style: Style,
}

impl Series {
    /// Create a new series from x and y data
    pub fn new(x: Vec<f64>, y: Vec<f64>, label: &str) -> Self {
        Self {
            x,
            y,
            label: label.to_string(),
            plot_type: PlotType::default(),
            style: Style::default(),
        }
    }

    /// Create a scatter plot series
    pub fn scatter(x: Vec<f64>, y: Vec<f64>, label: &str) -> Self {
        Self {
            x,
            y,
            label: label.to_string(),
            plot_type: PlotType::Scatter,
            style: Style::default(),
        }
    }

    /// Create a bar chart series
    pub fn bar(x: Vec<f64>, y: Vec<f64>, label: &str) -> Self {
        Self {
            x,
            y,
            label: label.to_string(),
            plot_type: PlotType::Bar,
            style: Style::default(),
        }
    }
}

/// A subplot within a figure
#[derive(Debug, Clone)]
pub struct SubPlot {
    pub series: Vec<Series>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
}

impl Default for SubPlot {
    fn default() -> Self {
        Self {
            series: Vec::new(),
            title: String::new(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
        }
    }
}

/// Main plot/figure container
#[derive(Debug, Clone)]
pub struct Figure {
    pub title: String,
    pub width: f64,
    pub height: f64,
    pub subplots: Vec<SubPlot>,
    pub legend: bool,
    pub grid: bool,
}

impl Figure {
    /// Create a new figure
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            width: 800.0,
            height: 600.0,
            subplots: vec![SubPlot::default()],
            legend: true,
            grid: true,
        }
    }

    /// Set figure size
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Enable/disable legend
    pub fn legend(mut self, show: bool) -> Self {
        self.legend = show;
        self
    }

    /// Enable/disable grid
    pub fn grid(mut self, show: bool) -> Self {
        self.grid = show;
        self
    }

    /// Add a series to the current subplot
    pub fn add_series(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        if let Some(subplot) = self.subplots.last_mut() {
            subplot
                .series
                .push(Series::new(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Add a scatter series
    pub fn scatter(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        if let Some(subplot) = self.subplots.last_mut() {
            subplot
                .series
                .push(Series::scatter(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Add a bar series
    pub fn bar(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        if let Some(subplot) = self.subplots.last_mut() {
            subplot
                .series
                .push(Series::bar(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Set subplot title
    pub fn title(&mut self, title: &str) -> &mut Self {
        if let Some(subplot) = self.subplots.last_mut() {
            subplot.title = title.to_string();
        }
        self
    }

    /// Set axis labels
    pub fn labels(&mut self, x_label: &str, y_label: &str) -> &mut Self {
        if let Some(subplot) = self.subplots.last_mut() {
            subplot.x_label = x_label.to_string();
            subplot.y_label = y_label.to_string();
        }
        self
    }

    /// Add a new subplot
    pub fn subplot(&mut self, rows: usize, cols: usize, _index: usize) -> &mut Self {
        while self.subplots.len() < rows * cols {
            self.subplots.push(SubPlot::default());
        }
        self
    }
}

/// Simple plot builder (single plot)
pub struct Plot {
    figure: Figure,
}

impl Plot {
    /// Create a new plot
    pub fn new(title: &str) -> Self {
        Self {
            figure: Figure::new(title),
        }
    }

    /// Add a data series
    pub fn add_series(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        self.figure.add_series(x, y, label);
        self
    }

    /// Add a scatter series
    pub fn scatter(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        self.figure.scatter(x, y, label);
        self
    }

    /// Set plot title
    pub fn title(&mut self, title: &str) -> &mut Self {
        self.figure.title = title.to_string();
        self
    }

    /// Set axis labels
    pub fn labels(&mut self, x_label: &str, y_label: &str) -> &mut Self {
        self.figure.labels(x_label, y_label);
        self
    }

    /// Enable legend
    pub fn legend(&mut self, show: bool) -> &mut Self {
        self.figure.legend = show;
        self
    }

    /// Enable grid
    pub fn grid(&mut self, show: bool) -> &mut Self {
        self.figure.grid = show;
        self
    }

    /// Set size
    pub fn size(&mut self, width: f64, height: f64) -> &mut Self {
        self.figure.width = width;
        self.figure.height = height;
        self
    }

    /// Get the figure
    pub fn build(self) -> Figure {
        self.figure
    }

    /// Save to SVG file
    pub fn save(&self, path: &str) -> Result<(), PlotError> {
        crate::export::save_svg(&self.figure, path)
    }

    /// Save to HTML file
    pub fn save_html(&self, path: &str) -> Result<(), PlotError> {
        crate::export::save_html(&self.figure, path)
    }

    /// Convert to SVG string
    pub fn to_svg(&self) -> String {
        crate::export::to_svg(&self.figure)
    }

    /// Convert to HTML string
    pub fn to_html(&self) -> String {
        crate::export::to_html(&self.figure)
    }
}
