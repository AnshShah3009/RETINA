//! Chart and Plot types

use crate::style::Style;
use crate::PlotError;

/// Type of plot to create
///
/// `Histogram` and `Heatmap` are declared but not drawn by any renderer: see
/// the crate documentation.
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
///
/// `x` and `y` are the pair of coordinate vectors. Every renderer walks them
/// with `zip`, so they must be the same length: `to_svg` draws the shorter of
/// the two, and `save_svg`/`save_html` reject a series whose lengths differ
/// rather than write a file that quietly holds less data than the series does.
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

/// Largest grid `subplot()` will allocate.
///
/// The panel count comes from two caller-supplied `usize`s, so `subplot()`
/// would otherwise try to allocate `rows * cols` panels for any pair of large
/// arguments - an out-of-memory abort for `subplot(usize::MAX, 2, 0)`. A cap
/// keeps a degenerate call from taking the process down; the grid shape itself
/// is still recorded verbatim.
const MAX_SUBPLOTS: usize = 4096;

/// Main plot/figure container
#[derive(Debug, Clone)]
pub struct Figure {
    pub title: String,
    pub width: f64,
    pub height: f64,
    pub subplots: Vec<SubPlot>,
    pub legend: bool,
    pub grid: bool,
    /// Grid shape requested by [`Figure::subplot`], as `(rows, cols)`.
    ///
    /// `None` until `subplot()` is called, in which case the exporter lays the
    /// panels out in a single row. A grid that is too small for
    /// `subplots.len()` (only possible if the panels were pushed by hand) falls
    /// back to the same single row.
    pub subplot_grid: Option<(usize, usize)>,
    /// Panel that the `&mut self` builder methods write into, i.e. the `index`
    /// last passed to [`Figure::subplot`].
    ///
    /// Clamped against `subplots.len()` on every use, so it can never address a
    /// panel that does not exist - including after a caller mutates the public
    /// `subplots` vector directly.
    pub current: usize,
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
            subplot_grid: None,
            current: 0,
        }
    }

    /// Index of the subplot the builder methods below write into.
    ///
    /// `subplots` and `current` are public and can drift apart, so the value is
    /// clamped at the point of use rather than when it is written.
    fn active_panel(&self) -> Option<usize> {
        if self.subplots.is_empty() {
            None
        } else {
            Some(self.current.min(self.subplots.len() - 1))
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
        if let Some(panel) = self.active_panel() {
            self.subplots[panel]
                .series
                .push(Series::new(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Add a scatter series
    pub fn scatter(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        if let Some(panel) = self.active_panel() {
            self.subplots[panel]
                .series
                .push(Series::scatter(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Add a bar series
    pub fn bar(&mut self, x: &[f64], y: &[f64], label: &str) -> &mut Self {
        if let Some(panel) = self.active_panel() {
            self.subplots[panel]
                .series
                .push(Series::bar(x.to_vec(), y.to_vec(), label));
        }
        self
    }

    /// Set the title of the current subplot.
    ///
    /// This is the *panel* title, not the figure title (which is set through
    /// `Figure::new` or `Plot::title`). A figure with one panel draws the
    /// figure title, falling back to this one when the figure has no title of
    /// its own; a figure with several panels draws this one above the panel it
    /// belongs to.
    pub fn title(&mut self, title: &str) -> &mut Self {
        if let Some(panel) = self.active_panel() {
            self.subplots[panel].title = title.to_string();
        }
        self
    }

    /// Set axis labels on the current subplot
    pub fn labels(&mut self, x_label: &str, y_label: &str) -> &mut Self {
        if let Some(panel) = self.active_panel() {
            self.subplots[panel].x_label = x_label.to_string();
            self.subplots[panel].y_label = y_label.to_string();
        }
        self
    }

    /// Add a new subplot, or select an existing one, and make it current.
    ///
    /// `rows` x `cols` is the grid the figure is laid out in and `index` is the
    /// panel inside it, counted from 0 in row-major order - the same order the
    /// panels are drawn in. Everything added afterwards (`add_series`,
    /// `scatter`, `bar`, `title`, `labels`) goes into that panel until the next
    /// `subplot()` call.
    ///
    /// The panel list grows to `rows * cols` entries if it is smaller, so
    /// `subplot(2, 2, 3)` is a legal way to reach the last panel of a fresh 2x2
    /// grid. `rows` or `cols` of 0 is treated as 1, an `index` past the last
    /// panel selects the last panel, and the whole call is capped at
    /// [`MAX_SUBPLOTS`] panels so that a degenerate argument cannot ask for an
    /// unbounded allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use cv_plot::Figure;
    ///
    /// let mut fig = Figure::new("two panels");
    /// fig.subplot(2, 1, 0).add_series(&[0.0, 1.0], &[0.0, 1.0], "top");
    /// fig.subplot(2, 1, 1).add_series(&[0.0, 1.0], &[1.0, 0.0], "bottom");
    ///
    /// assert!(fig.subplots[0].series[0].label == "top");
    /// assert!(fig.subplots[1].series[0].label == "bottom");
    /// ```
    pub fn subplot(&mut self, rows: usize, cols: usize, index: usize) -> &mut Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        // Saturating, because the product is caller-supplied and would overflow
        // in debug builds for large arguments (and wrap to a small number in
        // release, silently allocating the wrong grid).
        let panels = rows.saturating_mul(cols).min(MAX_SUBPLOTS);
        while self.subplots.len() < panels {
            self.subplots.push(SubPlot::default());
        }
        self.subplot_grid = Some((rows, cols));
        self.current = index.min(self.subplots.len() - 1);
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

    /// Set the figure title
    ///
    /// Note the deliberate difference from [`Figure::title`], which sets the
    /// title of the *current subplot*: a `Plot` owns one figure and has no
    /// panels to name.
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
