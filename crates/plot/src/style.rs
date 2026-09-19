//! Plot styling and colors

use std::fmt;

/// Color representation (RGB, 0-255)
#[derive(Debug, Clone, Copy)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color(r, g, b)
    }

    pub fn hex(hex: &str) -> Self {
        let hex = hex.trim_start_matches('#');
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        Color(r, g, b)
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }

    pub fn red() -> Self {
        Color(255, 0, 0)
    }
    pub fn green() -> Self {
        Color(0, 255, 0)
    }
    pub fn blue() -> Self {
        Color(0, 0, 255)
    }
    pub fn black() -> Self {
        Color(0, 0, 0)
    }
    pub fn white() -> Self {
        Color(255, 255, 255)
    }
    pub fn yellow() -> Self {
        Color(255, 255, 0)
    }
    pub fn cyan() -> Self {
        Color(0, 255, 255)
    }
    pub fn magenta() -> Self {
        Color(255, 0, 255)
    }

    pub fn from_depth(t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let r = if t < 0.5 {
            0.0
        } else {
            (t - 0.5) * 2.0 * 255.0
        };
        let g = if t < 0.5 {
            t * 2.0 * 255.0
        } else {
            (1.0 - t) * 2.0 * 255.0
        };
        let b = if t < 0.5 {
            (0.5 - t) * 2.0 * 255.0
        } else {
            0.0
        };
        Color(r as u8, g as u8, b as u8)
    }

    pub fn r(&self) -> f64 {
        self.0 as f64 / 255.0
    }
    pub fn g(&self) -> f64 {
        self.1 as f64 / 255.0
    }
    pub fn b(&self) -> f64 {
        self.2 as f64 / 255.0
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rgb({}, {}, {})", self.0, self.1, self.2)
    }
}

/// Predefined color palette (matplotlib/seaborn inspired)
pub const COLORS: &[&str] = &[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
];

/// Plot styling
#[derive(Debug, Clone)]
pub struct Style {
    pub color: String,
    pub line_width: f64,
    pub marker_size: f64,
    pub marker: String,
    pub fill_alpha: f64,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            color: COLORS[0].to_string(),
            line_width: 2.0,
            marker_size: 6.0,
            marker: "o".to_string(),
            fill_alpha: 1.0,
        }
    }
}

impl Style {
    pub fn new(color: &str) -> Self {
        Self {
            color: color.to_string(),
            ..Default::default()
        }
    }

    pub fn line_width(mut self, width: f64) -> Self {
        self.line_width = width;
        self
    }

    pub fn marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }

    pub fn marker(mut self, marker: &str) -> Self {
        self.marker = marker.to_string();
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.fill_alpha = alpha;
        self
    }
}

/// Legend configuration
#[derive(Debug, Clone)]
pub struct Legend {
    pub show: bool,
    pub position: String,
}

impl Default for Legend {
    fn default() -> Self {
        Self {
            show: true,
            position: "best".to_string(),
        }
    }
}

impl Legend {
    pub fn position(mut self, pos: &str) -> Self {
        self.position = pos.to_string();
        self
    }
}
