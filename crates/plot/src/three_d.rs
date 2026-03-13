//! 3D Visualization
//!
//! This module provides 3D plotting capabilities for point clouds and 3D data,
//! similar to plotly's 3D scatter plots.
//!
//! ## Quick Start
//!
//! ```rust
//! use cv_plot::three_d::{PointCloud3D, Plot3D};
//!
//! // Create a point cloud from (x, y, z) coordinates
//! let mut pc = PointCloud3D::new("My Point Cloud");
//! pc.add_points(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
//!
//! // Create and save a 3D plot
//! let plot = Plot3D::new()
//!     .add_point_cloud(pc)
//!     .title("3D Point Cloud");
//! plot.save_html("pointcloud.html").unwrap();
//! ```

use crate::style::Color;
use crate::PlotError;
use std::io::Write;

/// A 3D point with position, normal, color and size
#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub nx: Option<f64>,
    pub ny: Option<f64>,
    pub nz: Option<f64>,
    pub color: Color,
    pub size: f64,
}

impl Point3D {
    /// Create a new 3D point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            nx: None,
            ny: None,
            nz: None,
            color: Color::blue(),
            size: 3.0,
        }
    }

    /// Create a 3D point with normal
    pub fn with_normal(x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64) -> Self {
        Self {
            x,
            y,
            z,
            nx: Some(nx),
            ny: Some(ny),
            nz: Some(nz),
            color: Color::blue(),
            size: 3.0,
        }
    }

    /// Create a colored 3D point
    pub fn with_color(x: f64, y: f64, z: f64, color: Color) -> Self {
        Self {
            x,
            y,
            z,
            nx: None,
            ny: None,
            nz: None,
            color,
            size: 3.0,
        }
    }

    /// Create a colored 3D point with normal
    pub fn with_all(x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64, color: Color) -> Self {
        Self {
            x,
            y,
            z,
            nx: Some(nx),
            ny: Some(ny),
            nz: Some(nz),
            color,
            size: 3.0,
        }
    }

    /// Set point size
    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self
    }
}

/// A 3D point cloud
#[derive(Debug, Clone)]
pub struct PointCloud3D {
    pub name: String,
    pub points: Vec<Point3D>,
    pub color: Color,
}

impl PointCloud3D {
    /// Create a new empty point cloud
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            points: Vec::new(),
            color: Color::blue(),
        }
    }

    /// Add points from separate x, y, z arrays
    pub fn add_points(&mut self, x: &[f64], y: &[f64], z: &[f64]) -> &mut Self {
        for i in 0..x.len().min(y.len()).min(z.len()) {
            self.points.push(Point3D::new(x[i], y[i], z[i]));
        }
        self
    }

    /// Add points with normals from separate x, y, z, nx, ny, nz arrays
    pub fn add_points_with_normals(
        &mut self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        nx: &[f64],
        ny: &[f64],
        nz: &[f64],
    ) -> &mut Self {
        let n = x
            .len()
            .min(y.len())
            .min(z.len())
            .min(nx.len())
            .min(ny.len())
            .min(nz.len());
        for i in 0..n {
            self.points
                .push(Point3D::with_normal(x[i], y[i], z[i], nx[i], ny[i], nz[i]));
        }
        self
    }

    /// Add colored points with normals
    #[allow(clippy::too_many_arguments)]
    pub fn add_points_with_all(
        &mut self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        nx: &[f64],
        ny: &[f64],
        nz: &[f64],
        colors: &[Color],
    ) -> &mut Self {
        let n = x
            .len()
            .min(y.len())
            .min(z.len())
            .min(nx.len())
            .min(ny.len())
            .min(nz.len())
            .min(colors.len());
        for i in 0..n {
            self.points.push(Point3D::with_all(
                x[i], y[i], z[i], nx[i], ny[i], nz[i], colors[i],
            ));
        }
        self
    }

    /// Add a single point
    pub fn add_point(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.points.push(Point3D::new(x, y, z));
        self
    }

    /// Add colored points
    pub fn add_colored_points(
        &mut self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        colors: &[Color],
    ) -> &mut Self {
        for i in 0..x.len().min(y.len()).min(z.len()).min(colors.len()) {
            self.points
                .push(Point3D::with_color(x[i], y[i], z[i], colors[i]));
        }
        self
    }

    /// Colorize points by depth (z-coordinate)
    pub fn colorize_by_depth(&mut self) -> &mut Self {
        if self.points.is_empty() {
            return self;
        }
        let min_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::INFINITY, f64::min);
        let max_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_z - min_z;

        for point in &mut self.points {
            let t = if range > 0.0 {
                (point.z - min_z) / range
            } else {
                0.5
            };
            point.color = Color::from_depth(t);
        }
        self
    }

    /// Set default color for all points
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        for point in &mut self.points {
            point.color = color;
        }
        self
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> (f64, f64, f64, f64, f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        }
        let min_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::INFINITY, f64::min);
        let max_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        (min_x, max_x, min_y, max_y, min_z, max_z)
    }
}

/// 3D plot configuration
#[derive(Debug, Clone)]
pub struct Plot3D {
    point_clouds: Vec<PointCloud3D>,
    title: String,
    width: f64,
    height: f64,
    camera_angle: (f64, f64),
    show_axes: bool,
    show_grid: bool,
    point_size: f64,
}

impl Default for Plot3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Plot3D {
    /// Create a new 3D plot
    pub fn new() -> Self {
        Self {
            point_clouds: Vec::new(),
            title: "3D Plot".to_string(),
            width: 800.0,
            height: 600.0,
            camera_angle: (30.0, 45.0),
            show_axes: true,
            show_grid: true,
            point_size: 3.0,
        }
    }

    /// Add a point cloud
    pub fn add_point_cloud(mut self, point_cloud: PointCloud3D) -> Self {
        self.point_clouds.push(point_cloud);
        self
    }

    /// Set plot title
    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set plot size
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set camera angle (elevation, azimuth in degrees)
    pub fn camera_angle(mut self, elevation: f64, azimuth: f64) -> Self {
        self.camera_angle = (elevation, azimuth);
        self
    }

    /// Show/hide axes
    pub fn axes(mut self, show: bool) -> Self {
        self.show_axes = show;
        self
    }

    /// Show/hide grid
    pub fn grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set default point size
    pub fn point_size(mut self, size: f64) -> Self {
        self.point_size = size;
        self
    }

    /// Convert 3D point to 2D screen coordinates using simple orthographic projection
    fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let (elev, azim) = self.camera_angle;
        let elev_rad = elev.to_radians();
        let azim_rad = azim.to_radians();

        let cos_elev = elev_rad.cos();
        let sin_elev = elev_rad.sin();
        let cos_azim = azim_rad.cos();
        let sin_azim = azim_rad.sin();

        let x_proj = x * cos_azim - z * sin_azim;
        let y_proj = y * cos_elev - (x * sin_azim + z * cos_azim) * sin_elev;

        (x_proj, y_proj)
    }

    /// Generate HTML with 3D visualization using Three.js
    pub fn to_html(&self) -> String {
        let mut js_points = String::new();
        let mut js_colors = String::new();

        for pc in self.point_clouds.iter() {
            for point in &pc.points {
                js_points.push_str(&format!("{{x:{},y:{},z:{}}},", point.x, point.y, point.z));
                let c = &point.color;
                js_colors.push_str(&format!(
                    "[{},{},{}],",
                    (c.r() * 255.0) as u8,
                    (c.g() * 255.0) as u8,
                    (c.b() * 255.0) as u8
                ));
            }
        }

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #container {{ width: {width}px; height: {height}px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="container"></div>
    <script>
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(75, {width}/{height}, 0.1, 1000);
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);
        
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize({width}, {height});
        container.appendChild(renderer.domElement);
        
        const points = [{js_points}];
        const colors = [{js_colors}];
        
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(points.length * 3);
        const colorArray = new Float32Array(points.length * 3);
        
        for (let i = 0; i < points.length; i++) {{
            positions[i * 3] = points[i].x;
            positions[i * 3 + 1] = points[i].y;
            positions[i * 3 + 2] = points[i].z;
            colorArray[i * 3] = colors[i][0] / 255;
            colorArray[i * 3 + 1] = colors[i][1] / 255;
            colorArray[i * 3 + 2] = colors[i][2] / 255;
        }}
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
        
        const material = new THREE.PointsMaterial({{size: {point_size}, vertexColors: true}});
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);
        
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);
        
        let isDragging = false;
        let previousMousePosition = {{x: 0, y: 0}};
        
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            previousMousePosition = {{x: e.clientX, y: e.clientY}};
        }});
        
        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('mouseleave', () => isDragging = false);
        
        container.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            pointCloud.rotation.y += deltaX * 0.01;
            pointCloud.rotation.x += deltaY * 0.01;
            previousMousePosition = {{x: e.clientX, y: e.clientY}};
        }});
        
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"#,
            title = self.title,
            width = self.width,
            height = self.height,
            js_points = js_points,
            js_colors = js_colors,
            point_size = self.point_size
        )
    }

    /// Save 3D plot as interactive HTML
    pub fn save_html(&self, path: &str) -> Result<(), PlotError> {
        let html = self.to_html();
        let mut file = std::fs::File::create(path)?;
        file.write_all(html.as_bytes())?;
        Ok(())
    }

    /// Generate SVG (simple 2D projection)
    pub fn to_svg(&self) -> String {
        if self.point_clouds.is_empty() {
            return String::new();
        }

        let (min_x, max_x, min_y, max_y, _min_z, _max_z) = self.bounding_box_all();
        let range_x = (max_x - min_x).max(1.0);
        let range_y = (max_y - min_y).max(1.0);
        let scale = (self.width.min(self.height) * 0.8) / range_x.max(range_y);
        let offset_x = self.width / 2.0;
        let offset_y = self.height / 2.0;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"0 0 {:.0} {:.0}\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#1a1a2e\"/>\n  <text x=\"{:.0}\" y=\"30\" fill=\"white\" font-size=\"16\" text-anchor=\"middle\">{}</text>\n",
            self.width,
            self.height,
            self.width,
            self.height,
            self.width / 2.0,
            self.title
        );

        if self.show_axes {
            let origin = self.project(0.0, 0.0, 0.0);
            let x_end = self.project(1.0, 0.0, 0.0);
            let y_end = self.project(0.0, 1.0, 0.0);
            let z_end = self.project(0.0, 0.0, 1.0);

            svg.push_str(&format!(
                r#"  <line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="red" stroke-width="2"/>
  <text x="{:.0}" y="{:.0}" fill="red" font-size="12">X</text>
  <line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="green" stroke-width="2"/>
  <text x="{:.0}" y="{:.0}" fill="green" font-size="12">Y</text>
  <line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="blue" stroke-width="2"/>
  <text x="{:.0}" y="{:.0}" fill="blue" font-size="12">Z</text>
"#,
                origin.0 * scale + offset_x,
                -origin.1 * scale + offset_y,
                x_end.0 * scale + offset_x,
                -x_end.1 * scale + offset_y,
                x_end.0 * scale + offset_x + 10.0,
                -x_end.1 * scale + offset_y,
                origin.0 * scale + offset_x,
                -origin.1 * scale + offset_y,
                y_end.0 * scale + offset_x,
                -y_end.1 * scale + offset_y,
                y_end.0 * scale + offset_x + 10.0,
                -y_end.1 * scale + offset_y,
                origin.0 * scale + offset_x,
                -origin.1 * scale + offset_y,
                z_end.0 * scale + offset_x,
                -z_end.1 * scale + offset_y,
                z_end.0 * scale + offset_x + 10.0,
                -z_end.1 * scale + offset_y,
            ));
        }

        for pc in &self.point_clouds {
            for point in &pc.points {
                let proj = self.project(point.x, point.y, point.z);
                let screen_x = proj.0 * scale + offset_x;
                let screen_y = -proj.1 * scale + offset_y;
                svg.push_str(&format!(
                    r#"  <circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="rgb({},{},{})" opacity="0.8"/>
"#,
                    screen_x, screen_y, point.size, (point.color.r() * 255.0) as u8,
                    (point.color.g() * 255.0) as u8, (point.color.b() * 255.0) as u8
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }

    /// Get combined bounding box of all point clouds
    fn bounding_box_all(&self) -> (f64, f64, f64, f64, f64, f64) {
        if self.point_clouds.is_empty() {
            return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for pc in &self.point_clouds {
            let (x1, x2, y1, y2, z1, z2) = pc.bounding_box();
            min_x = min_x.min(x1);
            max_x = max_x.max(x2);
            min_y = min_y.min(y1);
            max_y = max_y.max(y2);
            min_z = min_z.min(z1);
            max_z = max_z.max(z2);
        }
        (min_x, max_x, min_y, max_y, min_z, max_z)
    }

    /// Save as SVG file
    pub fn save_svg(&self, path: &str) -> Result<(), PlotError> {
        let svg = self.to_svg();
        let mut file = std::fs::File::create(path)?;
        file.write_all(svg.as_bytes())?;
        Ok(())
    }
}

/// Get bounding box of combined point clouds
#[allow(dead_code)]
fn bounding_box_all(point_clouds: &[PointCloud3D]) -> (f64, f64, f64, f64, f64, f64) {
    if point_clouds.is_empty() {
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for pc in point_clouds {
        let (x1, x2, y1, y2, z1, z2) = pc.bounding_box();
        min_x = min_x.min(x1);
        max_x = max_x.max(x2);
        min_y = min_y.min(y1);
        max_y = max_y.max(y2);
        min_z = min_z.min(z1);
        max_z = max_z.max(z2);
    }
    (min_x, max_x, min_y, max_y, min_z, max_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let mut pc = PointCloud3D::new("Test");
        pc.add_points(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(pc.points.len(), 3);
    }

    #[test]
    fn test_point_cloud_bounding_box() {
        let mut pc = PointCloud3D::new("Test");
        pc.add_points(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]);
        let (min_x, max_x, min_y, max_y, min_z, max_z) = pc.bounding_box();
        assert_eq!((min_x, max_x), (1.0, 2.0));
        assert_eq!((min_y, max_y), (1.0, 2.0));
        assert_eq!((min_z, max_z), (1.0, 2.0));
    }

    #[test]
    fn test_plot3d_html_generation() {
        let mut pc = PointCloud3D::new("Test");
        pc.add_points(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);

        let plot = Plot3D::new().add_point_cloud(pc);
        let html = plot.to_html();

        assert!(html.contains("THREE.Points"));
        assert!(html.contains("three.min.js"));
    }

    #[test]
    fn test_plot3d_svg_generation() {
        let mut pc = PointCloud3D::new("Test");
        pc.add_points(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);

        let plot = Plot3D::new().add_point_cloud(pc);
        let svg = plot.to_svg();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_colorize_by_depth() {
        let mut pc = PointCloud3D::new("Test");
        pc.add_points(&[0.0, 0.5, 1.0], &[0.0, 0.0, 0.0], &[0.0, 0.5, 1.0]);
        pc.colorize_by_depth();

        assert!(pc.points[0].color.r() < pc.points[2].color.r());
    }
}
