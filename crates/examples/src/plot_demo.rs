//! Example: Plotting and 3D Visualization
//!
//! Run with: cargo run --example plot_demo

use cv_plot::{Plot, Plot3D, PointCloud3D};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== cv-plot Example ===\n");

    println!("1. Creating a simple 2D line plot...");
    create_line_plot();

    println!("\n2. Creating a scatter plot...");
    create_scatter_plot();

    println!("\n3. Creating a 3D point cloud visualization...");
    create_3d_plot();

    println!("\n=== Example Complete ===");
    println!("Check the generated files in the current directory:");
    println!("  - plot_line.svg");
    println!("  - plot_scatter.svg");
    println!("  - plot_3d.html (open in browser for interactive 3D)");
}

fn create_line_plot() {
    let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&x| x.sin()).collect();
    let y2: Vec<f64> = x.iter().map(|&x| x.cos()).collect();

    let mut plot = Plot::new("Trigonometric Functions");
    plot.add_series(&x, &y, "sin(x)");
    plot.add_series(&x, &y2, "cos(x)");
    plot.labels("x", "y");
    plot.legend(true);
    plot.grid(true);

    plot.save("plot_line.svg").unwrap();
    println!("  Saved plot_line.svg");
}

fn create_scatter_plot() {
    let mut rng = StdRng::seed_from_u64(42);

    let x: Vec<f64> = (0..50).map(|_| rng.random_range(-3.0..3.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&x| x * x + rng.random_range(-0.5..0.5))
        .collect();

    let mut plot = Plot::new("Scatter Plot");
    plot.scatter(&x, &y, "Data points");
    plot.labels("x", "y = x² + noise");
    plot.grid(true);

    plot.save("plot_scatter.svg").unwrap();
    println!("  Saved plot_scatter.svg");
}

fn create_3d_plot() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(42);

    let mut pc = PointCloud3D::new("Random Points");

    let x: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
    let z: Vec<f64> = (0..n)
        .map(|i| {
            let x = x[i];
            let y = y[i];
            (x * x + y * y).sin() * 0.5
        })
        .collect();

    pc.add_points(&x, &y, &z);
    pc.colorize_by_depth();

    let plot = Plot3D::new()
        .add_point_cloud(pc)
        .title("3D Point Cloud")
        .size(800.0, 600.0)
        .axes(true)
        .grid(true)
        .point_size(2.0);

    plot.save_html("plot_3d.html").unwrap();
    println!("  Saved plot_3d.html (open in browser)");

    plot.save_svg("plot_3d.svg").unwrap();
    println!("  Saved plot_3d.svg");
}
