//! Examples for cv-rendering (Gaussian Splatting)
//!
//! Run with: cargo run --example gaussian_splatting_basic

use cv_rendering::gaussian_splatting::{Camera, Gaussian, GaussianCloud, GaussianRasterizer};
use nalgebra::{Point3, Vector3, Vector4};

fn main() {
    println!("=== Gaussian Splatting Basic Example ===\n");

    // Create a simple scene with multiple Gaussians
    let mut cloud = GaussianCloud::new();

    // Add a few test Gaussians in a pattern
    for i in 0..5 {
        for j in 0..5 {
            let x = (i as f32 - 2.0) * 0.5;
            let y = (j as f32 - 2.0) * 0.5;
            let z = -5.0;

            // Color gradient based on position
            let r = (i as f32) / 5.0;
            let g = (j as f32) / 5.0;
            let b = 0.5;

            let gaussian = Gaussian::new(
                Point3::new(x, y, z),
                Vector3::new(0.1, 0.1, 0.1),
                Vector4::new(0.0, 0.0, 0.0, 1.0),
                Vector3::new(r, g, b),
            );
            cloud.push(gaussian);
        }
    }

    println!("Created cloud with {} Gaussians", cloud.num_gaussians());

    // Create camera
    let camera = Camera::new(
        Point3::new(0.0, 0.0, 0.0),       // Camera at origin
        Vector4::new(0.0, 0.0, 0.0, 1.0), // Looking forward
        500.0,                            // Focal length
        640,
        480, // Resolution
    );

    // Create rasterizer and render
    let rasterizer = GaussianRasterizer::new(camera, 16, 16);
    let result = rasterizer.rasterize(&cloud);

    println!("Rendered image: {}x{}", result.width, result.height);
    println!("Total pixels: {}", result.color.len());

    // Check output
    let non_black = result
        .color
        .iter()
        .filter(|c| c.x > 0.01 || c.y > 0.01 || c.z > 0.01)
        .count();
    println!("Non-black pixels: {}", non_black);

    // Save to PNG (if image crate available)
    let image_data = result.to_image();
    println!("Image buffer size: {} bytes", image_data.len());

    println!("\n=== Example Complete ===");
}
