#![forbid(unsafe_code)]
//! Examples for cv-rendering (Gaussian Splatting)
//!
//! Run with: cargo run --example gaussian_splatting_basic

#[path = "gaussian_splatting_scene.rs"]
mod gaussian_splatting_scene;

use gaussian_splatting_scene::{demo_camera, demo_cloud, non_black_pixels, render_demo};

fn main() {
    println!("=== Gaussian Splatting Basic Example ===\n");

    // A 5x5 grid of coloured splats, shared with the regression test in
    // `tests/gaussian_splatting_scene.rs` so the two cannot drift apart.
    let cloud = demo_cloud();
    println!("Created cloud with {} Gaussians", cloud.num_gaussians());

    // The scene is in front of the camera along +Z, which is the direction
    // `Gaussian::project` treats as forward.
    let camera = demo_camera();
    println!(
        "Camera: {}x{}, f = {}, looking down +Z",
        camera.width, camera.height, camera.focal_length
    );

    // Create rasterizer and render
    let result = render_demo();

    println!("Rendered image: {}x{}", result.width, result.height);
    println!("Total pixels: {}", result.color.len());

    // Check output
    let non_black = non_black_pixels(&result);
    println!("Non-black pixels: {}", non_black);

    // Save to PNG (if image crate available)
    let image_data = result.to_image();
    println!("Image buffer size: {} bytes", image_data.len());

    println!("\n=== Example Complete ===");
}
