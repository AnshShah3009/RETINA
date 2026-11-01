//! Scene shared by the `gaussian_splatting_basic` example and its regression
//! test.
//!
//! The example and the test have to agree on what the demo draws, so the scene
//! lives here rather than inline in `main`. `cv-examples` has no library
//! target, so both include this file with `#[path = ...]` (the same pattern
//! `bin/orb_matching.rs` uses for `visualization_utils`).

use cv_rendering::gaussian_splatting::{
    Camera, Gaussian, GaussianCloud, GaussianRasterizer, RasterizationResult,
};
use nalgebra::{Point3, Vector3, Vector4};

/// Distance from the camera to the plane the demo's grid is drawn on.
///
/// The renderer uses a right-handed camera looking down `+Z`: `Gaussian::project`
/// takes `view_pos[2]` as the depth and rejects anything with `depth <= 0.0` as
/// behind the camera. A grid at a negative `z` is therefore not "in front" but
/// entirely behind the eye, and every splat is culled.
pub const SCENE_DEPTH: f32 = 5.0;

/// The 5x5 grid of coloured splats the demo renders.
pub fn demo_cloud() -> GaussianCloud {
    let mut cloud = GaussianCloud::new();

    for i in 0..5 {
        for j in 0..5 {
            let x = (i as f32 - 2.0) * 0.5;
            let y = (j as f32 - 2.0) * 0.5;

            // Colour gradient based on position in the grid.
            let r = (i as f32) / 5.0;
            let g = (j as f32) / 5.0;
            let b = 0.5;

            let gaussian = Gaussian::new(
                Point3::new(x, y, SCENE_DEPTH),
                Vector3::new(0.1, 0.1, 0.1),
                Vector4::new(0.0, 0.0, 0.0, 1.0),
                Vector3::new(r, g, b),
            );
            cloud.push(gaussian);
        }
    }

    cloud
}

/// A camera at the origin looking down `+Z`, the direction the scene is in.
pub fn demo_camera() -> Camera {
    Camera::new(
        Point3::new(0.0, 0.0, 0.0),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        500.0,
        640,
        480,
    )
}

/// Renders the demo scene the way the example does.
pub fn render_demo() -> RasterizationResult {
    let rasterizer = GaussianRasterizer::new(demo_camera(), 16, 16);
    rasterizer.rasterize(&demo_cloud())
}

/// Pixels whose colour is not black, the same test `main` reports.
pub fn non_black_pixels(result: &RasterizationResult) -> usize {
    result
        .color
        .iter()
        .filter(|c| c.x > 0.01 || c.y > 0.01 || c.z > 0.01)
        .count()
}
