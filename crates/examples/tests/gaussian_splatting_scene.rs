//! Regression test for the `gaussian_splatting_basic` example.
//!
//! The example used to place its 5x5 splat grid at `z = -5.0` and still report
//! a rendered image. `cv_rendering::gaussian_splatting` views `+Z` as the
//! direction the camera looks: `Gaussian::project` reads `view_pos[2]` as the
//! depth and returns `ProjectedGaussian::invalid()` when it is `<= 0.0`. Every
//! splat was therefore behind the eye, every one was culled, and the "rendered"
//! 640x480 image was uniformly black - the example printed `Non-black pixels:
//! 0` and still said "Example Complete", which reads as a successful render.
//!
//! The scene now lives in `src/gaussian_splatting_scene.rs` so this test renders
//! exactly what the example renders.

#[path = "../src/gaussian_splatting_scene.rs"]
mod gaussian_splatting_scene;

use gaussian_splatting_scene::{
    demo_camera, demo_cloud, non_black_pixels, render_demo, SCENE_DEPTH,
};

#[test]
fn demo_scene_is_in_front_of_the_camera() {
    // The renderer looks down `+Z`; a scene at a non-positive depth is behind
    // the camera and cannot appear in the image at all.
    assert!(
        SCENE_DEPTH > 0.0,
        "the demo grid is at z = {SCENE_DEPTH}, but the renderer treats +Z as \
         forward, so a non-positive depth puts the whole scene behind the camera"
    );
}

#[test]
fn demo_scene_renders_visible_pixels() {
    let result = render_demo();
    assert_eq!(result.width, 640);
    assert_eq!(result.height, 480);

    let lit = non_black_pixels(&result);
    assert!(
        lit > 0,
        "the 5x5 demo grid rendered an all-black {}x{} image: every splat was culled",
        result.width,
        result.height
    );

    // The grid is the only thing in the scene, so a working render also has to
    // put colour on the pixels it covers rather than scattering a single speck.
    assert!(
        lit >= 800,
        "expected the 5x5 grid to cover a large part of the frame, but only {lit} \
         of {} pixels were non-black",
        result.width * result.height
    );
}

#[test]
fn demo_camera_sees_the_demo_cloud() {
    let camera = demo_camera();
    let cloud = demo_cloud();
    let rasterizer = cv_rendering::gaussian_splatting::GaussianRasterizer::new(camera, 16, 16);
    let projected = rasterizer.project_gaussians(&cloud);

    assert_eq!(projected.len(), cloud.num_gaussians());
    let valid = projected.iter().filter(|pg| pg.is_valid()).count();
    assert_eq!(
        valid,
        cloud.num_gaussians(),
        "only {valid} of {} demo splats projected in front of the camera",
        cloud.num_gaussians()
    );
}
