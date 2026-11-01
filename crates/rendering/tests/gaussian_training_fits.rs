//! `GaussianTrainer::train` must actually fit a target.
//!
//! With `backward` returning zeros the loss curve was flat - every iteration
//! the same value - and nothing distinguished that from convergence.

use cv_rendering::gaussian_splatting::{
    Camera, DifferentiableRasterizer, Gaussian, GaussianCloud, GaussianTrainer, TrainingConfig,
};
use nalgebra::{Point3, Vector3, Vector4};

const W: u32 = 48;
const H: u32 = 48;

fn camera() -> Camera {
    Camera::new(
        Point3::origin(),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        200.0,
        W,
        H,
    )
}

fn toy_config(iterations: usize) -> TrainingConfig {
    let mut config = TrainingConfig::default();
    config.iterations = iterations;
    config.densification.enabled = false;
    config.pruning.enabled = false;
    config.opacity_reset.enabled = false;
    // A toy scene of extent ~1: INRIA scales the position rate by the scene
    // extent, so use a rate appropriate to it.
    config.learning_rate_position = 2e-3;
    config.learning_rate_sh = 1e-2;
    config
}

fn target_image(bg: Vector3<f32>) -> Vec<Vector3<f32>> {
    let truth = GaussianCloud::from_gaussians(vec![
        Gaussian::new(
            Point3::new(0.06, -0.04, 3.0),
            Vector3::new(0.09, 0.06, 0.07),
            Vector4::new(0.0, 0.0, 0.2, 1.0),
            Vector3::new(0.9, 0.2, 0.1),
        )
        .with_opacity(0.85),
    ]);
    DifferentiableRasterizer::new(camera(), bg)
        .render(&truth)
        .color
}

#[test]
fn training_reduces_loss_on_a_toy_target() {
    let bg = Vector3::new(0.1, 0.1, 0.1);
    let target = target_image(bg);

    let mut cloud = GaussianCloud::from_gaussians(vec![
        Gaussian::new(
            Point3::new(0.0, 0.0, 3.0),
            Vector3::new(0.06, 0.06, 0.06),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.5),
    ]);

    let mut trainer =
        GaussianTrainer::new(DifferentiableRasterizer::new(camera(), bg), toy_config(300));
    let losses = trainer.train(&mut cloud, &target);
    let first = losses[0];
    let last = *losses.last().unwrap();
    println!("loss {first:.6e} -> {last:.6e}");
    println!("fitted {:?}", cloud.gaussians[0]);
    assert!(
        last < 0.1 * first,
        "loss must fall by at least 10x: {first} -> {last}"
    );
    // The fit moved towards the truth, not just somewhere cheaper.
    let g = &cloud.gaussians[0];
    assert!((g.spherical_harmonics.dc() - Vector3::new(0.9, 0.2, 0.1)).norm() < 0.15);
    assert!((g.position.x - 0.06).abs() < 0.02 && (g.position.y + 0.04).abs() < 0.02);
}
