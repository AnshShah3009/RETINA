//! A training run whose gradient is identically zero while the loss is not
//! must be reported as stalled, never as converged or as a normal finish.

use cv_rendering::gaussian_splatting::{
    Camera, DifferentiableRasterizer, Gaussian, GaussianCloud, GaussianTrainer, TrainingConfig,
    TrainingStatus,
};
use nalgebra::{Point3, Vector3, Vector4};

fn camera() -> Camera {
    Camera::new(
        Point3::origin(),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        100.0,
        24,
        24,
    )
}

fn config() -> TrainingConfig {
    let mut c = TrainingConfig::default();
    c.iterations = 50;
    c.densification.enabled = false;
    c.pruning.enabled = false;
    c.opacity_reset.enabled = false;
    c
}

#[test]
fn zero_gradient_with_positive_loss_is_reported_as_stalled() {
    // The only splat is behind the camera: it renders nothing, so no
    // parameter can change the image, yet the target is not the background.
    let mut cloud = GaussianCloud::from_gaussians(vec![Gaussian::new(
        Point3::new(0.0, 0.0, -3.0),
        Vector3::new(0.1, 0.1, 0.1),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(0.5, 0.5, 0.5),
    )]);
    let target = vec![Vector3::new(0.8, 0.8, 0.8); 24 * 24];
    let mut trainer = GaussianTrainer::new(
        DifferentiableRasterizer::new(camera(), Vector3::zeros()),
        config(),
    );
    let report = trainer.train_with_report(&mut cloud, &target);
    assert!(report.losses[0] > 0.1);
    assert_eq!(report.status, TrainingStatus::Stalled);
    assert_eq!(report.final_gradient_norm, 0.0);
    assert!(report.losses.len() < 50, "a stalled run must stop early");
}

#[test]
fn ordinary_run_reports_nonzero_gradient_and_completes() {
    let mut cloud = GaussianCloud::from_gaussians(vec![Gaussian::new(
        Point3::new(0.0, 0.0, 3.0),
        Vector3::new(0.1, 0.1, 0.1),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(0.5, 0.5, 0.5),
    )]);
    let target = vec![Vector3::new(0.8, 0.2, 0.2); 24 * 24];
    let mut trainer = GaussianTrainer::new(
        DifferentiableRasterizer::new(camera(), Vector3::zeros()),
        config(),
    );
    let report = trainer.train_with_report(&mut cloud, &target);
    assert_eq!(report.status, TrainingStatus::Completed);
    assert_eq!(report.losses.len(), 50);
    assert!(report.final_gradient_norm > 0.0);
    assert!(report.losses[49] < report.losses[0]);
}
