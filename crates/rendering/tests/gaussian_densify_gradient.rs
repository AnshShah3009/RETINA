//! 3DGS adaptive density control (Kerbl et al. 2023, §5.2; INRIA
//! `densify_and_clone` / `densify_and_split`):
//!
//! * accumulate the norm of each Gaussian's view-space (NDC) positional
//!   gradient over the iterations in which it is visible;
//! * at a densification step, select those whose *average* exceeds
//!   `grad_threshold`;
//! * clone the small ones (max scale <= `size_threshold`) unchanged;
//! * split the large ones into 2 samples drawn from the Gaussian itself, with
//!   scale divided by 1.6, and remove the original;
//! * reset the accumulators.

use cv_rendering::gaussian_splatting::rasterize::GaussianGradient;
use cv_rendering::gaussian_splatting::{
    Camera, DifferentiableRasterizer, Gaussian, GaussianCloud, GaussianOptimizer, GaussianTrainer,
    TrainingConfig,
};
use nalgebra::{Point3, Vector2, Vector3, Vector4};

fn gaussian(scale: Vector3<f32>) -> Gaussian {
    Gaussian::new(
        Point3::new(0.1, -0.2, 3.0),
        scale,
        Vector4::new(0.1, 0.2, 0.3, 0.9),
        Vector3::new(0.7, 0.4, 0.2),
    )
    .with_opacity(0.8)
}

/// A gradient carrying only a view-space positional gradient of NDC norm
/// `ndc_norm` for splat `idx`.
fn grad(idx: usize, ndc_norm: f32, visible: bool) -> GaussianGradient {
    GaussianGradient {
        idx,
        position_grad: Vector3::zeros(),
        scale_grad: Vector3::zeros(),
        rotation_grad: Vector4::zeros(),
        opacity_grad: 0.0,
        color_grad: Vector3::zeros(),
        mean2d_grad: Vector2::zeros(),
        mean2d_grad_ndc: Vector2::new(0.6 * ndc_norm, 0.8 * ndc_norm),
        visible,
    }
}

fn optimizer() -> GaussianOptimizer {
    GaussianOptimizer::new(TrainingConfig::default())
}

#[test]
fn small_gaussian_above_threshold_is_cloned() {
    let thr = TrainingConfig::default().densification.grad_threshold;
    let mut cloud =
        GaussianCloud::from_gaussians(vec![gaussian(Vector3::new(0.004, 0.005, 0.003))]);
    let mut opt = optimizer();
    opt.accumulate_densification_stats(&[grad(0, 2.0 * thr, true)]);
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 2);
    for g in &cloud.gaussians {
        assert_eq!(g.position, Point3::new(0.1, -0.2, 3.0), "clone is exact");
        assert_eq!(g.scale, Vector3::new(0.004, 0.005, 0.003));
        assert_eq!(g.opacity, 0.8);
    }
    // Statistics were reset: a second call without new gradients is a no-op.
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 2);
}

#[test]
fn large_gaussian_above_threshold_is_split_in_two() {
    let thr = TrainingConfig::default().densification.grad_threshold;
    let original = gaussian(Vector3::new(0.3, 0.1, 0.05));
    let mut cloud = GaussianCloud::from_gaussians(vec![original.clone()]);
    let mut opt = optimizer();
    opt.accumulate_densification_stats(&[grad(0, 2.0 * thr, true)]);
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 2, "original replaced by 2 samples");
    for g in &cloud.gaussians {
        assert!(
            (g.scale - original.scale / 1.6).norm() < 1e-7,
            "{:?}",
            g.scale
        );
        assert_eq!(g.rotation, original.rotation);
        assert_eq!(g.opacity, original.opacity);
        assert_eq!(g.spherical_harmonics, original.spherical_harmonics);
        // Sampled from the original Gaussian: offset within ~5 sigma along
        // its own axes.
        let local = original.rotation_matrix().transpose() * (g.position - original.position);
        let z = local.component_div(&original.scale);
        assert!(z.norm() < 6.0, "sample at {z:?} sigma");
    }
    assert_ne!(cloud.gaussians[0].position, cloud.gaussians[1].position);
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 2);
}

#[test]
fn below_threshold_and_invisible_gaussians_are_left_alone() {
    let thr = TrainingConfig::default().densification.grad_threshold;
    let mut cloud = GaussianCloud::from_gaussians(vec![
        gaussian(Vector3::new(0.004, 0.004, 0.004)),
        gaussian(Vector3::new(0.3, 0.3, 0.3)),
        gaussian(Vector3::new(0.004, 0.004, 0.004)),
    ]);
    let mut opt = optimizer();
    opt.accumulate_densification_stats(&[
        grad(0, 0.5 * thr, true),
        grad(1, 0.9 * thr, true),
        // Invisible: the gradient is not counted at all.
        grad(2, 100.0 * thr, false),
    ]);
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 3);
}

#[test]
fn threshold_applies_to_the_average_over_visible_iterations() {
    let thr = TrainingConfig::default().densification.grad_threshold;
    let small = Vector3::new(0.004, 0.004, 0.004);

    // Average (3 + 0 + 0) / 3 = 1.0 thr: densified (>= threshold, as INRIA).
    let mut cloud = GaussianCloud::from_gaussians(vec![gaussian(small)]);
    let mut opt = optimizer();
    opt.accumulate_densification_stats(&[grad(0, 3.0 * thr, true)]);
    opt.accumulate_densification_stats(&[grad(0, 0.0, true)]);
    opt.accumulate_densification_stats(&[grad(0, 0.0, true)]);
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 2);

    // Average (3 + 0 + 0 + 0) / 4 = 0.75 thr: not densified, although the
    // sum is above the threshold.
    let mut cloud = GaussianCloud::from_gaussians(vec![gaussian(small)]);
    let mut opt = optimizer();
    opt.accumulate_densification_stats(&[grad(0, 3.0 * thr, true)]);
    for _ in 0..3 {
        opt.accumulate_densification_stats(&[grad(0, 0.0, true)]);
    }
    opt.densify(&mut cloud);
    assert_eq!(cloud.num_gaussians(), 1);
}

#[test]
fn repeated_gradient_driven_densify_is_bounded_by_signal() {
    // Each call can at most double the selected set, and only with fresh
    // gradient; with gradient on splat 0 only, 10 calls add at most 10.
    let thr = TrainingConfig::default().densification.grad_threshold;
    let mut cloud = GaussianCloud::from_gaussians(vec![
        gaussian(Vector3::new(0.004, 0.004, 0.004)),
        gaussian(Vector3::new(0.004, 0.004, 0.004)),
    ]);
    let mut opt = optimizer();
    for _ in 0..10 {
        opt.accumulate_densification_stats(&[grad(0, 2.0 * thr, true)]);
        opt.densify(&mut cloud);
    }
    assert_eq!(cloud.num_gaussians(), 12);
}

#[test]
fn backward_reports_ndc_mean_gradient() {
    // mean2d_grad_ndc is the pixel-space gradient scaled by (W/2, H/2): the
    // units of the reference's view-space gradient that grad_threshold is in.
    let camera = Camera::new(
        Point3::origin(),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        100.0,
        40,
        30,
    );
    let r = DifferentiableRasterizer::new(camera, Vector3::zeros());
    let cloud = GaussianCloud::from_gaussians(vec![gaussian(Vector3::new(0.1, 0.1, 0.1))]);
    let target = vec![Vector3::new(0.3, 0.9, 0.1); 40 * 30];
    let g = &r.backward(&cloud, &r.render(&cloud), &target)[0];
    assert!(g.visible);
    assert!(g.mean2d_grad.norm() > 0.0);
    assert!((g.mean2d_grad_ndc.x - g.mean2d_grad.x * 20.0).abs() < 1e-9);
    assert!((g.mean2d_grad_ndc.y - g.mean2d_grad.y * 15.0).abs() < 1e-9);
}

#[test]
fn training_with_densification_stays_bounded_and_fits() {
    // Two adjacent blobs, one initial splat: densification inside the
    // training loop should add splats, stay bounded, and the fit must still
    // converge (measured over 12 runs: loss 1.1e-2 -> 1.2e-6..1.5e-5 with
    // 19-30 splats; the split samples are random).
    let camera = || {
        Camera::new(
            Point3::origin(),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            200.0,
            48,
            48,
        )
    };
    let bg = Vector3::new(0.1, 0.1, 0.1);
    let truth = GaussianCloud::from_gaussians(vec![
        Gaussian::new(
            Point3::new(-0.08, 0.0, 3.0),
            Vector3::new(0.05, 0.05, 0.05),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.9, 0.2, 0.1),
        )
        .with_opacity(0.9),
        Gaussian::new(
            Point3::new(0.08, 0.03, 3.0),
            Vector3::new(0.05, 0.05, 0.05),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.1, 0.3, 0.9),
        )
        .with_opacity(0.9),
    ]);
    let target = DifferentiableRasterizer::new(camera(), bg)
        .render(&truth)
        .color;

    let mut cloud = GaussianCloud::from_gaussians(vec![Gaussian::new(
        Point3::new(0.0, 0.0, 3.0),
        Vector3::new(0.1, 0.1, 0.1),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(0.5, 0.5, 0.5),
    )
    .with_opacity(0.5)]);

    let mut config = TrainingConfig::default();
    config.iterations = 600;
    config.densification.densification_interval = 100;
    config.densification.size_threshold = 0.05;
    config.opacity_reset.enabled = false;
    config.learning_rate_position = 2e-3;
    config.learning_rate_sh = 1e-2;
    let mut trainer = GaussianTrainer::new(DifferentiableRasterizer::new(camera(), bg), config);
    let losses = trainer.train(&mut cloud, &target);
    println!(
        "loss {:.4e} -> {:.4e}, {} gaussians",
        losses[0],
        losses.last().unwrap(),
        cloud.num_gaussians()
    );
    // 6 densify calls can at most double the cloud each time (64); the
    // gradient signal of two blobs should trigger at least one.
    assert!(cloud.num_gaussians() > 1 && cloud.num_gaussians() <= 64);
    assert!(*losses.last().unwrap() < 0.01 * losses[0]);
}
