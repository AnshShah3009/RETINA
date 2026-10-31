//! `GaussianOptimizer::densify` must not grow the cloud without a reason.
//!
//! 3DGS (Kerbl et al. 2023, §5.2) densifies only Gaussians whose accumulated
//! view-space positional gradient exceeds `grad_threshold`. `densify` used to
//! ignore `grad_threshold` and duplicate every Gaussian smaller than
//! `size_threshold * 0.5` on every call, so the count doubled per call.

use cv_rendering::gaussian_splatting::{
    Gaussian, GaussianCloud, GaussianOptimizer, TrainingConfig,
};
use nalgebra::{Point3, Vector3, Vector4};

fn gaussian(scale: f32) -> Gaussian {
    Gaussian::new(
        Point3::new(0.0, 0.0, 3.0),
        Vector3::new(scale, scale, scale),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
        Vector3::new(0.5, 0.5, 0.5),
    )
    .with_opacity(0.8)
}

#[test]
fn densify_without_gradient_signal_leaves_the_cloud_alone() {
    // One small, one mid-sized and one large splat (default size_threshold
    // 0.01), with no gradient ever accumulated.
    let mut cloud =
        GaussianCloud::from_gaussians(vec![gaussian(0.001), gaussian(0.007), gaussian(0.2)]);
    let mut optimizer = GaussianOptimizer::new(TrainingConfig::default());
    for call in 1..=10 {
        optimizer.densify(&mut cloud);
        println!("after densify #{call}: {} gaussians", cloud.num_gaussians());
    }
    assert_eq!(cloud.num_gaussians(), 3);
    let scales: Vec<f32> = cloud.gaussians.iter().map(|g| g.scale.x).collect();
    assert_eq!(scales, vec![0.001, 0.007, 0.2], "no splat may be resized");
}
