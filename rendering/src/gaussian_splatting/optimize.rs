use nalgebra::{Point3, Vector3};
use rand::Rng;

use super::rasterize::DifferentiableRasterizer;
use super::types::{Gaussian, GaussianCloud};

#[derive(Clone, Debug)]
pub struct DensificationConfig {
    pub enabled: bool,
    pub grad_threshold: f32,
    pub size_threshold: f32,
    pub min_opacity: f32,
    pub densification_interval: usize,
    pub density_factor: f32,
    pub clone_threshold: f32,
    pub split_threshold: f32,
}

impl Default for DensificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            grad_threshold: 0.0002,
            size_threshold: 0.01,
            min_opacity: 0.005,
            densification_interval: 100,
            density_factor: 1.5,
            clone_threshold: 0.05,
            split_threshold: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PruningConfig {
    pub enabled: bool,
    pub split_threshold: f32,
    pub clone_threshold: f32,
    pub min_opacity: f32,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            split_threshold: 0.1,
            clone_threshold: 0.05,
            min_opacity: 0.005,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpacityResetConfig {
    pub enabled: bool,
    pub interval: usize,
    pub target: f32,
}

impl Default for OpacityResetConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 3000,
            target: 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub densification: DensificationConfig,
    pub pruning: PruningConfig,
    pub opacity_reset: OpacityResetConfig,
    pub learning_rate_position: f32,
    pub learning_rate_scale: f32,
    pub learning_rate_rotation: f32,
    pub learning_rate_opacity: f32,
    pub learning_rate_sh: f32,
    pub iterations: usize,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            densification: DensificationConfig::default(),
            pruning: PruningConfig::default(),
            opacity_reset: OpacityResetConfig::default(),
            learning_rate_position: 0.00016,
            learning_rate_scale: 0.005,
            learning_rate_rotation: 0.001,
            learning_rate_opacity: 0.05,
            learning_rate_sh: 0.0025,
            iterations: 30000,
            checkpoint_interval: 1000,
        }
    }
}

pub struct GaussianOptimizer {
    pub config: TrainingConfig,
    iteration: usize,
}

impl GaussianOptimizer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            iteration: 0,
        }
    }

    pub fn step(
        &mut self,
        cloud: &mut GaussianCloud,
        gradients: &[super::rasterize::GaussianGradient],
    ) {
        for (idx, grad) in gradients.iter().enumerate() {
            if idx >= cloud.gaussians.len() {
                break;
            }
            let gaussian = &mut cloud.gaussians[idx];

            gaussian.position.x -= self.config.learning_rate_position * grad.position_grad.x;
            gaussian.position.y -= self.config.learning_rate_position * grad.position_grad.y;
            gaussian.position.z -= self.config.learning_rate_position * grad.position_grad.z;

            gaussian.scale.x = (gaussian.scale.x
                - self.config.learning_rate_scale * grad.scale_grad.x)
                .max(0.0001);
            gaussian.scale.y = (gaussian.scale.y
                - self.config.learning_rate_scale * grad.scale_grad.y)
                .max(0.0001);
            gaussian.scale.z = (gaussian.scale.z
                - self.config.learning_rate_scale * grad.scale_grad.z)
                .max(0.0001);

            gaussian.rotation.x -= self.config.learning_rate_rotation * grad.rotation_grad.x;
            gaussian.rotation.y -= self.config.learning_rate_rotation * grad.rotation_grad.y;
            gaussian.rotation.z -= self.config.learning_rate_rotation * grad.rotation_grad.z;
            gaussian.rotation.w -= self.config.learning_rate_rotation * grad.rotation_grad.w;
            gaussian.rotation = gaussian.rotation.normalize();

            gaussian.opacity -= self.config.learning_rate_opacity * grad.opacity_grad;
            gaussian.opacity = gaussian.opacity.clamp(0.0, 1.0);
        }

        self.iteration += 1;

        if self.config.densification.enabled
            && self
                .iteration
                .is_multiple_of(self.config.densification.densification_interval)
        {
            self.densify(cloud);
        }

        if self.config.pruning.enabled {
            self.prune(cloud);
        }

        if self.config.opacity_reset.enabled
            && self
                .iteration
                .is_multiple_of(self.config.opacity_reset.interval)
        {
            self.reset_opacity(cloud);
        }
    }

    pub fn densify(&self, cloud: &mut GaussianCloud) {
        let mut new_gaussians: Vec<Gaussian> = Vec::new();
        let mut to_remove: Vec<usize> = Vec::new();
        let mut updates: Vec<(usize, (Vector3<f32>, f32))> = Vec::new();

        for (idx, gaussian) in cloud.gaussians.iter().enumerate() {
            let max_scale = gaussian.scale.x.max(gaussian.scale.y).max(gaussian.scale.z);

            if max_scale > self.config.densification.size_threshold {
                let mut rng = rand::thread_rng();
                let offset = Vector3::new(
                    rng.gen_range(-0.5..0.5) * max_scale,
                    rng.gen_range(-0.5..0.5) * max_scale,
                    rng.gen_range(-0.5..0.5) * max_scale,
                );

                let mut new_gaussian = gaussian.clone();
                new_gaussian.position = Point3::new(
                    gaussian.position.x + offset.x,
                    gaussian.position.y + offset.y,
                    gaussian.position.z + offset.z,
                );
                new_gaussian.opacity = gaussian.opacity * 0.5;
                new_gaussians.push(new_gaussian);

                let new_scale = gaussian.scale / self.config.densification.density_factor;
                updates.push((idx, (new_scale, gaussian.opacity * 0.5)));
            } else if max_scale < self.config.densification.size_threshold * 0.5 {
                let mut new_gaussian = gaussian.clone();
                new_gaussian.position = Point3::new(
                    gaussian.position.x + gaussian.scale.x,
                    gaussian.position.y,
                    gaussian.position.z,
                );
                new_gaussian.opacity = gaussian.opacity;
                new_gaussians.push(new_gaussian);

                let mut new_gaussian2 = gaussian.clone();
                new_gaussian2.position = Point3::new(
                    gaussian.position.x - gaussian.scale.x,
                    gaussian.position.y,
                    gaussian.position.z,
                );
                new_gaussians.push(new_gaussian2);

                to_remove.push(idx);
            }
        }

        for (idx, (new_scale, new_opacity)) in updates {
            if idx < cloud.gaussians.len() {
                cloud.gaussians[idx].scale = new_scale;
                cloud.gaussians[idx].opacity = new_opacity;
            }
        }

        for &idx in to_remove.iter().rev() {
            cloud.remove(idx);
        }

        for g in new_gaussians {
            cloud.push(g);
        }
    }

    pub fn prune(&self, cloud: &mut GaussianCloud) {
        cloud
            .gaussians
            .retain(|g| g.opacity > self.config.pruning.min_opacity);
        cloud.active_indices = (0..cloud.gaussians.len()).collect();
    }

    pub fn reset_opacity(&self, cloud: &mut GaussianCloud) {
        for gaussian in &mut cloud.gaussians {
            if gaussian.opacity < self.config.opacity_reset.target {
                gaussian.opacity = self.config.opacity_reset.target;
            }
        }
    }

    pub fn get_iteration(&self) -> usize {
        self.iteration
    }
}

pub struct GaussianTrainer {
    pub optimizer: GaussianOptimizer,
    rasterizer: DifferentiableRasterizer,
}

impl GaussianTrainer {
    pub fn new(rasterizer: DifferentiableRasterizer, config: TrainingConfig) -> Self {
        Self {
            optimizer: GaussianOptimizer::new(config),
            rasterizer,
        }
    }

    pub fn train_step(
        &mut self,
        cloud: &mut GaussianCloud,
        target: &[nalgebra::Vector3<f32>],
    ) -> f32 {
        let rendered = self.rasterizer.render(cloud);
        let loss = self.rasterizer.compute_loss(&rendered, target);
        let gradients = self.rasterizer.backward(cloud, &rendered, target);

        self.optimizer.step(cloud, &gradients);

        loss
    }

    pub fn train(
        &mut self,
        cloud: &mut GaussianCloud,
        target: &[nalgebra::Vector3<f32>],
    ) -> Vec<f32> {
        let mut losses = Vec::new();

        for _ in 0..self.optimizer.config.iterations {
            let loss = self.train_step(cloud, target);
            losses.push(loss);
        }

        losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;

    #[test]
    fn test_densification_config_default() {
        let config = DensificationConfig::default();
        assert!(config.enabled);
        assert!(config.grad_threshold > 0.0);
        assert!(config.size_threshold > 0.0);
        assert!(config.min_opacity > 0.0);
        assert!(config.densification_interval > 0);
    }

    #[test]
    fn test_pruning_config_default() {
        let config = PruningConfig::default();
        assert!(config.enabled);
        assert!(config.min_opacity > 0.0);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.learning_rate_position > 0.0);
        assert!(config.learning_rate_scale > 0.0);
        assert!(config.learning_rate_rotation > 0.0);
        assert!(config.learning_rate_opacity > 0.0);
        assert!(config.iterations > 0);
    }

    #[test]
    fn test_gaussian_optimizer_new() {
        let config = TrainingConfig::default();
        let optimizer = GaussianOptimizer::new(config);
        assert_eq!(optimizer.get_iteration(), 0);
    }

    #[test]
    fn test_gaussian_optimizer_prune() {
        let mut cloud = GaussianCloud::new();

        let g1 = Gaussian::new(
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.9);

        let g2 = Gaussian::new(
            Point3::new(2.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.001);

        cloud.push(g1);
        cloud.push(g2);

        let config = TrainingConfig::default();
        let optimizer = GaussianOptimizer::new(config);
        optimizer.prune(&mut cloud);

        assert_eq!(cloud.num_gaussians(), 1);
    }

    #[test]
    fn test_gaussian_optimizer_reset_opacity() {
        let mut cloud = GaussianCloud::new();

        let g1 = Gaussian::new(
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
        )
        .with_opacity(0.001);

        cloud.push(g1);

        let config = TrainingConfig::default();
        let optimizer = GaussianOptimizer::new(config);
        optimizer.reset_opacity(&mut cloud);

        assert!(
            cloud.gaussians[0].opacity >= optimizer.config.opacity_reset.target,
            "Opacity should be reset to at least the target value"
        );
    }
}
