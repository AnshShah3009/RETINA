use nalgebra::{Point3, Vector3};
use rand::Rng;

use super::rasterize::DifferentiableRasterizer;
use super::types::{unit_quaternion, Gaussian, GaussianCloud};

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

/// Number of optimised scalars per Gaussian: position 3, log-scale 3,
/// rotation 4, logit-opacity 1, DC colour 3.
const N_PARAMS: usize = 14;

/// Adam moments for one Gaussian.
#[derive(Clone, Debug, Default)]
struct AdamState {
    m: [f32; N_PARAMS],
    v: [f32; N_PARAMS],
}

/// 3DGS training optimizer: Adam (Kerbl et al. 2023, §7) over the activated
/// parametrisation the reference uses - position, **log** scale, rotation,
/// **logit** opacity and the colour - plus densification, pruning and opacity
/// reset.
///
/// The gradients it consumes ([`GaussianGradient`]) are with respect to the
/// stored parameters (`scale`, `opacity`); they are chain-ruled into log /
/// logit space here (`dL/dlog s = s dL/ds`, `dL/dlogit o = o (1 - o) dL/do`),
/// so a learning rate means the same thing as in the reference: a relative
/// step for scale, and a step in logit space for opacity.
///
/// The Adam state is kept index-aligned with `cloud.gaussians` across
/// [`prune`](Self::prune) and [`densify`](Self::densify). If the cloud is
/// edited between steps by other code and its length no longer matches, the
/// state is truncated or extended with zeros, which is exact for appended
/// Gaussians and a reset for anything else.
pub struct GaussianOptimizer {
    pub config: TrainingConfig,
    iteration: usize,
    adam: Vec<AdamState>,
}

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
/// The reference implementation's `eps=1e-15`.
const ADAM_EPS: f32 = 1e-15;

fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl GaussianOptimizer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            iteration: 0,
            adam: Vec::new(),
        }
    }

    fn sync_state(&mut self, n: usize) {
        self.adam.resize(n, AdamState::default());
    }

    /// Learning rate of each of the [`N_PARAMS`] slots.
    fn learning_rates(&self) -> [f32; N_PARAMS] {
        let c = &self.config;
        let mut lr = [0.0; N_PARAMS];
        lr[0..3].fill(c.learning_rate_position);
        lr[3..6].fill(c.learning_rate_scale);
        lr[6..10].fill(c.learning_rate_rotation);
        lr[10] = c.learning_rate_opacity;
        lr[11..14].fill(c.learning_rate_sh);
        lr
    }

    pub fn step(
        &mut self,
        cloud: &mut GaussianCloud,
        gradients: &[super::rasterize::GaussianGradient],
    ) {
        self.sync_state(cloud.gaussians.len());
        let lr = self.learning_rates();
        let t = (self.iteration + 1) as i32;
        let bias1 = 1.0 - ADAM_BETA1.powi(t);
        let bias2 = 1.0 - ADAM_BETA2.powi(t);

        for grad in gradients {
            let idx = grad.idx;
            if idx >= cloud.gaussians.len() {
                continue;
            }
            let gaussian = &mut cloud.gaussians[idx];

            // Gradient in the optimised parametrisation.
            let mut g = [0.0f32; N_PARAMS];
            g[0..3].copy_from_slice(grad.position_grad.as_slice());
            for k in 0..3 {
                g[3 + k] = grad.scale_grad[k] * gaussian.scale[k];
            }
            g[6..10].copy_from_slice(grad.rotation_grad.as_slice());
            let o = gaussian.opacity.clamp(1e-6, 1.0 - 1e-6);
            g[10] = grad.opacity_grad * o * (1.0 - o);
            g[11..14].copy_from_slice(grad.color_grad.as_slice());

            let state = &mut self.adam[idx];
            let mut step = [0.0f32; N_PARAMS];
            for k in 0..N_PARAMS {
                if !g[k].is_finite() {
                    continue;
                }
                state.m[k] = ADAM_BETA1 * state.m[k] + (1.0 - ADAM_BETA1) * g[k];
                state.v[k] = ADAM_BETA2 * state.v[k] + (1.0 - ADAM_BETA2) * g[k] * g[k];
                let m_hat = state.m[k] / bias1;
                let v_hat = state.v[k] / bias2;
                step[k] = lr[k] * m_hat / (v_hat.sqrt() + ADAM_EPS);
            }

            for k in 0..3 {
                gaussian.position[k] -= step[k];
                gaussian.scale[k] =
                    (gaussian.scale[k].max(f32::MIN_POSITIVE).ln() - step[3 + k]).exp();
            }
            for k in 0..4 {
                gaussian.rotation[k] -= step[6 + k];
            }
            // Not an unconditional `normalize()`: that is `0/0` for a zero-norm
            // quaternion and writes NaN into a public field. `unit_quaternion`
            // is the crate's existing decision on the same question.
            gaussian.rotation = unit_quaternion(&gaussian.rotation);

            gaussian.opacity = sigmoid(logit(gaussian.opacity) - step[10]);

            let coeffs = &mut gaussian.spherical_harmonics.coeffs;
            if coeffs.len() < 3 {
                coeffs.resize(3, 0.0);
            }
            for c in 0..3 {
                coeffs[c] -= step[11 + c];
            }
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
                let mut rng = rand::rng();
                let offset = Vector3::new(
                    rng.random_range(-0.5..0.5) * max_scale,
                    rng.random_range(-0.5..0.5) * max_scale,
                    rng.random_range(-0.5..0.5) * max_scale,
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

    /// Remove every Gaussian whose opacity is at or below
    /// `pruning.min_opacity`, keeping the optimizer state aligned.
    pub fn prune(&mut self, cloud: &mut GaussianCloud) {
        self.sync_state(cloud.gaussians.len());
        let min = self.config.pruning.min_opacity;
        let keep: Vec<bool> = cloud.gaussians.iter().map(|g| g.opacity > min).collect();
        self.retain(cloud, &keep);
    }

    /// Keep the Gaussians (and their optimizer state) whose `keep` flag is set.
    fn retain(&mut self, cloud: &mut GaussianCloud, keep: &[bool]) {
        let mut i = 0;
        cloud.gaussians.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut i = 0;
        self.adam.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        cloud.active_indices = (0..cloud.gaussians.len()).collect();
    }

    /// 3DGS opacity reset (Kerbl et al. 2023, §5.2): *cap* every opacity at
    /// `target` so that floaters must re-earn their opacity and the ones that
    /// cannot are pruned. This used to *raise* opacities below the target and
    /// leave opaque ones alone - the opposite: `[0.9, 0.001]` became
    /// `[0.9, 0.01]`, reviving near-dead splats and resetting nothing.
    ///
    /// As in the reference (`replace_tensor_to_optimizer`), the opacity's Adam
    /// moments are reset with it.
    pub fn reset_opacity(&mut self, cloud: &mut GaussianCloud) {
        self.sync_state(cloud.gaussians.len());
        let target = self.config.opacity_reset.target;
        for (gaussian, state) in cloud.gaussians.iter_mut().zip(&mut self.adam) {
            gaussian.opacity = gaussian.opacity.min(target);
            state.m[10] = 0.0;
            state.v[10] = 0.0;
        }
    }

    pub fn get_iteration(&self) -> usize {
        self.iteration
    }
}

/// How a [`GaussianTrainer::train_with_report`] run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingStatus {
    /// The loss reached zero.
    Converged,
    /// The gradient was identically zero (or not finite) while the loss was
    /// positive: no parameter can change the rendered image, so further
    /// iterations cannot make progress. This is a failure, not convergence -
    /// e.g. every splat is behind the camera or outside the image.
    Stalled,
    /// All configured iterations ran.
    Completed,
}

/// Losses and outcome of a training run.
#[derive(Clone, Debug)]
pub struct TrainingReport {
    /// Loss before each optimizer step.
    pub losses: Vec<f32>,
    pub status: TrainingStatus,
    /// Euclidean norm over all Gaussians' parameter gradients at the last step.
    pub final_gradient_norm: f32,
}

pub struct GaussianTrainer {
    pub optimizer: GaussianOptimizer,
    rasterizer: DifferentiableRasterizer,
    last_gradient_norm: f32,
}

impl GaussianTrainer {
    pub fn new(rasterizer: DifferentiableRasterizer, config: TrainingConfig) -> Self {
        Self {
            optimizer: GaussianOptimizer::new(config),
            rasterizer,
            last_gradient_norm: 0.0,
        }
    }

    /// One render / loss / backward / optimizer step. Returns the loss before
    /// the step.
    pub fn train_step(
        &mut self,
        cloud: &mut GaussianCloud,
        target: &[nalgebra::Vector3<f32>],
    ) -> f32 {
        let rendered = self.rasterizer.render(cloud);
        let loss = self.rasterizer.compute_loss(&rendered, target);
        let gradients = self.rasterizer.backward(cloud, &rendered, target);
        self.last_gradient_norm = gradients
            .iter()
            .map(|g| g.norm_squared())
            .sum::<f32>()
            .sqrt();

        self.optimizer.step(cloud, &gradients);

        loss
    }

    /// Gradient norm of the most recent [`train_step`](Self::train_step).
    pub fn last_gradient_norm(&self) -> f32 {
        self.last_gradient_norm
    }

    /// Run `config.iterations` steps and return the per-step losses. See
    /// [`train_with_report`](Self::train_with_report) for a run that also says
    /// whether it stalled.
    pub fn train(
        &mut self,
        cloud: &mut GaussianCloud,
        target: &[nalgebra::Vector3<f32>],
    ) -> Vec<f32> {
        self.train_with_report(cloud, target).losses
    }

    /// Run up to `config.iterations` steps, stopping early when the loss
    /// reaches zero ([`TrainingStatus::Converged`]) or when the gradient is
    /// identically zero with a positive loss ([`TrainingStatus::Stalled`]).
    pub fn train_with_report(
        &mut self,
        cloud: &mut GaussianCloud,
        target: &[nalgebra::Vector3<f32>],
    ) -> TrainingReport {
        let mut losses = Vec::new();
        let mut status = TrainingStatus::Completed;

        for _ in 0..self.optimizer.config.iterations {
            let loss = self.train_step(cloud, target);
            losses.push(loss);
            if loss == 0.0 {
                status = TrainingStatus::Converged;
                break;
            }
            let g = self.last_gradient_norm;
            if !loss.is_finite() || !g.is_finite() || g == 0.0 {
                status = TrainingStatus::Stalled;
                break;
            }
        }

        TrainingReport {
            losses,
            status,
            final_gradient_norm: self.last_gradient_norm,
        }
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
        let mut optimizer = GaussianOptimizer::new(config);
        optimizer.prune(&mut cloud);

        assert_eq!(cloud.num_gaussians(), 1);
    }

    #[test]
    fn test_gaussian_optimizer_reset_opacity() {
        // Opacity reset caps opacities at the target; it does not raise them.
        let mut cloud = GaussianCloud::new();
        for opacity in [0.9, 0.001] {
            cloud.push(
                Gaussian::new(
                    Point3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.1, 0.1, 0.1),
                    Vector4::new(0.0, 0.0, 0.0, 1.0),
                    Vector3::new(0.5, 0.5, 0.5),
                )
                .with_opacity(opacity),
            );
        }

        let mut optimizer = GaussianOptimizer::new(TrainingConfig::default());
        let target = optimizer.config.opacity_reset.target;
        optimizer.reset_opacity(&mut cloud);

        assert_eq!(
            cloud.gaussians[0].opacity, target,
            "opaque splat must drop to the target"
        );
        assert_eq!(
            cloud.gaussians[1].opacity, 0.001,
            "faint splat must not be revived"
        );
    }
}
