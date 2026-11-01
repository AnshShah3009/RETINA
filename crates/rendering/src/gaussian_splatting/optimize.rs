use nalgebra::{Point3, Vector3};
use rand::Rng;

use super::rasterize::{DifferentiableRasterizer, GaussianGradient};
use super::types::{Gaussian, GaussianCloud, unit_quaternion};

/// Adaptive density control settings. See [`GaussianOptimizer::densify`].
#[derive(Clone, Debug)]
pub struct DensificationConfig {
    pub enabled: bool,
    /// Average view-space (NDC) positional gradient norm at or above which a
    /// Gaussian is densified (reference `densify_grad_threshold`, 0.0002).
    pub grad_threshold: f32,
    /// World-space scale separating clone (`<=`) from split (`>`); the
    /// reference's `percent_dense * scene_extent`.
    pub size_threshold: f32,
    /// Not read by `densify`; pruning uses `PruningConfig::min_opacity`.
    pub min_opacity: f32,
    /// Densify every this many optimizer steps.
    pub densification_interval: usize,
    /// Scale divisor for split Gaussians (reference: `0.8 * N`, N = 2).
    pub density_factor: f32,
    /// Not read by `densify` (kept for compatibility).
    pub clone_threshold: f32,
    /// Not read by `densify` (kept for compatibility).
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
            density_factor: 1.6,
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

/// Optimizer state for one Gaussian: Adam moments and densification
/// statistics.
#[derive(Clone, Debug, Default)]
struct GaussianState {
    m: [f32; N_PARAMS],
    v: [f32; N_PARAMS],
    /// Sum of view-space positional gradient norms since the last densify.
    grad_accum: f32,
    /// Iterations in which the Gaussian was visible since the last densify.
    grad_count: u32,
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
/// The per-Gaussian state (Adam moments and densification statistics) is
/// kept index-aligned with `cloud.gaussians` across
/// [`prune`](Self::prune) and [`densify`](Self::densify). If the cloud is
/// edited between steps by other code and its length no longer matches, the
/// state is truncated or extended with zeros, which is exact for appended
/// Gaussians and a reset for anything else.
pub struct GaussianOptimizer {
    pub config: TrainingConfig,
    iteration: usize,
    state: Vec<GaussianState>,
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
            state: Vec::new(),
        }
    }

    fn sync_state(&mut self, n: usize) {
        self.state.resize(n, GaussianState::default());
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

    pub fn step(&mut self, cloud: &mut GaussianCloud, gradients: &[GaussianGradient]) {
        self.sync_state(cloud.gaussians.len());
        self.accumulate_densification_stats(gradients);
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

            let state = &mut self.state[idx];
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
    /// Add one iteration's view-space positional gradients to the
    /// densification statistics (INRIA `add_densification_stats`): for every
    /// visible Gaussian, `accum += |dL/d mean2d_ndc|` and `count += 1`.
    ///
    /// [`step`](Self::step) calls this; it is public so a caller driving its
    /// own loop can feed the statistics without an optimizer step.
    pub fn accumulate_densification_stats(&mut self, gradients: &[GaussianGradient]) {
        for grad in gradients {
            if !grad.visible {
                continue;
            }
            if grad.idx >= self.state.len() {
                self.state.resize(grad.idx + 1, GaussianState::default());
            }
            let norm = grad.mean2d_grad_ndc.norm();
            if norm.is_finite() {
                let s = &mut self.state[grad.idx];
                s.grad_accum += norm;
                s.grad_count += 1;
            }
        }
    }

    /// 3DGS adaptive density control (Kerbl et al. 2023, §5.2; INRIA
    /// `densify_and_clone` / `densify_and_split`).
    ///
    /// A Gaussian is selected when its average accumulated view-space
    /// positional gradient (see
    /// [`accumulate_densification_stats`](Self::accumulate_densification_stats))
    /// is at least `grad_threshold`. A selected Gaussian whose largest scale is
    /// at most `size_threshold` (the reference's `percent_dense * extent`) is
    /// under-reconstructing and is **cloned** unchanged; a larger one is
    /// over-reconstructing and is **split**: replaced by two Gaussians whose
    /// centres are sampled from the original Gaussian `N(mu, R S S^T R^T)` and
    /// whose scales are divided by `density_factor` (1.6 = 0.8 N for N = 2 in
    /// the reference). New Gaussians start with zero optimizer state, and all
    /// statistics are reset afterwards.
    ///
    /// Without accumulated gradient this is a no-op: it used to duplicate
    /// every Gaussian below `size_threshold / 2` on each call regardless of
    /// gradient, doubling the cloud per call.
    pub fn densify(&mut self, cloud: &mut GaussianCloud) {
        let n = cloud.gaussians.len();
        self.sync_state(n);
        let cfg = &self.config.densification;
        let threshold = cfg.grad_threshold;
        let size_threshold = cfg.size_threshold;
        let factor = cfg.density_factor;

        let mut rng = rand::rng();
        let mut normal = || {
            // Box-Muller; `1 - u` keeps the logarithm's argument in (0, 1].
            let u1: f32 = 1.0 - rng.random::<f32>();
            let u2: f32 = rng.random::<f32>();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };

        let mut keep = vec![true; n];
        let mut added: Vec<Gaussian> = Vec::new();
        for (i, g) in cloud.gaussians.iter().enumerate() {
            let s = &self.state[i];
            if s.grad_count == 0 {
                continue;
            }
            let avg = s.grad_accum / s.grad_count as f32;
            if avg < threshold {
                continue;
            }
            let max_scale = g.scale.x.max(g.scale.y).max(g.scale.z);
            if max_scale <= size_threshold {
                added.push(g.clone());
            } else {
                let rot = g.rotation_matrix();
                for _ in 0..2 {
                    let z = Vector3::new(normal(), normal(), normal());
                    let offset = rot * g.scale.component_mul(&z);
                    let mut sample = g.clone();
                    sample.position = Point3::from(g.position.coords + offset);
                    sample.scale = g.scale / factor;
                    added.push(sample);
                }
                keep[i] = false;
            }
        }

        self.retain(cloud, &keep);
        for g in added {
            cloud.push(g);
        }
        self.sync_state(cloud.gaussians.len());
        for s in &mut self.state {
            s.grad_accum = 0.0;
            s.grad_count = 0;
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
        self.state.retain(|_| {
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
        for (gaussian, state) in cloud.gaussians.iter_mut().zip(&mut self.state) {
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
