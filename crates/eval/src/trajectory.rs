//! Trajectory error metrics: ATE, RPE and their supporting types.
//!
//! The pose convention follows [`cv_core::Pose`]: a pose maps points from the
//! camera (local) frame into the world (parent) frame via `R * p + t`. The
//! camera center in world coordinates is therefore the pose translation.
//!
//! # Example
//!
//! ```
//! use cv_eval::{Alignment, Trajectory};
//! use nalgebra::{UnitQuaternion, Vector3};
//!
//! let positions = [
//!     Vector3::new(0.0, 0.0, 0.0),
//!     Vector3::new(1.0, 0.0, 0.0),
//!     Vector3::new(2.0, 0.0, 0.0),
//! ];
//! let quaternions = [UnitQuaternion::identity(); 3];
//! let gt = Trajectory::from_positions_and_quaternions(&positions, &quaternions);
//!
//! assert!((gt.path_length() - 2.0).abs() < 1e-12);
//! assert!(gt.ate(&gt, Alignment::Se3).rmse < 1e-9);
//! ```

use cv_core::Pose;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// Alignment mode used before computing absolute trajectory error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alignment {
    /// Compare the raw camera centers without any alignment.
    #[default]
    None,
    /// Align with a rigid SE(3) transform (Umeyama, scale forced to 1).
    Se3,
    /// Align with a similarity Sim(3) transform (Umeyama, scale estimated).
    Sim3,
}

/// A similarity transform `p -> s * R * p + t` recovered by Umeyama alignment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
    /// Rotation component.
    pub rotation: UnitQuaternion<f64>,
    /// Translation component.
    pub translation: Vector3<f64>,
    /// Uniform scale factor (`1.0` for rigid SE(3) alignment).
    pub scale: f64,
}

impl SimilarityTransform {
    /// The identity transform.
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Apply the transform to a 3D point.
    pub fn apply(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.scale * (self.rotation * *point) + self.translation
    }
}

impl Default for SimilarityTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Summary statistics of an error series (never panics, empty input yields zeros).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ErrorStats {
    /// Root mean square error.
    pub rmse: f64,
    /// Arithmetic mean of the errors.
    pub mean: f64,
    /// Median of the errors.
    pub median: f64,
    /// Maximum error.
    pub max: f64,
    /// Standard deviation of the errors.
    pub std: f64,
}

impl ErrorStats {
    /// Compute the summary statistics of `errors`.
    ///
    /// Returns all-zero statistics for an empty slice.
    pub fn from_errors(errors: &[f64]) -> Self {
        if errors.is_empty() {
            return Self::default();
        }
        let n = errors.len() as f64;
        let mean = errors.iter().sum::<f64>() / n;
        let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
        let max = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;

        let mut sorted = errors.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            0.5 * (sorted[mid - 1] + sorted[mid])
        };

        Self {
            rmse,
            mean,
            median,
            max,
            std: variance.sqrt(),
        }
    }
}

/// A timestamped sequence of poses.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    /// Poses, ordered by increasing timestamp.
    pub poses: Vec<Pose>,
    /// Timestamps, parallel to `poses`.
    pub timestamps: Vec<f64>,
}

impl Trajectory {
    /// Build a trajectory from poses and matching timestamps.
    ///
    /// If the lengths differ the shorter one wins (no panic).
    pub fn new(poses: Vec<Pose>, timestamps: Vec<f64>) -> Self {
        let n = poses.len().min(timestamps.len());
        let mut poses = poses;
        let mut timestamps = timestamps;
        poses.truncate(n);
        timestamps.truncate(n);
        Self { poses, timestamps }
    }

    /// Build a trajectory from poses, using the index as timestamp.
    pub fn from_poses(poses: &[Pose]) -> Self {
        Self {
            timestamps: (0..poses.len()).map(|i| i as f64).collect(),
            poses: poses.to_vec(),
        }
    }

    /// Build a trajectory from positions and quaternions, using the index as
    /// timestamp. The shorter of the two inputs wins (no panic).
    pub fn from_positions_and_quaternions(
        positions: &[Vector3<f64>],
        quaternions: &[UnitQuaternion<f64>],
    ) -> Self {
        let poses: Vec<Pose> = positions
            .iter()
            .zip(quaternions.iter())
            .map(|(p, q)| Pose::from_quat_translation(*q, *p))
            .collect();
        Self::from_poses(&poses)
    }

    /// Build a trajectory from positions, quaternions and explicit timestamps.
    pub fn from_positions_and_quaternions_with_timestamps(
        positions: &[Vector3<f64>],
        quaternions: &[UnitQuaternion<f64>],
        timestamps: &[f64],
    ) -> Self {
        let mut trajectory = Self::from_positions_and_quaternions(positions, quaternions);
        trajectory.timestamps = timestamps
            .iter()
            .take(trajectory.poses.len())
            .copied()
            .collect();
        trajectory
    }

    /// Number of poses.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Whether the trajectory has no poses.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Camera centers in world coordinates (the pose translations).
    pub fn camera_centers(&self) -> Vec<Vector3<f64>> {
        self.poses.iter().map(camera_center).collect()
    }

    /// Accumulated length of the camera-center polyline.
    pub fn path_length(&self) -> f64 {
        self.poses
            .windows(2)
            .map(|w| (camera_center(&w[1]) - camera_center(&w[0])).norm())
            .sum()
    }

    /// RMSE between the camera centers of `self` and `gt` (no alignment).
    ///
    /// Returns `0.0` for an empty overlap and ignores extra poses in the
    /// longer trajectory.
    pub fn camera_center_rmse(&self, gt: &Trajectory) -> f64 {
        let n = self.poses.len().min(gt.poses.len());
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self.poses[..n]
            .iter()
            .zip(gt.poses[..n].iter())
            .map(|(a, b)| (camera_center(a) - camera_center(b)).norm_squared())
            .sum();
        (sum_sq / n as f64).sqrt()
    }

    /// Absolute trajectory error against a ground-truth trajectory.
    ///
    /// Camera centers are optionally aligned (`align`) to the ground truth
    /// before the per-pose errors are measured. Returns zeroed statistics
    /// (and `transform: None`) when there is no overlapping pose.
    pub fn ate(&self, gt: &Trajectory, align: Alignment) -> AteResult {
        let n = self.poses.len().min(gt.poses.len());
        if n == 0 {
            return AteResult {
                rmse: 0.0,
                mean: 0.0,
                median: 0.0,
                max: 0.0,
                std: 0.0,
                errors: Vec::new(),
                alignment: align,
                scale: 1.0,
                transform: None,
            };
        }

        let est_centers: Vec<Vector3<f64>> = self.poses[..n].iter().map(camera_center).collect();
        let gt_centers: Vec<Vector3<f64>> = gt.poses[..n].iter().map(camera_center).collect();

        let transform = match align {
            Alignment::None => None,
            Alignment::Se3 => umeyama(&est_centers, &gt_centers, false),
            Alignment::Sim3 => umeyama(&est_centers, &gt_centers, true),
        };

        let errors: Vec<f64> = est_centers
            .iter()
            .zip(gt_centers.iter())
            .map(|(e, g)| {
                let aligned = match &transform {
                    Some(t) => t.apply(e),
                    None => *e,
                };
                (aligned - g).norm()
            })
            .collect();

        let stats = ErrorStats::from_errors(&errors);
        AteResult {
            rmse: stats.rmse,
            mean: stats.mean,
            median: stats.median,
            max: stats.max,
            std: stats.std,
            errors,
            alignment: align,
            scale: transform.map(|t| t.scale).unwrap_or(1.0),
            transform,
        }
    }

    /// Relative pose error over a fixed frame gap `delta_frames`.
    ///
    /// For every `i` with `i + delta_frames < n`, the relative motion
    /// `inv(T_i) * T_{i+delta}` is compared between `self` and `gt`. A global
    /// rigid transform between the two trajectories cancels out, so RPE is
    /// invariant to the (unobservable) world frame.
    pub fn rpe(&self, gt: &Trajectory, delta_frames: usize) -> RpeResult {
        let n = self.poses.len().min(gt.poses.len());
        let mut translation_errors = Vec::new();
        let mut rotation_errors = Vec::new();

        let mut i = 0;
        while i + delta_frames < n {
            let est_rel = self.poses[i]
                .inverse()
                .compose(&self.poses[i + delta_frames]);
            let gt_rel = gt.poses[i].inverse().compose(&gt.poses[i + delta_frames]);
            let error = gt_rel.inverse().compose(&est_rel);
            translation_errors.push(error.translation.norm());
            rotation_errors.push(error.rotation.angle());
            i += 1;
        }

        RpeResult {
            translation: ErrorStats::from_errors(&translation_errors),
            rotation: ErrorStats::from_errors(&rotation_errors),
            translation_errors,
            rotation_errors,
            delta_frames,
        }
    }
}

/// Result of [`Trajectory::ate`].
#[derive(Debug, Clone)]
pub struct AteResult {
    /// Root mean square of the per-pose errors.
    pub rmse: f64,
    /// Mean per-pose error.
    pub mean: f64,
    /// Median per-pose error.
    pub median: f64,
    /// Maximum per-pose error.
    pub max: f64,
    /// Standard deviation of the per-pose errors.
    pub std: f64,
    /// Per-pose errors, in trajectory order.
    pub errors: Vec<f64>,
    /// Alignment that was requested.
    pub alignment: Alignment,
    /// Scale recovered by the alignment (`1.0` for `None`/`Se3`).
    pub scale: f64,
    /// Transform applied before measuring the error (`None` for `Alignment::None`).
    pub transform: Option<SimilarityTransform>,
}

/// Result of [`Trajectory::rpe`].
#[derive(Debug, Clone)]
pub struct RpeResult {
    /// Statistics of the translational relative error.
    pub translation: ErrorStats,
    /// Statistics of the rotational relative error (radians).
    pub rotation: ErrorStats,
    /// Per-sample translational errors.
    pub translation_errors: Vec<f64>,
    /// Per-sample rotational errors (radians).
    pub rotation_errors: Vec<f64>,
    /// Frame gap used.
    pub delta_frames: usize,
}

fn camera_center(pose: &Pose) -> Vector3<f64> {
    pose.translation
}

/// Umeyama similarity/rigid alignment of `src` onto `dst`.
///
/// Returns `(s, R, t)` such that `dst ~= s * R * src + t`. With `with_scale`
/// set the uniform scale is estimated, otherwise `s = 1` (rigid SE(3)).
/// Returns `None` when there are no points.
fn umeyama(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    with_scale: bool,
) -> Option<SimilarityTransform> {
    let n = src.len().min(dst.len());
    if n == 0 {
        return None;
    }
    let nf = n as f64;

    let mut mean_src = Vector3::zeros();
    let mut mean_dst = Vector3::zeros();
    for (s, d) in src[..n].iter().zip(dst[..n].iter()) {
        mean_src += s;
        mean_dst += d;
    }
    mean_src /= nf;
    mean_dst /= nf;

    let mut covariance = Matrix3::zeros();
    let mut variance_src = 0.0;
    for (s, d) in src[..n].iter().zip(dst[..n].iter()) {
        let s = s - mean_src;
        let d = d - mean_dst;
        covariance += d * s.transpose();
        variance_src += s.norm_squared();
    }
    covariance /= nf;
    variance_src /= nf;

    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let singular = svd.singular_values;

    let mut correction = Matrix3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = u * correction * v_t;

    let scale = if with_scale && variance_src > 1e-12 {
        (singular[0] * correction[(0, 0)]
            + singular[1] * correction[(1, 1)]
            + singular[2] * correction[(2, 2)])
            / variance_src
    } else {
        1.0
    };

    let translation = mean_dst - scale * (rotation * mean_src);
    Some(SimilarityTransform {
        rotation: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
        translation,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn sample_trajectory() -> Trajectory {
        let positions = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        let quaternions: Vec<_> = (0..5)
            .map(|i| UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2 * i as f64))
            .collect();
        Trajectory::from_positions_and_quaternions(&positions, &quaternions)
    }

    fn known_transform() -> Pose {
        Pose::from_quat_translation(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7),
            Vector3::new(1.0, -2.0, 3.0),
        )
    }

    fn apply_rigid(gt: &Trajectory, t: &Pose) -> Trajectory {
        let poses: Vec<_> = gt.poses.iter().map(|p| t.compose(p)).collect();
        Trajectory::from_poses(&poses)
    }

    #[test]
    fn se3_alignment_yields_zero_ate() {
        let gt = sample_trajectory();
        let est = apply_rigid(&gt, &known_transform());

        let aligned = est.ate(&gt, Alignment::Se3);
        assert!(
            aligned.rmse < 1e-9,
            "aligned ATE should vanish, got {}",
            aligned.rmse
        );
        assert!(aligned.max < 1e-9);

        let raw = est.ate(&gt, Alignment::None);
        assert!(
            raw.rmse > 1e-3,
            "unaligned ATE should be non-zero, got {}",
            raw.rmse
        );
    }

    #[test]
    fn sim3_recovers_known_scale() {
        let gt = sample_trajectory();
        let scale = 2.5_f64;
        let poses: Vec<_> = gt
            .poses
            .iter()
            .map(|p| Pose::from_quat_translation(p.rotation, p.translation * scale))
            .collect();
        let est = Trajectory::from_poses(&poses);

        let sim3 = est.ate(&gt, Alignment::Sim3);
        assert!(
            sim3.rmse < 1e-9,
            "Sim(3) ATE should vanish, got {}",
            sim3.rmse
        );
        assert!(
            (sim3.scale - 1.0 / scale).abs() < 1e-9,
            "recovered scale {} should be {}",
            sim3.scale,
            1.0 / scale
        );

        let se3 = est.ate(&gt, Alignment::Se3);
        assert!(
            se3.rmse > 1e-3,
            "rigid alignment must not fix a scale change, got {}",
            se3.rmse
        );
    }

    #[test]
    fn rpe_of_rigidly_transformed_trajectory_is_zero() {
        let gt = sample_trajectory();
        let est = apply_rigid(&gt, &known_transform());

        let rpe = est.rpe(&gt, 1);
        assert!(
            rpe.translation.rmse < 1e-9,
            "translation RPE {}",
            rpe.translation.rmse
        );
        assert!(
            rpe.rotation.rmse < 1e-9,
            "rotation RPE {}",
            rpe.rotation.rmse
        );

        let self_rpe = gt.rpe(&gt, 2);
        assert!(self_rpe.translation.rmse < 1e-12);
        assert!(self_rpe.rotation.rmse < 1e-12);
    }

    #[test]
    fn camera_center_rmse_matches_known_shift() {
        let gt = sample_trajectory();
        let shift = Vector3::new(0.1, 0.0, 0.0);
        let poses: Vec<_> = gt
            .poses
            .iter()
            .map(|p| Pose::from_quat_translation(p.rotation, p.translation + shift))
            .collect();
        let est = Trajectory::from_poses(&poses);

        assert!((est.camera_center_rmse(&gt) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn path_length_and_empty_inputs() {
        let positions = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
        ];
        let quaternions = [UnitQuaternion::identity(); 3];
        let traj = Trajectory::from_positions_and_quaternions(&positions, &quaternions);
        assert!((traj.path_length() - 2.0).abs() < 1e-12);

        let empty = Trajectory::default();
        assert_eq!(empty.path_length(), 0.0);
        assert_eq!(empty.camera_center_rmse(&traj), 0.0);
        let ate = empty.ate(&traj, Alignment::Sim3);
        assert_eq!(ate.rmse, 0.0);
        assert!(ate.transform.is_none());
        assert_eq!(empty.rpe(&traj, 1).translation.rmse, 0.0);
    }
}
