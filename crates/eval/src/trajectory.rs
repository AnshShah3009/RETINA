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

/// Summary statistics of an error series (never panics).
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
    /// Every field is `NaN` for an empty slice. There is no error to summarise,
    /// and a zero would be read as a perfect score by every comparison built on
    /// it (`if stats.rmse < best_so_far` selects the trajectory that was never
    /// measured). Callers that have to distinguish "no samples" from "a large
    /// error" can test `errors.is_empty()` on the error list they passed in.
    ///
    /// A slice holding a non-finite value is rejected the same way. `rmse`,
    /// `mean` and `std` propagated it on their own, but `max` did not: it is a
    /// `fold` with `f64::max`, and `f64::max` returns *the non-NaN operand*, so
    /// `max` walked straight past the NaN. Measured on `[1.0, NaN, 3.0]` the
    /// result was `mean = NaN, rmse = NaN, std = NaN` but `max = 3.0` - one
    /// plausible-looking number in a struct whose every other field says
    /// "unmeasurable", and one a caller would print as the worst error.
    /// The returned median is likewise `NaN` for such a slice, because the sort
    /// places the NaN at one end of the order.
    pub fn from_errors(errors: &[f64]) -> Self {
        if errors.is_empty() || !errors.iter().all(|e| e.is_finite()) {
            return Self {
                rmse: f64::NAN,
                mean: f64::NAN,
                median: f64::NAN,
                max: f64::NAN,
                std: f64::NAN,
            };
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
    ///
    /// `timestamps` is trimmed to the number of poses, and the poses are trimmed
    /// to the number of timestamps when the caller supplies fewer: the two
    /// fields are documented to be parallel, and leaving a short `timestamps`
    /// slice in place produced a trajectory whose `poses` and `timestamps` had
    /// different lengths, which every `zip` over them silently truncates.
    pub fn from_positions_and_quaternions_with_timestamps(
        positions: &[Vector3<f64>],
        quaternions: &[UnitQuaternion<f64>],
        timestamps: &[f64],
    ) -> Self {
        let trajectory = Self::from_positions_and_quaternions(positions, quaternions);
        Self::new(trajectory.poses, timestamps.to_vec())
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
    /// The comparison stops at the shorter trajectory. Returns `NaN` when the
    /// two share no pose: nothing was compared, and `0.0` would be read as a
    /// perfect match by any caller that thresholds or ranks the result.
    pub fn camera_center_rmse(&self, gt: &Trajectory) -> f64 {
        let n = self.poses.len().min(gt.poses.len());
        if n == 0 {
            return f64::NAN;
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
    /// before the per-pose errors are measured. The comparison stops at the
    /// shorter trajectory.
    ///
    /// When the two trajectories share no pose there is nothing to measure: the
    /// result carries `NaN` statistics, an empty `errors` list and no
    /// `transform`, with `is_valid` still `true` because the inputs themselves
    /// are well formed (see [`AteResult::is_valid`]). Reporting `rmse = 0.0`
    /// here made "compared nothing" rank as a perfect reconstruction.
    pub fn ate(&self, gt: &Trajectory, align: Alignment) -> AteResult {
        let n = self.poses.len().min(gt.poses.len());
        if n == 0 {
            return AteResult {
                rmse: f64::NAN,
                mean: f64::NAN,
                median: f64::NAN,
                max: f64::NAN,
                std: f64::NAN,
                errors: Vec::new(),
                is_valid: true,
                alignment: align,
                scale: f64::NAN,
                transform: None,
            };
        }

        let est_centers: Vec<Vector3<f64>> = self.poses[..n].iter().map(camera_center).collect();
        let gt_centers: Vec<Vector3<f64>> = gt.poses[..n].iter().map(camera_center).collect();

        // A non-finite coordinate produces `rmse = NaN` and every other statistic
        // NaN with it, which reads as a number and silently poisons any
        // comparison built on it.
        //
        // The dataset readers already reject `inf`/`nan`/`1e400` at parse time,
        // so this is not reachable from a file - it is reachable from a
        // `Trajectory` built directly in memory. Checked here because `ate`
        // returns a struct of plain `f64` with no error channel, so the only way
        // to report absence is to make it visible.
        if est_centers
            .iter()
            .chain(gt_centers.iter())
            .any(|c| !c.iter().all(|v| v.is_finite()))
        {
            return AteResult {
                rmse: f64::NAN,
                mean: f64::NAN,
                median: f64::NAN,
                max: f64::NAN,
                std: f64::NAN,
                errors: Vec::new(),
                is_valid: false,
                alignment: align,
                scale: f64::NAN,
                transform: None,
            };
        }

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
            is_valid: true,
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
    ///
    /// Returns empty error lists and `NaN` statistics when there is no pair to
    /// compare: `delta_frames == 0` (every pose against itself, which is the
    /// identity for *any* pair of trajectories) or `delta_frames >= n`.
    pub fn rpe(&self, gt: &Trajectory, delta_frames: usize) -> RpeResult {
        let n = self.poses.len().min(gt.poses.len());
        let mut translation_errors = Vec::new();
        let mut rotation_errors = Vec::new();

        // `delta_frames == 0` would compare each pose with itself: `inv(T_i) *
        // T_i` is the identity regardless of what the two trajectories contain,
        // so the loop below used to emit `n` exact zeros and report a perfect
        // RPE (translation and rotation rmse of 0.0) for trajectories that share
        // nothing. Only a positive gap compares distinct poses.
        if delta_frames > 0 {
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
    ///
    /// Empty when there is no overlapping pose (then every statistic above is
    /// `NaN`), or when the result is not valid - see [`AteResult::is_valid`].
    pub errors: Vec<f64>,
    /// `false` when the input contained a non-finite coordinate, so every
    /// statistic above is NaN.
    ///
    /// `ate` has no error channel and returns a struct of plain `f64`, so
    /// without this a malformed trajectory is indistinguishable from one that
    /// simply registered poorly: both report `rmse = NaN`.
    ///
    /// This flag reports the *input*: a pair of well-formed trajectories that
    /// share no pose is still valid, but carries `NaN` statistics because
    /// nothing was measured, so a caller must not read those numbers as an
    /// error of zero. Check `errors.is_empty()` or `rmse.is_nan()` for that.
    pub is_valid: bool,
    /// Alignment that was requested.
    pub alignment: Alignment,
    /// Scale recovered by the alignment (`1.0` for `None`/`Se3`), or `NaN` when
    /// nothing was aligned because there is no overlapping pose.
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
        // A sum over no segments genuinely is zero.
        assert_eq!(empty.path_length(), 0.0);

        // The rest are measurements, and a measurement over zero poses has no
        // value to report. A zero reads as a perfect match to any caller that
        // thresholds or ranks it; NaN is the honest "no data".
        assert!(
            empty.camera_center_rmse(&traj).is_nan(),
            "camera-centre RMSE over no overlap = {}",
            empty.camera_center_rmse(&traj)
        );
        let ate = empty.ate(&traj, Alignment::Sim3);
        assert!(ate.rmse.is_nan(), "ATE over no overlap = {}", ate.rmse);
        assert!(ate.is_valid, "an empty trajectory is a well-formed input");
        assert!(ate.errors.is_empty());
        assert!(ate.transform.is_none());
        let rpe = empty.rpe(&traj, 1);
        assert!(rpe.translation.rmse.is_nan());
        assert!(rpe.rotation.rmse.is_nan());
    }

    #[test]
    fn error_stats_of_no_samples_are_nan() {
        let stats = ErrorStats::from_errors(&[]);
        assert!(stats.rmse.is_nan(), "rmse = {}", stats.rmse);
        assert!(stats.mean.is_nan());
        assert!(stats.median.is_nan());
        assert!(stats.max.is_nan());
        assert!(stats.std.is_nan());

        // Control: one sample is summarised exactly (a zero standard deviation
        // here is computed from the data, not fabricated).
        let one = ErrorStats::from_errors(&[3.0]);
        assert_eq!(one.rmse, 3.0);
        assert_eq!(one.mean, 3.0);
        assert_eq!(one.median, 3.0);
        assert_eq!(one.max, 3.0);
        assert_eq!(one.std, 0.0);
    }

    #[test]
    fn rpe_with_zero_frame_gap_reports_no_measurement() {
        // Ground truth moves one metre per frame; the estimate also drifts in y,
        // so the frame-to-frame motion genuinely differs.
        let gt = Trajectory::from_positions_and_quaternions(
            &(0..6)
                .map(|i| Vector3::new(i as f64, 0.0, 0.0))
                .collect::<Vec<_>>(),
            &[UnitQuaternion::identity(); 6],
        );
        let est = Trajectory::from_positions_and_quaternions(
            &(0..6)
                .map(|i| Vector3::new(i as f64, 0.5 * i as f64, 0.0))
                .collect::<Vec<_>>(),
            &[UnitQuaternion::identity(); 6],
        );

        // A zero gap compares every pose with itself, which is the identity for
        // *any* pair of trajectories: it must not report a perfect 0.0.
        let zero = est.rpe(&gt, 0);
        assert!(
            zero.translation_errors.is_empty(),
            "delta 0 fabricated {} samples: {:?}",
            zero.translation_errors.len(),
            zero.translation_errors
        );
        assert!(zero.rotation_errors.is_empty());
        assert!(zero.translation.rmse.is_nan(), "{}", zero.translation.rmse);
        assert!(zero.rotation.rmse.is_nan());

        // Control: a positive gap measures the real discrepancy, and a gap at
        // the last possible index still has one sample.
        let one = est.rpe(&gt, 1);
        assert_eq!(one.translation_errors.len(), 5);
        assert!(
            one.translation.rmse > 0.4 && one.translation.rmse.is_finite(),
            "delta 1 translation rmse = {}",
            one.translation.rmse
        );
        let last = est.rpe(&gt, 5);
        assert_eq!(last.translation_errors.len(), 1);
        assert!(last.translation.rmse.is_finite());

        // A gap beyond the trajectory has no pair to compare either.
        let beyond = est.rpe(&gt, 6);
        assert!(beyond.translation_errors.is_empty());
        assert!(beyond.translation.rmse.is_nan());
    }

    #[test]
    fn timestamps_stay_parallel_to_poses() {
        let positions: Vec<Vector3<f64>> =
            (0..4).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect();
        let quaternions = [UnitQuaternion::identity(); 4];

        let full = Trajectory::from_positions_and_quaternions_with_timestamps(
            &positions,
            &quaternions,
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(full.poses.len(), 4);
        assert_eq!(full.timestamps, vec![1.0, 2.0, 3.0, 4.0]);

        // Too few timestamps: the shorter input wins, as `Trajectory::new`
        // documents, rather than leaving the two fields out of step.
        let short = Trajectory::from_positions_and_quaternions_with_timestamps(
            &positions,
            &quaternions,
            &[1.0, 2.0],
        );
        assert_eq!(short.poses.len(), short.timestamps.len());
        assert_eq!(short.timestamps, vec![1.0, 2.0]);

        // Too many: trimmed to the poses.
        let long = Trajectory::from_positions_and_quaternions_with_timestamps(
            &positions,
            &quaternions,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert_eq!(long.poses.len(), long.timestamps.len());
        assert_eq!(long.timestamps.len(), 4);
    }
}
