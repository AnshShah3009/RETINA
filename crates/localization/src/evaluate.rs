//! Batch evaluation of localization results against ground-truth poses.

use crate::localizer::LocalizationResult;
use cv_core::Pose;

/// Aggregate error statistics over a batch of localization queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizationStats {
    /// Number of query/ground-truth pairs considered.
    pub queries: usize,
    /// Number of queries that produced a pose.
    pub succeeded: usize,
    /// Fraction of queries that produced a pose.
    ///
    /// `NaN` when there were no queries to evaluate, matching the error
    /// statistics below. A batch that asked something and failed everything
    /// reports `0.0`.
    pub success_rate: f64,
    /// Mean translation error over successful queries (`NaN` when none succeed).
    pub mean_translation_error: f64,
    /// Median translation error over successful queries (`NaN` when none succeed).
    pub median_translation_error: f64,
    /// Mean rotation error in degrees over successful queries (`NaN` when none succeed).
    pub mean_rotation_error_deg: f64,
    /// Median rotation error in degrees (`NaN` when none succeed).
    pub median_rotation_error_deg: f64,
}

/// Translation error `||a.t - b.t||` between two poses.
///
/// # Convention
///
/// This is the difference of the poses' own translation parameters, which for a
/// **world-to-camera** pose (`x_cam = R·x_world + t`, the convention
/// [`crate::LocalizationResult::pose`] uses) is *not* the distance between the
/// camera centres: `t = -R·C` depends on the rotation, so two estimates of the
/// same camera position at different orientations report a non-zero error, and
/// two poses at different positions can report zero. Measured for a camera at
/// `C = (1, 0, 0)`: 0.5176 at 30° of yaw, 1.4142 at 90°, both with a camera-centre
/// distance of 0.
///
/// Callers that want the camera-centre error — the TUM/absolute-trajectory
/// convention — should invert world-to-camera poses into camera-to-world poses
/// first, where `t` *is* the centre (`benchmark.rs` does exactly that before
/// calling [`evaluate_localization`]).
pub fn translation_error(a: &Pose, b: &Pose) -> f64 {
    (a.translation - b.translation).norm()
}

/// Rotation error between two poses, in degrees.
///
/// The angle of the relative rotation `bᵀa` is taken from the trace of its
/// rotation matrix and clamped for numerical safety, so it lies in `[0, 180]`.
pub fn rotation_error_degrees(a: &Pose, b: &Pose) -> f64 {
    let relative = b.rotation_matrix().transpose() * a.rotation_matrix();
    let trace = relative[(0, 0)] + relative[(1, 1)] + relative[(2, 2)];
    (((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
}

/// Aggregate `results` against `ground_truth` into [`LocalizationStats`].
///
/// A query counts as successful when `results[i]` is `Some` and a matching
/// ground-truth pose exists at `i`. Queries beyond the shorter of the two slices
/// are ignored. Error statistics cover successful queries only; when none
/// succeed they are `NaN`. `success_rate` is `succeeded / queries`, so it is
/// `NaN` when there are no queries at all (0/0 must not read as "everything
/// failed") and `0.0` when queries were attempted and none succeeded.
///
/// `results` and `ground_truth` must use the same pose convention; see
/// [`translation_error`] for what "translation error" means under each one.
pub fn evaluate_localization(
    results: &[Option<LocalizationResult>],
    ground_truth: &[Pose],
) -> LocalizationStats {
    let queries = results.len().min(ground_truth.len());

    let mut translation_errors = Vec::new();
    let mut rotation_errors = Vec::new();

    for (result, &truth) in results.iter().zip(ground_truth.iter()).take(queries) {
        // `Some` is not success, and the difference is the whole point of a
        // benchmark. A *stored* result can carry a pose with zero supporting
        // inliers; `Localizer::localize` never produces one, because RANSAC needs an
        // inlier and the acceptance floor is `min_matches`. Counting it anyway
        // inflates the success rate and folds an unsupported pose into the error
        // statistics.
        //
        // Measured before the fix, for a single zero-inlier result whose pose
        // happens to equal the ground truth:
        //
        //     (queries, succeeded, success_rate) = (1, 1, 1.0)
        //
        // A perfect score for a result that localizes nothing - the same failure
        // class as a registration reporting `fitness: 1.0` from a transform that
        // never moved, and worse here than elsewhere because a benchmark's number
        // is what every downstream comparison is made against. It errs *optimistic*,
        // so a method that localizes nothing scores as well as one that localizes
        // everything.
        //
        // `succeeded` is derived from these vectors' lengths below, so gating here
        // fixes the count, the rate and the means together.
        if let Some(result) = result.as_ref().filter(|r| r.success()) {
            translation_errors.push(translation_error(&result.pose, &truth));
            rotation_errors.push(rotation_error_degrees(&result.pose, &truth));
        }
    }

    let succeeded = translation_errors.len();

    LocalizationStats {
        queries,
        succeeded,
        // `succeeded / queries` is 0/0 = NaN for an empty batch: no query
        // succeeded because none was asked, which is not a 0% success rate.
        success_rate: succeeded as f64 / queries as f64,
        mean_translation_error: mean(&translation_errors),
        median_translation_error: median(&translation_errors),
        mean_rotation_error_deg: mean(&rotation_errors),
        median_rotation_error_deg: median(&rotation_errors),
    }
}

/// Arithmetic mean, or `NaN` for an empty slice.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Median (mean of the two middle values for even counts), or `NaN` when empty.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}
