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
    /// Fraction of queries that produced a pose (`0.0` when there are none).
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
/// succeed they are `NaN` (`success_rate` is still reported as `0.0`).
pub fn evaluate_localization(
    results: &[Option<LocalizationResult>],
    ground_truth: &[Pose],
) -> LocalizationStats {
    let queries = results.len().min(ground_truth.len());

    let mut translation_errors = Vec::new();
    let mut rotation_errors = Vec::new();

    for (result, &truth) in results.iter().zip(ground_truth.iter()).take(queries) {
        if let Some(result) = result {
            translation_errors.push(translation_error(&result.pose, &truth));
            rotation_errors.push(rotation_error_degrees(&result.pose, &truth));
        }
    }

    let succeeded = translation_errors.len();

    LocalizationStats {
        queries,
        succeeded,
        success_rate: if queries == 0 {
            0.0
        } else {
            succeeded as f64 / queries as f64
        },
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
