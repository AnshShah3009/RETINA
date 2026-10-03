//! Reconstruction-quality metrics for SfM / MVS-style pipelines.
//!
//! All functions are pure and total: none of them panics, whatever the input.
//! What an empty or unnormalisable input returns depends on which way the metric
//! points, and the difference is deliberate:
//!
//! * **scores**, where larger is better (`registration_rate`, `f_score`), return
//!   `0.0` - their own worst value, so an input that was never measured cannot
//!   outrank one that was;
//! * **errors**, where smaller is better (`reprojection_rmse`,
//!   `rmse_over_extent`, `chamfer_distance`), return `NaN`, because `0.0` is a
//!   *perfect* error and would put "measured nothing" at the top of any ranking.
//!
//! # Example
//!
//! ```
//! use cv_eval::{chamfer_distance, f_score, registration_rate, reprojection_rmse, rmse_over_extent};
//!
//! assert!((registration_rate(90, 100) - 0.9).abs() < 1e-12);
//! assert!((reprojection_rmse(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
//! assert!((rmse_over_extent(0.5, 2.0) - 0.25).abs() < 1e-12);
//! assert_eq!(chamfer_distance(&[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]]), 0.0);
//! assert!((f_score(1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
//! ```

/// Fraction of supplied views that were successfully registered.
///
/// Returns `0.0` when `n_supplied == 0`.
pub fn registration_rate(n_registered: usize, n_supplied: usize) -> f64 {
    if n_supplied == 0 {
        0.0
    } else {
        n_registered as f64 / n_supplied as f64
    }
}

/// Root mean square of a set of residual magnitudes.
///
/// Returns `NaN` for an empty slice: there is no residual to square, and `0.0`
/// is the error of a perfect reconstruction.
pub fn reprojection_rmse(errors: &[f64]) -> f64 {
    if errors.is_empty() {
        return f64::NAN;
    }
    (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt()
}

/// Normalise an RMSE by the scene extent (a scale-free reconstruction error).
///
/// Returns `NaN` when `extent` is not a finite positive length - the ratio is
/// undefined for a scene with no extent (there is no scale to divide by), and
/// `0.0` would report a perfect scale-free error for a value that was never
/// normalised.
pub fn rmse_over_extent(rmse: f64, extent: f64) -> f64 {
    if !extent.is_finite() || extent <= 0.0 {
        f64::NAN
    } else {
        rmse / extent
    }
}

/// Symmetric, nearest-neighbour Chamfer distance between two point sets.
///
/// Returns `NaN` if either set is empty: with nothing to measure from the
/// distance is undefined, and `0.0` is the distance between two identical sets,
/// so an empty input would score as a perfect match. For a non-empty set
/// compared with a translated copy of itself the result equals the translation
/// magnitude.
pub fn chamfer_distance(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    0.5 * (mean_nearest(a, b) + mean_nearest(b, a))
}

/// F-score combining precision and recall with weighting `beta` (`beta = 1` is F1).
///
/// Returns `0.0` when the combination is degenerate.
pub fn f_score(precision: f64, recall: f64, beta: f64) -> f64 {
    // A non-finite input makes the whole expression NaN, and this module
    // documents itself as total: "empty inputs yield 0.0 instead of panicking or
    // returning NaN". `precision = inf` gives `inf / inf` here, which slipped
    // through the `denominator <= 0.0` guard because `inf` is not <= 0.
    if !precision.is_finite() || !recall.is_finite() || !beta.is_finite() {
        return 0.0;
    }
    let beta_sq = beta * beta;
    let denominator = beta_sq * precision + recall;
    if denominator <= 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * precision * recall / denominator
    }
}

fn mean_nearest(from: &[[f64; 3]], to: &[[f64; 3]]) -> f64 {
    let total: f64 = from
        .iter()
        .map(|p| {
            to.iter()
                .map(|q| squared_distance(p, q))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .sum();
    total / from.len() as f64
}

fn squared_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_rate_is_ratio() {
        assert!((registration_rate(75, 100) - 0.75).abs() < 1e-12);
        assert!((registration_rate(0, 10) - 0.0).abs() < 1e-12);
        assert_eq!(registration_rate(5, 0), 0.0);
    }

    #[test]
    fn reprojection_rmse_known_values() {
        assert!((reprojection_rmse(&[3.0, 4.0]) - 12.5_f64.sqrt()).abs() < 1e-12);
        assert!((reprojection_rmse(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-12);
        // No residuals is no measurement, not a perfect reconstruction.
        assert!(reprojection_rmse(&[]).is_nan());
    }

    #[test]
    fn rmse_over_extent_normalises() {
        assert!((rmse_over_extent(0.5, 2.0) - 0.25).abs() < 1e-12);
        // A scene with no extent has no scale to normalise by.
        assert!(rmse_over_extent(0.5, 0.0).is_nan());
        assert!(rmse_over_extent(0.5, -1.0).is_nan());
        assert!(rmse_over_extent(0.5, f64::INFINITY).is_nan());
        assert!(rmse_over_extent(0.5, f64::NAN).is_nan());
    }

    #[test]
    fn chamfer_self_is_zero() {
        let points = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]];
        assert_eq!(chamfer_distance(&points, &points), 0.0);
    }

    #[test]
    fn chamfer_translated_equals_translation() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // Translation small relative to the ~1.0 point spacing, so every point's
        // nearest neighbour is its own translated copy.
        let d = [0.1, 0.0, 0.0];
        let b: Vec<[f64; 3]> = a
            .iter()
            .map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
            .collect();

        let expected = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((chamfer_distance(&a, &b) - expected).abs() < 1e-12);
        // An empty set is not a perfect match.
        assert!(chamfer_distance(&a, &[]).is_nan());
        assert!(chamfer_distance(&[], &[]).is_nan());
    }

    #[test]
    fn f_score_known_values() {
        assert!((f_score(1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((f_score(0.5, 1.0, 1.0) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(f_score(0.0, 0.0, 1.0), 0.0);
        // beta -> 0 weights precision.
        assert!((f_score(0.4, 0.9, 0.0) - 0.4).abs() < 1e-12);
    }
}
