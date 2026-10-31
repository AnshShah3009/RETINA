//! RANSAC (Random Sample Consensus) for geometric verification
//!
//! RANSAC is used to robustly estimate geometric transformations
//! (homography, fundamental matrix) from feature matches with outliers.

use cv_core::{Matches, Ransac, RobustConfig, RobustModel};
use nalgebra::{Matrix3, Vector3};

/// RANSAC configuration — re-exported from [`cv_core::RobustConfig`].
pub type RansacConfig = RobustConfig;

/// A correspondence between a source and destination 2-D point.
#[derive(Clone, Debug)]
pub struct MatchPair {
    /// Source point (x, y) in the query image.
    pub src: (f64, f64),
    /// Destination point (x, y) in the training image.
    pub dst: (f64, f64),
}

/// RANSAC-compatible estimator for projective homographies (DLT, 4-point minimum).
pub struct HomographyEstimator;

impl RobustModel<MatchPair> for HomographyEstimator {
    type Model = Matrix3<f64>;

    fn min_sample_size(&self) -> usize {
        4
    }

    fn estimate(&self, data: &[&MatchPair]) -> Option<Self::Model> {
        let src: Vec<[f64; 2]> = data.iter().map(|m| [m.src.0, m.src.1]).collect();
        let dst: Vec<[f64; 2]> = data.iter().map(|m| [m.dst.0, m.dst.1]).collect();
        // Normalised DLT, shared with `cv-calib3d`. This estimator previously
        // solved the raw (unnormalised) system, which loses accuracy for
        // large pixel coordinates.
        cv_calib3d::dlt::solve_dlt_homography(&src, &dst)
    }

    fn compute_error(&self, model: &Self::Model, data: &MatchPair) -> f64 {
        let p1 = Vector3::new(data.src.0, data.src.1, 1.0);
        let p2_pred = model * p1;
        if p2_pred[2].abs() > 1e-10 {
            let x2_pred = p2_pred[0] / p2_pred[2];
            let y2_pred = p2_pred[1] / p2_pred[2];
            ((x2_pred - data.dst.0).powi(2) + (y2_pred - data.dst.1).powi(2)).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// RANSAC-compatible estimator for fundamental matrices (8-point algorithm with rank-2 enforcement).
pub struct FundamentalEstimator;

impl RobustModel<MatchPair> for FundamentalEstimator {
    type Model = Matrix3<f64>;

    fn min_sample_size(&self) -> usize {
        8
    }

    fn estimate(&self, data: &[&MatchPair]) -> Option<Self::Model> {
        let pts1: Vec<[f64; 2]> = data.iter().map(|m| [m.src.0, m.src.1]).collect();
        let pts2: Vec<[f64; 2]> = data.iter().map(|m| [m.dst.0, m.dst.1]).collect();
        // Normalised 8-point algorithm, shared with `cv-calib3d` (this
        // estimator previously solved the raw system and enforced rank 2 by
        // recomposition).
        cv_calib3d::dlt::solve_dlt_fundamental(&pts1, &pts2)
    }

    fn compute_error(&self, model: &Self::Model, data: &MatchPair) -> f64 {
        let p1 = Vector3::new(data.src.0, data.src.1, 1.0);
        let p2 = Vector3::new(data.dst.0, data.dst.1, 1.0);
        let l = model * p1;
        let denom = l[0].powi(2) + l[1].powi(2);
        if denom > 1e-10 {
            (p2.dot(&l)).abs() / denom.sqrt()
        } else {
            f64::INFINITY
        }
    }
}

pub use cv_core::robust::RobustResult as RansacResult;

/// Lift descriptor matches to (source, destination) point pairs.
///
/// `query_idx` / `train_idx` are `i32` indices produced by a descriptor
/// matcher, so a malformed or truncated match set indexes past the end of the
/// point arrays. This used to `as usize` and index directly, which panics on a
/// negative index (wrapping to a huge offset) or on an index past the end — a
/// panic in the middle of a calibration pipeline for input that is merely
/// *wrong*, not a programming error. The pattern here matches
/// `cv_localization::Localizer::correspondences`: `usize::try_from` plus `.get()`,
/// and an unusable match is skipped.
///
/// Returns the pairs plus, for each, the index of the match it came from, so the
/// caller can map the per-correspondence inlier flags back onto the original
/// match list. Without that mapping a single skipped match would shift every
/// later inlier flag by one and `filter_matches_by_inliers` would keep the wrong
/// matches.
fn collect_pairs(
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> (Vec<MatchPair>, Vec<usize>) {
    let mut data = Vec::with_capacity(matches.matches.len());
    let mut origin = Vec::with_capacity(matches.matches.len());

    for (i, m) in matches.matches.iter().enumerate() {
        let Some(query_idx) = usize::try_from(m.query_idx).ok() else {
            continue;
        };
        let Some(train_idx) = usize::try_from(m.train_idx).ok() else {
            continue;
        };
        let (Some(&src), Some(&dst)) = (src_points.get(query_idx), dst_points.get(train_idx))
        else {
            continue;
        };

        data.push(MatchPair { src, dst });
        origin.push(i);
    }

    (data, origin)
}

/// Re-index a RANSAC inlier mask from `collect_pairs` order back to match order.
fn scatter_inliers(inliers: &[bool], origin: &[usize], n_matches: usize) -> Vec<bool> {
    let mut scattered = vec![false; n_matches];
    for (pair_idx, &match_idx) in origin.iter().enumerate() {
        scattered[match_idx] = inliers.get(pair_idx).copied().unwrap_or(false);
    }
    scattered
}

/// Run RANSAC over the usable correspondences of `matches`, returning inlier
/// flags aligned with `matches.matches` (invalid matches are never inliers).
fn run_ransac<E>(
    estimator: &E,
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    config: &RansacConfig,
) -> RansacResult<Matrix3<f64>>
where
    E: RobustModel<MatchPair, Model = Matrix3<f64>>,
{
    let (data, origin) = collect_pairs(matches, src_points, dst_points);
    let ransac = Ransac::new(config.clone());
    let mut result = ransac.run(estimator, &data);
    result.inliers = scatter_inliers(&result.inliers, &origin, matches.matches.len());
    result
}

/// Estimate homography using RANSAC
///
/// Matches whose `query_idx` / `train_idx` do not index a point in both
/// argument arrays are skipped rather than panicking, and the returned inlier
/// flags stay parallel to `matches.matches`.
pub fn estimate_homography(
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    config: &RansacConfig,
) -> RansacResult<Matrix3<f64>> {
    run_ransac(
        &HomographyEstimator,
        matches,
        src_points,
        dst_points,
        config,
    )
}

/// Estimate fundamental matrix using RANSAC
///
/// Matches whose `query_idx` / `train_idx` do not index a point in both
/// argument arrays are skipped rather than panicking, and the returned inlier
/// flags stay parallel to `matches.matches`.
pub fn estimate_fundamental(
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    config: &RansacConfig,
) -> RansacResult<Matrix3<f64>> {
    run_ransac(
        &FundamentalEstimator,
        matches,
        src_points,
        dst_points,
        config,
    )
}

/// Filter matches to keep only inliers
pub fn filter_matches_by_inliers(matches: &Matches, inliers: &[bool]) -> Matches {
    let mut filtered = Matches::new();

    for (i, m) in matches.matches.iter().enumerate() {
        if i < inliers.len() && inliers[i] {
            filtered.push(*m);
        }
    }

    filtered
}

/// Convenience wrapper that runs RANSAC and returns the geometrically consistent subset of matches.
pub struct RansacMatcher {
    config: RansacConfig,
    use_fundamental: bool,
}

impl Default for RansacMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl RansacMatcher {
    /// Create a new `RansacMatcher` using the default config and homography estimation.
    pub fn new() -> Self {
        Self {
            config: RansacConfig::default(),
            use_fundamental: false,
        }
    }

    /// Switch to fundamental matrix estimation instead of homography.
    pub fn with_fundamental(mut self) -> Self {
        self.use_fundamental = true;
        self
    }

    /// Override the default RANSAC configuration.
    pub fn with_config(mut self, config: RansacConfig) -> Self {
        self.config = config;
        self
    }

    /// Filter matches using RANSAC, returning inlier matches and the estimated model.
    pub fn filter_matches(
        &self,
        matches: &Matches,
        src_points: &[(f64, f64)],
        dst_points: &[(f64, f64)],
    ) -> (Matches, RansacResult<Matrix3<f64>>) {
        let result = if self.use_fundamental {
            estimate_fundamental(matches, src_points, dst_points, &self.config)
        } else {
            estimate_homography(matches, src_points, dst_points, &self.config)
        };

        let filtered = filter_matches_by_inliers(matches, &result.inliers);
        (filtered, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::FeatureMatch;

    fn create_synthetic_matches() -> (Matches, Vec<(f64, f64)>, Vec<(f64, f64)>) {
        // Points in general position. The previous fixture laid them out on the
        // ray `y = x/2`, so *every* 4-point sample — every sample RANSAC draws —
        // was collinear. The DLT solver now rejects a collinear sample as rank
        // deficient (`cv_calib3d::dlt::DLT_RANK_TOLERANCE`), so the estimator
        // correctly returned no model for every iteration and the test failed.
        // A ray of points was never a valid homography fixture.
        let mut matches = Matches::new();
        let mut src_points = Vec::new();
        let mut dst_points = Vec::new();

        // Create exact matches with a known homography: a translation by
        // (+10, +5), on a jittered grid so no 4 points are collinear.
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for i in 0..20 {
            let x = 20.0 + 15.0 * (i % 5) as f64 + next();
            let y = 20.0 + 13.0 * (i / 5) as f64 + next();

            src_points.push((x, y));
            dst_points.push((x + 10.0, y + 5.0));

            matches.push(FeatureMatch::new(i, i, 0.0));
        }

        // Add some outliers
        for i in 20..25 {
            src_points.push((20.0 + 15.0 * (i % 5) as f64 + next(), 60.0 + next() * 40.0));
            dst_points.push((20.0 + 15.0 * (i % 5) as f64 + next(), 200.0 + next() * 40.0)); // Wrong correspondence
            matches.push(FeatureMatch::new(i, i, 0.0));
        }

        (matches, src_points, dst_points)
    }

    #[test]
    fn test_ransac_homography() {
        let (matches, src_points, dst_points) = create_synthetic_matches();

        let config = RansacConfig {
            threshold: 15.0, // Higher threshold for translation
            max_iterations: 1000,
            confidence: 0.99,
        };

        let result = estimate_homography(&matches, &src_points, &dst_points, &config);

        // With identity homography and translation, we should get some inliers
        // The actual number depends on the threshold
        assert!(result.model.is_some(), "Should find a model");
        assert_eq!(
            result.inliers.len(),
            matches.len(),
            "the inlier mask must stay parallel to the match list"
        );
    }

    #[test]
    fn test_ransac_matcher() {
        let (matches, src_points, dst_points) = create_synthetic_matches();

        let matcher = RansacMatcher::new();
        let (filtered, result) = matcher.filter_matches(&matches, &src_points, &dst_points);

        // With identity homography and translation transformation,
        // we won't get perfect inliers but the pipeline should work: the
        // filtered set must be a subset of the input, index-parallel with it.
        assert!(filtered.len() <= matches.len());
        assert_eq!(
            filtered.matches.len(),
            result.inliers.iter().filter(|&&i| i).count(),
            "filtering must keep exactly the inliers the result reported"
        );
    }

    /// Two pinhole cameras looking at deterministic random 3D points.
    fn synthetic_correspondences(n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        use nalgebra::{Matrix3, Vector3};
        let k = Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0);
        let angle = 0.15f64;
        let r = Matrix3::new(
            angle.cos(),
            0.0,
            angle.sin(),
            0.0,
            1.0,
            0.0,
            -angle.sin(),
            0.0,
            angle.cos(),
        );
        let t = Vector3::new(-0.5, 0.0, 0.0);

        let mut s = 7u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        let mut i = 0usize;
        while pts1.len() < n && i < 10 * n {
            i += 1;
            let x = Vector3::new(next() * 4.0, next() * 4.0, 5.0 + next() * 3.0);
            let y = r * x + t;
            let u = k * x;
            let v = k * y;
            pts1.push((u[0] / u[2], u[1] / u[2]));
            pts2.push((v[0] / v[2], v[1] / v[2]));
        }
        (pts1, pts2)
    }

    /// The three former copies of the 8-point solver (cv-calib3d's
    /// `FundamentalSolver`, cv-calib3d's `find_fundamental_mat`, and this
    /// crate's RANSAC estimator) must all return the same matrix up to sign and
    /// scale.
    #[test]
    fn fundamental_solvers_agree_up_to_sign() {
        let (pts1, pts2) = synthetic_correspondences(12);

        let flat1: Vec<[f64; 2]> = pts1.iter().map(|p| [p.0, p.1]).collect();
        let flat2: Vec<[f64; 2]> = pts2.iter().map(|p| [p.0, p.1]).collect();
        let f_solver = cv_calib3d::fundamental::FundamentalSolver::estimate(&flat1, &flat2)
            .expect("FundamentalSolver");

        let p1: Vec<nalgebra::Point2<f64>> = pts1
            .iter()
            .map(|p| nalgebra::Point2::new(p.0, p.1))
            .collect();
        let p2: Vec<nalgebra::Point2<f64>> = pts2
            .iter()
            .map(|p| nalgebra::Point2::new(p.0, p.1))
            .collect();
        let f_free = cv_calib3d::find_fundamental_mat(&p1, &p2).expect("find_fundamental_mat");

        let data: Vec<MatchPair> = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| MatchPair { src: *a, dst: *b })
            .collect();
        let refs: Vec<&MatchPair> = data.iter().collect();
        let f_ransac = FundamentalEstimator
            .estimate(&refs)
            .expect("RANSAC estimator");

        let unit = |m: &Matrix3<f64>| m / m.norm();
        for (name, other) in [("find_fundamental_mat", &f_free), ("ransac", &f_ransac)] {
            let direct = (unit(&f_solver) - unit(other)).norm();
            let flipped = (unit(&f_solver) + unit(other)).norm();
            assert!(
                direct < 1e-9 || flipped < 1e-9,
                "FundamentalSolver and {name} disagree: {direct} / {flipped}"
            );
        }
    }

    /// 20 exact correspondences of a real two-view geometry, so the 8-point
    /// minimum sample is available and the solution is not degenerate.
    fn valid_matches() -> (Matches, Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let (src, dst) = synthetic_correspondences(20);
        let mut matches = Matches::new();
        for i in 0..src.len() as i32 {
            matches.push(FeatureMatch::new(i, i, 0.0));
        }
        (matches, src, dst)
    }

    /// Malformed match indices must be skipped, not panicked on — and the
    /// surviving correspondences must still produce the right answer.
    ///
    /// The old code was `src_points[m.query_idx as usize]`: a negative
    /// `query_idx` wrapped to a huge offset and a too-large one indexed past the
    /// end, either panicking. Before the fix every assertion below panicked at
    /// the indexing expression.
    #[test]
    fn out_of_range_match_indices_are_skipped_and_the_rest_still_solve() {
        let (base, src, dst) = valid_matches();

        let estimate = |m: &Matches| estimate_fundamental(m, &src, &dst, &RansacConfig::default());

        let result = estimate(&base);
        assert!(
            result.model.is_some(),
            "control: the 20 valid matches must still yield a model"
        );

        // The four malformed shapes: negative query index (wraps huge under
        // `as usize`), negative train index, both past the end of the arrays.
        for (label, bad) in [
            ("negative query index", FeatureMatch::new(-1, 0, 0.0)),
            ("negative train index", FeatureMatch::new(0, -3, 0.0)),
            (
                "query index past the end",
                FeatureMatch::new(src.len() as i32 + 7, 0, 0.0),
            ),
            (
                "train index past the end",
                FeatureMatch::new(0, dst.len() as i32 + 9, 0.0),
            ),
        ] {
            let mut matches = base.clone();
            // Put a broken match in the *middle* so an off-by-one in the
            // survivor mapping would also show up.
            matches.matches.insert(4, bad);
            let r = estimate(&matches);

            assert!(
                r.model.is_some(),
                "{label}: 20 valid correspondences remain, so a model must still be found"
            );
            assert_eq!(
                r.inliers.len(),
                matches.matches.len(),
                "{label}: inlier flags must stay parallel to matches.matches"
            );
            assert!(
                !r.inliers[4],
                "{label}: the skipped match must not be reported as an inlier"
            );

            let f = r.model.unwrap();
            let unit = f / f.norm();
            // The correspondence set is unchanged by skipping the bad match, so
            // the recovered epipolar geometry must be the same matrix.
            let mut base_n = result.model.clone().unwrap();
            base_n /= base_n.norm();
            let agree = (unit - base_n).norm() < 1e-9 || (unit + base_n).norm() < 1e-9;
            assert!(
                agree,
                "{label}: skipping the bad match changed the estimated fundamental matrix"
            );
        }
    }

    /// A skipped match must not shift the inlier flags of the matches after it:
    /// `filter_matches_by_inliers` indexes by position, so a one-position shift
    /// would keep the wrong matches.
    #[test]
    fn inlier_flags_stay_aligned_when_a_match_is_skipped() {
        let (base, src, dst) = valid_matches();
        let mut matches = base.clone();
        // All valid matches + 1 broken one in the middle.
        matches.matches.insert(2, FeatureMatch::new(-5, -5, 0.0));

        let result = estimate_fundamental(&matches, &src, &dst, &RansacConfig::default());
        let filtered = filter_matches_by_inliers(&matches, &result.inliers);

        assert_eq!(
            filtered.len(),
            base.len(),
            "every exact correspondence must survive"
        );
        for kp in &filtered.matches {
            assert_ne!(
                kp.query_idx, -5,
                "the skipped match must not appear in the filtered output"
            );
            assert_eq!(kp.query_idx, kp.train_idx);
            assert!(kp.query_idx >= 0 && (kp.query_idx as usize) < src.len());
        }
        let ids: Vec<i32> = filtered.matches.iter().map(|m| m.query_idx).collect();
        let expected: Vec<i32> = (0..base.len() as i32).collect();
        assert_eq!(ids, expected, "the wrong matches were kept");
    }

    /// `estimate_homography` shared the identical unchecked indexing.
    #[test]
    fn out_of_range_match_indices_are_skipped_by_the_homography_path_too() {
        let (base, src, dst) = valid_matches();
        let mut matches = base.clone();
        matches
            .matches
            .insert(0, FeatureMatch::new(9999, 9999, 0.0));

        let result = estimate_homography(&matches, &src, &dst, &RansacConfig::default());
        assert!(result.model.is_some(), "valid correspondences remain");
        assert_eq!(result.inliers.len(), matches.matches.len());
        assert!(!result.inliers[0]);
    }
}
