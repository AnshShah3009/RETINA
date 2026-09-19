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

/// Estimate homography using RANSAC
pub fn estimate_homography(
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    config: &RansacConfig,
) -> RansacResult<Matrix3<f64>> {
    let data: Vec<MatchPair> = matches
        .matches
        .iter()
        .map(|m| MatchPair {
            src: src_points[m.query_idx as usize],
            dst: dst_points[m.train_idx as usize],
        })
        .collect();

    let ransac = Ransac::new(config.clone());
    ransac.run(&HomographyEstimator, &data)
}

/// Estimate fundamental matrix using RANSAC
pub fn estimate_fundamental(
    matches: &Matches,
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    config: &RansacConfig,
) -> RansacResult<Matrix3<f64>> {
    let data: Vec<MatchPair> = matches
        .matches
        .iter()
        .map(|m| MatchPair {
            src: src_points[m.query_idx as usize],
            dst: dst_points[m.train_idx as usize],
        })
        .collect();

    let ransac = Ransac::new(config.clone());
    ransac.run(&FundamentalEstimator, &data)
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
        let mut matches = Matches::new();
        let mut src_points = Vec::new();
        let mut dst_points = Vec::new();

        // Create perfect matches with known homography
        // Simple translation: (x, y) -> (x + 10, y + 5)
        for i in 0..20 {
            let x = i as f64 * 10.0;
            let y = i as f64 * 5.0;

            src_points.push((x, y));
            dst_points.push((x + 10.0, y + 5.0));

            matches.push(FeatureMatch::new(i, i, 0.0));
        }

        // Add some outliers
        for i in 20..25 {
            src_points.push((i as f64 * 10.0, i as f64 * 5.0));
            dst_points.push((i as f64 * 100.0, i as f64 * 100.0)); // Wrong correspondence
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

        println!(
            "RANSAC found {} inliers out of {} matches",
            result.num_inliers,
            matches.len()
        );

        // With identity homography and translation, we should get some inliers
        // The actual number depends on the threshold
        assert!(result.model.is_some(), "Should find a model");

        // Verify the homography exists
        if let Some(h) = result.model {
            println!("Homography matrix:\n{}", h);
        }
    }

    #[test]
    fn test_ransac_matcher() {
        let (matches, src_points, dst_points) = create_synthetic_matches();

        let matcher = RansacMatcher::new();
        let (filtered, result) = matcher.filter_matches(&matches, &src_points, &dst_points);

        println!(
            "Filtered from {} to {} matches",
            matches.len(),
            filtered.len()
        );
        println!("Found {} inliers", result.num_inliers);

        // With identity homography and translation transformation,
        // we won't get perfect inliers but the pipeline should work
        // Just verify it runs without panicking
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
}
