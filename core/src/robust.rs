//! Robust Estimation Module
//!
//! Provides a generic RANSAC implementation that can be used for any model estimation task.

use rand::seq::SliceRandom;
use std::marker::PhantomData;

/// Configuration for robust estimation
#[derive(Debug, Clone)]
pub struct RobustConfig {
    pub threshold: f64,
    pub max_iterations: usize,
    pub confidence: f64,
    pub min_sample_size: usize,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            max_iterations: 1000,
            confidence: 0.99,
            min_sample_size: 4,
        }
    }
}

/// Result of robust estimation
#[derive(Debug, Clone)]
pub struct RobustResult<M> {
    pub model: Option<M>,
    pub inliers: Vec<bool>,
    pub num_inliers: usize,
    pub residual: f64,
}

/// Trait for models that can be estimated robustly
pub trait RobustModel<D> {
    type Model: Clone;

    /// Minimum number of data points required to estimate the model
    fn min_sample_size(&self) -> usize;

    /// Estimate model from a minimal sample
    fn estimate(&self, data: &[&D]) -> Option<Self::Model>;

    /// Compute error for a single data point against the model
    fn compute_error(&self, model: &Self::Model, data: &D) -> f64;
}

/// Generic RANSAC engine
pub struct Ransac<D, M: RobustModel<D>> {
    config: RobustConfig,
    _phantom: PhantomData<(D, M)>,
}

impl<D, M: RobustModel<D>> Ransac<D, M> {
    pub fn new(config: RobustConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    pub fn run(&self, estimator: &M, data: &[D]) -> RobustResult<M::Model> {
        let n = data.len();
        let k = estimator.min_sample_size();

        if n < k {
            return RobustResult {
                model: None,
                inliers: vec![false; n],
                num_inliers: 0,
                residual: f64::INFINITY,
            };
        }

        let mut best_model = None;
        let mut best_inliers = vec![false; n];
        let mut best_num_inliers = 0;
        let mut best_residual = f64::INFINITY;

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.config.max_iterations {
            // 1. Sample
            indices.shuffle(&mut rng);
            let sample: Vec<&D> = (0..k).map(|i| &data[indices[i]]).collect();

            // 2. Estimate
            if let Some(model) = estimator.estimate(&sample) {
                // 3. Score
                let mut inliers = vec![false; n];
                let mut num_inliers = 0;
                let mut total_error = 0.0;

                for (j, d) in data.iter().enumerate() {
                    let err = estimator.compute_error(&model, d);
                    if err < self.config.threshold {
                        inliers[j] = true;
                        num_inliers += 1;
                        total_error += err;
                    }
                }

                let residual = if num_inliers > 0 {
                    total_error / num_inliers as f64
                } else {
                    f64::INFINITY
                };

                if num_inliers > best_num_inliers
                    || (num_inliers == best_num_inliers && residual < best_residual)
                {
                    best_num_inliers = num_inliers;
                    best_inliers = inliers;
                    best_model = Some(model);
                    best_residual = residual;

                    // Early exit check
                    if num_inliers as f64 > n as f64 * self.config.confidence {
                        break;
                    }
                }
            }
        }

        RobustResult {
            model: best_model,
            inliers: best_inliers,
            num_inliers: best_num_inliers,
            residual: best_residual,
        }
    }
}

/// Generic LMedS (Least Median of Squares) engine
pub struct LMedS<D, M: RobustModel<D>> {
    config: RobustConfig,
    _phantom: PhantomData<(D, M)>,
}

impl<D, M: RobustModel<D>> LMedS<D, M> {
    pub fn new(config: RobustConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    pub fn run(&self, estimator: &M, data: &[D]) -> RobustResult<M::Model> {
        let n = data.len();
        let k = estimator.min_sample_size();

        if n < k {
            return RobustResult {
                model: None,
                inliers: vec![false; n],
                num_inliers: 0,
                residual: f64::INFINITY,
            };
        }

        let mut best_model = None;
        let mut best_median_error = f64::INFINITY;

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.config.max_iterations {
            indices.shuffle(&mut rng);
            let sample: Vec<&D> = (0..k).map(|i| &data[indices[i]]).collect();

            if let Some(model) = estimator.estimate(&sample) {
                let mut errors: Vec<f64> = data
                    .iter()
                    .map(|d| estimator.compute_error(&model, d))
                    .filter(|&e| e.is_finite())
                    .collect();

                if errors.is_empty() {
                    continue;
                }

                errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median_error = errors[errors.len() / 2];

                if median_error < best_median_error {
                    best_median_error = median_error;
                    best_model = Some(model);
                }
            }
        }

        // Final inlier count based on the best model and a derived threshold
        // Standard LMedS threshold: 2.5 * 1.4826 * (1 + 5/(n-k)) * sqrt(best_median_error)
        let mut inliers = vec![false; n];
        let mut num_inliers = 0;
        let mut best_residual = f64::INFINITY;

        if let Some(ref model) = best_model {
            let sigma = 1.4826 * (1.0 + 5.0 / (n - k) as f64) * best_median_error.sqrt();
            let threshold = 2.5 * sigma;
            let mut total_error = 0.0;

            for (j, d) in data.iter().enumerate() {
                let err = estimator.compute_error(model, d);
                if err < threshold {
                    inliers[j] = true;
                    num_inliers += 1;
                    total_error += err;
                }
            }
            best_residual = if num_inliers > 0 {
                total_error / num_inliers as f64
            } else {
                f64::INFINITY
            };
        }

        RobustResult {
            model: best_model,
            inliers,
            num_inliers,
            residual: best_residual,
        }
    }
}

/// Generic PROSAC (Progressive Sample Consensus) engine
pub struct Prosac<D, M: RobustModel<D>> {
    config: RobustConfig,
    _phantom: PhantomData<(D, M)>,
}

impl<D, M: RobustModel<D>> Prosac<D, M> {
    pub fn new(config: RobustConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Run PROSAC estimation.
    /// data: The data points, MUST BE SORTED BY QUALITY (best first).
    pub fn run(&self, estimator: &M, data: &[D]) -> RobustResult<M::Model> {
        let n = data.len();
        let m = estimator.min_sample_size();

        if n < m {
            return RobustResult {
                model: None,
                inliers: vec![false; n],
                num_inliers: 0,
                residual: f64::INFINITY,
            };
        }

        let mut best_model = None;
        let mut best_inliers = vec![false; n];
        let mut best_num_inliers = 0;
        let mut best_residual = f64::INFINITY;

        let mut rng = rand::thread_rng();

        // PROSAC parameters
        let t_max = self.config.max_iterations;
        let mut n_sub = m;
        let mut t_n = 1.0;

        for t in 1..=t_max {
            // 1. Determine size of subset to sample from (Growth Function)
            // Progressive sampling: starts with small subset of high-quality data
            if t > t_n as usize && n_sub < n {
                n_sub += 1;
                let mut num = 1.0;
                for i in 0..m {
                    num *= (n_sub - i) as f64 / (m - i) as f64;
                }
                t_n = num * t_max as f64 / n as f64;
            }

            // 2. Sample
            let mut sample_indices = Vec::with_capacity(m);
            if n_sub < n {
                // Choice of m-1 points from n_sub-1 and 1 point being the n_sub-th
                sample_indices.push(n_sub - 1);
                let mut pool: Vec<usize> = (0..n_sub - 1).collect();
                pool.shuffle(&mut rng);
                sample_indices.extend(pool.iter().take(m - 1));
            } else {
                // Standard RANSAC sampling if subset reached full data
                let mut pool: Vec<usize> = (0..n).collect();
                pool.shuffle(&mut rng);
                sample_indices.extend(pool.iter().take(m));
            }

            let sample: Vec<&D> = sample_indices.iter().map(|&idx| &data[idx]).collect();

            // 3. Estimate & Score
            if let Some(model) = estimator.estimate(&sample) {
                let mut inliers = vec![false; n];
                let mut num_inliers = 0;
                let mut total_error = 0.0;

                for (j, d) in data.iter().enumerate() {
                    let err = estimator.compute_error(&model, d);
                    if err < self.config.threshold {
                        inliers[j] = true;
                        num_inliers += 1;
                        total_error += err;
                    }
                }

                let residual = if num_inliers > 0 {
                    total_error / num_inliers as f64
                } else {
                    f64::INFINITY
                };

                if num_inliers > best_num_inliers
                    || (num_inliers == best_num_inliers && residual < best_residual)
                {
                    best_num_inliers = num_inliers;
                    best_inliers = inliers;
                    best_model = Some(model);
                    best_residual = residual;

                    // Standard PROSAC early exit check
                    if num_inliers as f64 > n as f64 * self.config.confidence {
                        break;
                    }
                }
            }
        }

        RobustResult {
            model: best_model,
            inliers: best_inliers,
            num_inliers: best_num_inliers,
            residual: best_residual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    #[derive(Clone)]
    struct LineModel {
        a: f64,
        b: f64,
        c: f64,
    }

    struct LineEstimator;

    impl LineEstimator {
        fn fit_line(p1: &Point2<f64>, p2: &Point2<f64>) -> LineModel {
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let norm = (dx * dx + dy * dy).sqrt();
            if norm < 1e-10 {
                LineModel {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                }
            } else {
                LineModel {
                    a: dy / norm,
                    b: -dx / norm,
                    c: -(dy * p1.x - dx * p1.y) / norm,
                }
            }
        }

        fn distance(model: &LineModel, p: &Point2<f64>) -> f64 {
            (model.a * p.x + model.b * p.y + model.c).abs()
        }
    }

    impl RobustModel<Point2<f64>> for LineEstimator {
        type Model = LineModel;

        fn min_sample_size(&self) -> usize {
            2
        }

        fn estimate(&self, data: &[&Point2<f64>]) -> Option<Self::Model> {
            if data.len() < 2 {
                return None;
            }
            Some(Self::fit_line(data[0], data[1]))
        }

        fn compute_error(&self, model: &Self::Model, point: &Point2<f64>) -> f64 {
            Self::distance(model, point)
        }
    }

    fn create_line_points(
        a: f64,
        b: f64,
        c: f64,
        n_inliers: usize,
        n_outliers: usize,
        noise: f64,
    ) -> Vec<Point2<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut points = Vec::new();

        for i in 0..n_inliers {
            let x = i as f64 * 0.1;
            let y = if b.abs() > 1e-10 {
                -(a * x + c) / b
            } else {
                -c / a
            };
            let y = y + rng.gen_range(-noise..noise);
            points.push(Point2::new(x, y));
        }

        for _ in 0..n_outliers {
            let x = rng.gen_range(0.0..10.0);
            let y = rng.gen_range(-10.0..10.0);
            points.push(Point2::new(x, y));
        }

        points
    }

    mod ransac_tests {
        use super::*;

        #[test]
        fn test_ransac_perfect_data() {
            let points: Vec<Point2<f64>> = (0..10)
                .map(|i| Point2::new(i as f64, i as f64 * 2.0))
                .collect();

            let ransac = Ransac::new(RobustConfig {
                threshold: 0.1,
                confidence: 0.99,
                max_iterations: 100,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = ransac.run(&estimator, &points);

            assert!(result.model.is_some());
            assert!(result.num_inliers >= 8);
        }

        #[test]
        fn test_ransac_with_outliers() {
            let points = create_line_points(2.0, -1.0, 0.0, 20, 5, 0.01);

            let ransac = Ransac::new(RobustConfig {
                threshold: 0.2,
                confidence: 0.99,
                max_iterations: 1000,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = ransac.run(&estimator, &points);

            assert!(result.model.is_some());
            assert!(result.num_inliers >= 15);
        }

        #[test]
        fn test_ransac_insufficient_points() {
            let points = vec![Point2::new(0.0, 0.0)];

            let ransac = Ransac::new(RobustConfig {
                threshold: 0.1,
                confidence: 0.99,
                max_iterations: 100,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = ransac.run(&estimator, &points);

            assert!(result.model.is_none());
            assert_eq!(result.num_inliers, 0);
        }

        #[test]
        fn test_ransac_empty_data() {
            let points: Vec<Point2<f64>> = vec![];

            let ransac = Ransac::new(RobustConfig {
                threshold: 0.1,
                confidence: 0.99,
                max_iterations: 100,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = ransac.run(&estimator, &points);

            assert!(result.model.is_none());
        }

        #[test]
        fn test_ransac_config_default() {
            let config = RobustConfig::default();

            assert_eq!(config.threshold, 1.0);
            assert_eq!(config.confidence, 0.99);
            assert_eq!(config.max_iterations, 1000);
        }
    }

    mod lmeds_tests {
        use super::*;

        #[test]
        fn test_lmeds_perfect_data() {
            let points: Vec<Point2<f64>> = (0..10)
                .map(|i| Point2::new(i as f64, i as f64 * 2.0))
                .collect();

            let lmeds = LMedS::new(RobustConfig {
                threshold: 0.1,
                max_iterations: 100,
                confidence: 0.99,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = lmeds.run(&estimator, &points);

            assert!(result.model.is_some());
        }

        #[test]
        fn test_lmeds_with_outliers() {
            let points = create_line_points(1.0, -1.0, 0.0, 20, 10, 0.01);

            let lmeds = LMedS::new(RobustConfig {
                threshold: 0.3,
                max_iterations: 500,
                confidence: 0.99,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = lmeds.run(&estimator, &points);

            assert!(result.model.is_some());
        }

        #[test]
        fn test_lmeds_handles_nan() {
            let mut points: Vec<Point2<f64>> =
                (0..10).map(|i| Point2::new(i as f64, i as f64)).collect();
            points.push(Point2::new(f64::NAN, 0.0));

            let lmeds = LMedS::new(RobustConfig {
                threshold: 0.1,
                max_iterations: 100,
                confidence: 0.99,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = lmeds.run(&estimator, &points);

            assert!(result.model.is_some());
        }
    }

    mod prosac_tests {
        use super::*;

        #[test]
        fn test_prosac_perfect_data() {
            let points: Vec<Point2<f64>> = (0..10)
                .map(|i| Point2::new(i as f64, i as f64 * 2.0))
                .collect();

            let prosac = Prosac::new(RobustConfig {
                threshold: 0.1,
                confidence: 0.99,
                max_iterations: 100,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = prosac.run(&estimator, &points);

            assert!(result.model.is_some());
        }

        #[test]
        fn test_prosac_with_outliers() {
            let points = create_line_points(1.5, -1.0, 0.5, 20, 5, 0.01);

            let prosac = Prosac::new(RobustConfig {
                threshold: 0.2,
                confidence: 0.99,
                max_iterations: 500,
                min_sample_size: 2,
            });

            let estimator = LineEstimator;
            let result = prosac.run(&estimator, &points);

            assert!(result.model.is_some());
        }
    }

    mod robust_result_tests {
        use super::*;

        #[test]
        fn test_robust_result_default() {
            let result: RobustResult<LineModel> = RobustResult {
                model: None,
                inliers: vec![],
                num_inliers: 0,
                residual: f64::INFINITY,
            };

            assert!(result.model.is_none());
            assert_eq!(result.num_inliers, 0);
        }
    }
}
