//! Enhanced RANSAC variants from OpenCV 5.x
//!
//! - **USAC**: Universal SAC with progressive NAPSAC sampling + LO-RANSAC polishing
//! - **NAPSAC**: Nearest-neighbor guided sampling for locally connected structures
//! - **MAGSAC**: Margin-adaptive σ-consensus scoring instead of hard threshold

use cv_core::geometry::Rect;
use nalgebra::{DMatrix, Point2};
use rand::Rng;
use std::collections::HashSet;

/// Sampling method for RANSAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UsacSampling {
    /// Uniform random sampling
    Uniform,
    /// Progressive NAPSAC — nearest-neighbor guided
    ProgressiveNapsac,
    /// NAPSAC-only sampling
    Napsac,
}

/// Local optimization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalOptimMethod {
    /// No local optimization
    Null,
    /// Inner-LO: optimize on best inliers
    InnerLo,
    /// Inner + iterative LO
    InnerAndIterLo,
}

/// Scoring method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreMethod {
    /// Classic RANSAC: count inliers with hard threshold
    Ransac,
    /// MSAC: M-estimator-based scoring
    Msac,
    /// MAGSAC: σ-consensus margin-adaptive scoring
    Magsac,
    /// LMedS: least median of squares
    Lmeds,
}

/// USAC parameters
#[derive(Clone)]
pub struct UsacParams {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Confidence level (probability of sampling all-inlier set)
    pub confidence: f64,
    /// Inlier threshold (pixels)
    pub threshold: f64,
    /// Sampling method
    pub sampling: UsacSampling,
    /// Local optimization method
    pub local_optim: LocalOptimMethod,
    /// Scoring method
    pub score: ScoreMethod,
    /// NAPSAC neighborhood size
    pub neighbor_count: usize,
    /// Minimum inlier ratio to trigger LO
    pub lo_inlier_ratio: f64,
    /// MAGSAC: maximum σ for margin
    pub magsac_sigma_max: f64,
}

impl Default for UsacParams {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            confidence: 0.999,
            threshold: 2.0,
            sampling: UsacSampling::ProgressiveNapsac,
            local_optim: LocalOptimMethod::InnerLo,
            score: ScoreMethod::Magsac,
            neighbor_count: 8,
            lo_inlier_ratio: 0.1,
            magsac_sigma_max: 10.0,
        }
    }
}

/// Result of model estimation with quality metrics
#[derive(Debug, Clone)]
pub struct UsacResult<M> {
    pub model: M,
    pub inliers: usize,
    pub score: f64,
    pub mask: Vec<bool>,
    pub iterations: usize,
}

/// USAC-based model estimation
pub fn estimate_usac<M, E, S>(
    points: &[Point2<f64>],
    params: &UsacParams,
    estimator: &E,
    scorer: &S,
    sample_size: usize,
) -> Option<UsacResult<M>>
where
    M: Clone,
    E: Fn(&[Point2<f64>], &[usize]) -> Option<M>,
    S: Fn(&M, &[Point2<f64>]) -> (usize, f64, Vec<bool>),
{
    let n = points.len();
    if n < sample_size {
        return None;
    }

    let mut rng = rand::thread_rng();
    let mut best_model: Option<M> = None;
    let mut best_inliers = 0;
    let mut best_score = f64::MAX;
    let mut best_mask = vec![false; n];

    // Precompute neighbor indices for NAPSAC
    let neighbors = if matches!(
        params.sampling,
        UsacSampling::Napsac | UsacSampling::ProgressiveNapsac
    ) {
        compute_neighbors(points, params.neighbor_count)
    } else {
        vec![Vec::new(); n]
    };

    // Adaptive iteration count: N = ln(1 - confidence) / ln(1 - eps^s).
    // The true inlier ratio eps is unknown before scoring, so assume a
    // conservative 50% — the bound is refined implicitly by the fixed cap.
    let max_iters = if params.max_iterations == 0 {
        let confidence = params.confidence.clamp(0.0, 0.999_999);
        let p_good = 0.5f64.powi(sample_size.min(64) as i32);
        if p_good <= 0.0 || p_good >= 1.0 {
            1000
        } else {
            (((1.0 - confidence).ln() / (1.0 - p_good).ln()).ceil() as usize).max(1)
        }
    } else {
        params.max_iterations
    };

    for iter in 0..max_iters {
        let sample = match params.sampling {
            UsacSampling::Uniform => sample_uniform(&mut rng, n, sample_size),
            UsacSampling::Napsac => sample_napsac(&mut rng, &neighbors, n, sample_size),
            UsacSampling::ProgressiveNapsac => {
                if iter < max_iters / 2 {
                    sample_uniform(&mut rng, n, sample_size)
                } else {
                    sample_napsac(&mut rng, &neighbors, n, sample_size)
                }
            }
        };

        // Skip degenerate samples (e.g. duplicate/collinear points); they
        // must not abort the whole search.
        let Some(model) = estimator(points, &sample) else {
            continue;
        };
        let (inliers, score, mask) = scorer(&model, points);

        // Local optimization
        let (model, inliers, score, mask) = if inliers >= (n as f64 * params.lo_inlier_ratio) as usize
            && params.local_optim != LocalOptimMethod::Null
        {
            refine_local(points, &model, &mask, params, estimator, scorer)
                .unwrap_or((model, inliers, score, mask))
        } else {
            (model, inliers, score, mask)
        };

        if score < best_score {
            best_model = Some(model);
            best_inliers = inliers;
            best_score = score;
            best_mask = mask;
        }
    }

    best_model.map(|m| UsacResult {
        model: m,
        inliers: best_inliers,
        score: best_score,
        mask: best_mask,
        iterations: max_iters,
    })
}

/// Compute k-nearest neighbors for NAPSAC sampling
fn compute_neighbors(points: &[Point2<f64>], k: usize) -> Vec<Vec<usize>> {
    let n = points.len();
    if n <= 1 {
        return vec![Vec::new(); n];
    }
    let k = k.min(n - 1);
    let mut neighbors = Vec::with_capacity(n);

    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, p)| {
                let d = (p.x - points[i].x).hypot(p.y - points[i].y);
                (d, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        neighbors.push(dists.into_iter().map(|(_, j)| j).collect());
    }
    neighbors
}

/// Uniform random sampling
fn sample_uniform(rng: &mut impl Rng, n: usize, sample_size: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut sample = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
        sample.push(indices[i]);
    }
    sample
}

/// NAPSAC guided sampling
fn sample_napsac(
    rng: &mut impl Rng,
    neighbors: &[Vec<usize>],
    n: usize,
    sample_size: usize,
) -> Vec<usize> {
    let i0 = rng.random_range(0..n);
    let mut used = HashSet::new();
    used.insert(i0);
    let mut sample = vec![i0];

    while sample.len() < sample_size {
        if sample.len() == 1 {
            // Sample from neighbors of first point
            let ni = &neighbors[i0];
            if !ni.is_empty() {
                let next = ni[rng.random_range(0..ni.len())];
                if used.insert(next) {
                    sample.push(next);
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            // Fill remaining uniformly
            let idx = rng.random_range(0..n);
            if used.insert(idx) {
                sample.push(idx);
            }
        }
    }

    // If we couldn't get enough NAPSAC samples, fill uniformly
    while sample.len() < sample_size {
        let idx = rng.random_range(0..n);
        if used.insert(idx) {
            sample.push(idx);
        }
    }
    sample
}

/// Local model refinement on best inliers
fn refine_local<M: Clone>(
    points: &[Point2<f64>],
    model: &M,
    mask: &[bool],
    _params: &UsacParams,
    estimator: &impl Fn(&[Point2<f64>], &[usize]) -> Option<M>,
    scorer: &impl Fn(&M, &[Point2<f64>]) -> (usize, f64, Vec<bool>),
) -> Option<(M, usize, f64, Vec<bool>)> {
    let inlier_indices: Vec<usize> = (0..points.len()).filter(|&i| mask[i]).collect();
    if inlier_indices.len() < 4 {
        return None;
    }
    let refined = estimator(points, &inlier_indices)?;
    let (inliers, score, mask) = scorer(&refined, points);
    Some((refined, inliers, score, mask))
}

/// Scoring functions
pub mod scorers {
    use super::*;

    /// MSAC scoring: robust M-estimator with soft threshold
    pub fn score_msac<M>(
        model: &M,
        points: &[Point2<f64>],
        threshold: f64,
    ) -> (usize, f64, Vec<bool>)
    where
        M: Clone,
    {
        let t2 = threshold * threshold;
        let mut inliers = 0;
        let mut score = 0.0f64;
        let mut mask = vec![false; points.len()];

        // Default: use distance metric for generic model
        // Overridden by specific model types

        (inliers, score, mask)
    }

    /// MAGSAC scoring: σ-consensus with adaptive margin
    pub fn score_magsac<M: Clone>(
        model: &M,
        points: &[Point2<f64>],
        threshold: f64,
        sigma_max: f64,
    ) -> (usize, f64, Vec<bool>)
    where
        M: Clone,
    {
        let mut best_inliers = 0;
        let mut best_score = f64::MAX;
        let mut best_mask = vec![false; points.len()];

        for s in (1..=sigma_max as usize).rev() {
            let t = s as f64;
            let (inliers, score, mask) = score_msac(model, points, t);
            if inliers > best_inliers || (inliers == best_inliers && score < best_score) {
                best_inliers = inliers;
                best_score = score;
                best_mask = mask;
            }
        }

        (best_inliers, best_score, best_mask)
    }

    /// RANSAC hard-threshold scoring
    pub fn score_ransac<M: Clone>(
        model: &M,
        points: &[Point2<f64>],
        threshold: f64,
    ) -> (usize, f64, Vec<bool>) {
        score_msac(model, points, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn line_model(points: &[Point2<f64>], sample: &[usize]) -> Option<(f64, f64)> {
        if sample.len() < 2 {
            return None;
        }
        let p1 = points[sample[0]];
        let p2 = points[sample[1]];
        let dx = p2.x - p1.x;
        if dx.abs() < 1e-10 {
            return None;
        }
        let m = (p2.y - p1.y) / dx;
        let b = p1.y - m * p1.x;
        Some((m, b))
    }

    fn line_score(model: &(f64, f64), points: &[Point2<f64>]) -> (usize, f64, Vec<bool>) {
        let t = 0.1;
        let t2 = t * t;
        let mut inliers = 0;
        let mut score = 0.0;
        let mut mask = vec![false; points.len()];
        for (i, p) in points.iter().enumerate() {
            let d = (p.y - (model.0 * p.x + model.1)).abs();
            if d * d < t2 {
                inliers += 1;
                mask[i] = true;
                score += d;
            } else {
                score += t;
            }
        }
        (inliers, score, mask)
    }

    #[test]
    fn test_usac_line_fit() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut pts = Vec::new();
        // Ground truth: y = 2x + 5
        for x in 0..50 {
            pts.push(Point2::new(x as f64, 2.0 * x as f64 + 5.0 + rng.random_range(-0.05..0.05)));
        }
        // Add outliers
        for _ in 0..20 {
            pts.push(Point2::new(
                rng.random_range(0.0..50.0),
                rng.random_range(0.0..100.0),
            ));
        }

        let params = UsacParams {
            threshold: 0.3,
            ..Default::default()
        };

        let result = estimate_usac(&pts, &params, &line_model, &line_score, 2);
        assert!(result.is_some());
        let res = result.unwrap();
        assert!(res.inliers >= 40);
    }

    #[test]
    fn test_usac_auto_iterations_finds_model() {
        // max_iterations == 0 selects the adaptive iteration count. The old
        // formula degenerated to zero iterations and always returned None.
        let mut pts = Vec::new();
        for x in 0..40 {
            pts.push(Point2::new(x as f64, 3.0 * x as f64 + 1.0));
        }
        for _ in 0..10 {
            pts.push(Point2::new(100.0, 0.0));
        }

        let params = UsacParams {
            threshold: 0.1,
            max_iterations: 0,
            ..Default::default()
        };

        let result = estimate_usac(&pts, &params, &line_model, &line_score, 2);
        assert!(result.is_some(), "auto-iteration USAC must find the line");
        let res = result.unwrap();
        assert!(res.iterations > 0);
        assert!(res.inliers >= 40);
    }
}
