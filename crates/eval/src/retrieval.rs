//! Image-retrieval metrics (recall@K, precision@K, mean average precision).
//!
//! Metrics are computed over a list of queries. Each query pairs a *ranked*
//! list of predicted item ids with the set of ground-truth relevant ids.
//!
//! # Example
//!
//! ```
//! use cv_eval::retrieval::{recall_at_k, precision_at_k, mean_average_precision};
//!
//! let predictions = vec![vec![7usize, 2, 5]];
//! let ground_truth = vec![vec![2usize]];
//!
//! assert!((recall_at_k(&predictions, &ground_truth, 2) - 1.0).abs() < 1e-12);
//! assert!((precision_at_k(&predictions, &ground_truth, 2) - 0.5).abs() < 1e-12);
//! assert!((mean_average_precision(&predictions, &ground_truth) - 0.5).abs() < 1e-12);
//! ```

/// Mean recall@K over all queries.
///
/// For each query the recall is `|top-k predictions ∩ relevant| / |relevant|`;
/// queries with an empty ground-truth set are skipped. Returns `0.0` when
/// `k == 0`, when there are no queries, or when every query is empty.
pub fn recall_at_k(predictions: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f64 {
    let n = predictions.len().min(ground_truth.len());
    if n == 0 || k == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut counted = 0usize;
    for (pred, gt) in predictions.iter().zip(ground_truth.iter()).take(n) {
        if gt.is_empty() {
            continue;
        }
        let hits = pred.iter().take(k).filter(|p| gt.contains(p)).count();
        sum += hits as f64 / gt.len() as f64;
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        sum / counted as f64
    }
}

/// Mean precision@K over all queries.
///
/// For each query the precision is `|top-k predictions ∩ relevant| / k`;
/// queries with an empty ground-truth set contribute `0.0`. Returns `0.0` when
/// `k == 0` or there are no queries.
pub fn precision_at_k(predictions: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f64 {
    let n = predictions.len().min(ground_truth.len());
    if n == 0 || k == 0 {
        return 0.0;
    }

    let sum: f64 = predictions
        .iter()
        .zip(ground_truth.iter())
        .take(n)
        .map(|(pred, gt)| {
            let hits = pred.iter().take(k).filter(|p| gt.contains(p)).count();
            hits as f64 / k as f64
        })
        .sum();

    sum / n as f64
}

/// Mean average precision over all queries.
///
/// The average precision of a query is the mean of the precision values
/// evaluated at each rank where a relevant item is retrieved. Queries with no
/// relevant items or no hits are skipped. Returns `0.0` when there are no
/// counted queries.
pub fn mean_average_precision(predictions: &[Vec<usize>], ground_truth: &[Vec<usize>]) -> f64 {
    let n = predictions.len().min(ground_truth.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut counted = 0usize;
    for (pred, gt) in predictions.iter().zip(ground_truth.iter()).take(n) {
        if gt.is_empty() {
            continue;
        }

        let mut hits = 0usize;
        let mut average_precision = 0.0;
        for (rank, item) in pred.iter().enumerate() {
            if gt.contains(item) {
                hits += 1;
                average_precision += hits as f64 / (rank + 1) as f64;
            }
        }

        if hits > 0 {
            sum += average_precision / hits as f64;
            counted += 1;
        }
    }

    if counted == 0 {
        0.0
    } else {
        sum / counted as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_at_k_matches_hand_computation() {
        let predictions = vec![vec![0usize, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
        let ground_truth = vec![vec![0usize, 1], vec![4], vec![]];

        // q0: top1={0} -> 1/2 ; q1: top1={3} -> 0/1 ; q2: empty -> skipped.
        assert!((recall_at_k(&predictions, &ground_truth, 1) - 0.25).abs() < 1e-12);
        // q0: {0,1} -> 2/2 ; q1: {3,4} -> 1/1 ; averaged.
        assert!((recall_at_k(&predictions, &ground_truth, 2) - 1.0).abs() < 1e-12);
        assert!((recall_at_k(&predictions, &ground_truth, 3) - 1.0).abs() < 1e-12);
        assert_eq!(recall_at_k(&predictions, &ground_truth, 0), 0.0);
    }

    #[test]
    fn precision_at_k_matches_hand_computation() {
        let predictions = vec![vec![0usize, 1, 2, 3]];
        let ground_truth = vec![vec![1usize, 3]];

        // top2={0,1}: 1 hit / 2.
        assert!((precision_at_k(&predictions, &ground_truth, 2) - 0.5).abs() < 1e-12);
        // top4={0,1,2,3}: 2 hits / 4.
        assert!((precision_at_k(&predictions, &ground_truth, 4) - 0.5).abs() < 1e-12);
        assert_eq!(precision_at_k(&predictions, &ground_truth, 0), 0.0);
    }

    #[test]
    fn mean_average_precision_matches_hand_computation() {
        let predictions = vec![vec![0usize, 1, 2, 3]];
        let ground_truth = vec![vec![1usize, 3]];

        // Hits at rank 2 (P=1/2) and rank 4 (P=2/4): AP = (0.5 + 0.5) / 2 = 0.5.
        assert!((mean_average_precision(&predictions, &ground_truth) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_inputs_are_zero() {
        assert_eq!(recall_at_k(&[], &[], 5), 0.0);
        assert_eq!(precision_at_k(&[], &[], 5), 0.0);
        assert_eq!(mean_average_precision(&[], &[]), 0.0);
    }
}
