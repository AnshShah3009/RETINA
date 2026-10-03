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

/// Number of distinct ground-truth ids among the first `k` predictions.
///
/// A ranked list that repeats an id has retrieved that item *once*: counting
/// every occurrence made `recall_at_k` and `precision_at_k` exceed 1 and made an
/// all-repeats list look perfect. `trec_eval` and every other reference
/// implementation score a ranked list as a ranking of distinct items.
fn distinct_hits(pred: &[usize], gt: &[usize], k: usize) -> usize {
    let mut seen: Vec<usize> = Vec::new();
    let mut hits = 0usize;
    for item in pred.iter().take(k) {
        if gt.contains(item) && !seen.contains(item) {
            seen.push(*item);
            hits += 1;
        }
    }
    hits
}

/// Mean recall@K over all queries.
///
/// For each query the recall is `|top-k predictions ∩ relevant| / |relevant|`,
/// where the top-k predictions are treated as a set of distinct ids (a repeated
/// id counts once). Queries with an empty ground-truth set are skipped. Returns
/// `0.0` when `k == 0`, when there are no queries, or when every query is empty.
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
        let hits = distinct_hits(pred, gt, k);
        sum += hits as f64 / gt.len() as f64;
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        sum / counted as f64
    }
}

/// Hit rate@K: the fraction of queries whose top-K contains at least one
/// relevant item.
///
/// This is the convention used for place recognition / visual localization
/// ("recall@K" in that literature): a query counts as retrieved if *any* of the
/// top-K candidates is correct. It is not the same as [`recall_at_k`], which
/// measures the fraction of *all* relevant items that were retrieved and
/// therefore collapses when a query has many relevant items (for example a
/// dense database where dozens of frames sit within the hit radius of the
/// query). Report both when the distinction matters.
///
/// Queries with an empty ground-truth set are ignored. Returns 0.0 for empty
/// input or `k == 0`.
///
/// ```
/// use cv_eval::retrieval::hit_rate_at_k;
/// let predictions = vec![vec![7, 3, 9], vec![1, 2, 3]];
/// let ground_truth = vec![vec![3], vec![1]];
/// // First query finds its target at rank 2, second at rank 1.
/// assert_eq!(hit_rate_at_k(&predictions, &ground_truth, 1), 0.5);
/// assert_eq!(hit_rate_at_k(&predictions, &ground_truth, 2), 1.0);
/// ```
pub fn hit_rate_at_k(predictions: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f64 {
    let n = predictions.len().min(ground_truth.len());
    if n == 0 || k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    let mut counted = 0usize;
    for (pred, gt) in predictions.iter().zip(ground_truth.iter()).take(n) {
        if gt.is_empty() {
            continue;
        }
        if pred.iter().take(k).any(|p| gt.contains(p)) {
            hits += 1;
        }
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        hits as f64 / counted as f64
    }
}

/// Mean precision@K over all queries.
///
/// For each query the precision is `|top-k predictions ∩ relevant| / k`, where
/// the top-k predictions are treated as a set of distinct ids; queries with an
/// empty ground-truth set contribute `0.0`. Returns `0.0` when `k == 0` or there
/// are no queries.
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
            let hits = distinct_hits(pred, gt, k);
            hits as f64 / k as f64
        })
        .sum();

    sum / n as f64
}

/// Mean average precision over all queries.
///
/// The average precision of a query is the sum of the precision values
/// evaluated at each rank where a relevant item is retrieved, divided by the
/// number of relevant items for that query — the standard (TREC) definition. A
/// query that retrieved only one of its relevant items therefore scores at most
/// `1 / |relevant|`, and a query that retrieved none of them contributes `0.0`
/// rather than being dropped from the mean. Queries with an empty ground-truth
/// set are skipped, since their average precision is undefined. Returns `0.0`
/// when there are no counted queries.
///
/// A repeated id in a ranked list counts once, so the result is always in
/// `[0, 1]`.
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
        let mut seen: Vec<usize> = Vec::new();
        for (rank, item) in pred.iter().enumerate() {
            if gt.contains(item) && !seen.contains(item) {
                seen.push(*item);
                hits += 1;
                average_precision += hits as f64 / (rank + 1) as f64;
            }
        }

        // Divided by the number of *relevant* items, not by the number of hits:
        // dividing by the hits made a query that found one of ten relevant items
        // at rank 1 score exactly 1.0, and made a query that found nothing
        // vanish from the mean instead of pulling it down.
        sum += average_precision / gt.len() as f64;
        counted += 1;
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

    #[test]
    fn average_precision_is_normalised_by_the_relevant_count() {
        // One query, three relevant ids, only one of them retrieved - and at
        // rank 1, the best possible place for it.
        let predictions = vec![vec![7usize, 8, 9, 10]];
        let ground_truth = vec![vec![7usize, 11, 12]];
        // Sum of precision at hit ranks = 1/1; divided by |relevant| = 3.
        assert!(
            (mean_average_precision(&predictions, &ground_truth) - 1.0 / 3.0).abs() < 1e-12,
            "a query that found 1 of 3 relevant items scored {}",
            mean_average_precision(&predictions, &ground_truth)
        );

        // Control: the same query set with every relevant item retrieved first.
        let perfect = vec![vec![7usize, 11, 12]];
        assert!((mean_average_precision(&perfect, &ground_truth) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_query_with_no_hits_pulls_the_mean_down() {
        let predictions = vec![vec![7usize, 8, 9], vec![90usize, 91, 92]];
        let ground_truth = vec![vec![7usize, 8, 9], vec![1usize, 2, 3]];
        // q0 is perfect (AP 1.0); q1 retrieved nothing (AP 0.0) and is part of
        // the mean rather than being dropped from it.
        let map = mean_average_precision(&predictions, &ground_truth);
        assert!((map - 0.5).abs() < 1e-12, "MAP = {}", map);
    }

    #[test]
    fn repeated_ids_count_once() {
        let predictions = vec![vec![7usize, 7, 7]];
        let ground_truth = vec![vec![7usize]];

        // One relevant item in a three-slot list is a recall of 1.0, not 3.0.
        assert_eq!(recall_at_k(&predictions, &ground_truth, 3), 1.0);
        assert!((precision_at_k(&predictions, &ground_truth, 3) - 1.0 / 3.0).abs() < 1e-12);
        assert!((mean_average_precision(&predictions, &ground_truth) - 1.0).abs() < 1e-12);

        // Control: a list of distinct, all-relevant ids scores full precision.
        let distinct = vec![vec![7usize, 8, 9]];
        let gt3 = vec![vec![7usize, 8, 9]];
        assert!((precision_at_k(&distinct, &gt3, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn metrics_stay_within_unit_range() {
        let predictions = vec![vec![1usize, 1, 2, 2, 2, 3], vec![0usize, 0, 0]];
        let ground_truth = vec![vec![1usize, 2, 3], vec![0usize, 5]];

        for k in 0..8 {
            let recall = recall_at_k(&predictions, &ground_truth, k);
            let precision = precision_at_k(&predictions, &ground_truth, k);
            assert!((0.0..=1.0).contains(&recall), "recall@{k} = {recall}");
            assert!(
                (0.0..=1.0).contains(&precision),
                "precision@{k} = {precision}"
            );
        }
        let map = mean_average_precision(&predictions, &ground_truth);
        assert!((0.0..=1.0).contains(&map), "MAP = {map}");
    }
}
