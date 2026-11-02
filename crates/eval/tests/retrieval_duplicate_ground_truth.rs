//! The relevant set of a query is a *set*. A ground-truth list that repeats an
//! id names that item once.
//!
//! The ranked predictions were already deduplicated ("a repeated id counts
//! once", matching `trec_eval`, whose qrels are a set), but the ground truth
//! was not: `recall_at_k` and `mean_average_precision` divided by
//! `gt.len()`, so `gt = [2, 2]` made a perfect retrieval of item 2 score 0.5.
//! Ground-truth lists built by concatenating matches (every database frame
//! within a radius of each query pose, say) repeat ids routinely.

use cv_eval::retrieval::{mean_average_precision, recall_at_k};

#[test]
fn repeated_ground_truth_ids_count_once_in_recall() {
    let predictions = vec![vec![2usize, 9]];
    let ground_truth = vec![vec![2usize, 2]];
    assert_eq!(recall_at_k(&predictions, &ground_truth, 1), 1.0);
    // Two relevant items, one found.
    assert_eq!(
        recall_at_k(&[vec![2usize, 9]], &[vec![2usize, 5, 2, 5]], 2),
        0.5
    );
}

#[test]
fn repeated_ground_truth_ids_count_once_in_map() {
    let predictions = vec![vec![2usize, 9]];
    let ground_truth = vec![vec![2usize, 2]];
    assert_eq!(mean_average_precision(&predictions, &ground_truth), 1.0);
    // Relevant {2, 5}: 2 at rank 1 (precision 1), 5 at rank 3 (precision 2/3).
    let ap = mean_average_precision(&[vec![2usize, 9, 5]], &[vec![5usize, 2, 5, 2]]);
    assert!((ap - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-12, "{ap}");
}
