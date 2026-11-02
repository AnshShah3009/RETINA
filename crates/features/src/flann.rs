//! FLANN-based approximate nearest neighbor search
//!
//! FLANN (Fast Library for Approximate Nearest Neighbors) uses hierarchical
//! k-means trees and randomized kd-trees for fast approximate matching.

use crate::descriptor::Descriptors;
use cv_core::{FeatureMatch, Matches};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// FLANN index for fast approximate nearest neighbor search
pub struct FlannIndex {
    trees: Vec<KdTree>,
    branching_factor: usize,
    num_trees: usize,
    num_checks: usize,
}

/// KD-Tree node for spatial partitioning
enum KdTreeNode {
    Leaf(Vec<usize>), // Indices of points in this leaf
    Internal {
        dimension: usize,
        threshold: u8,
        left: Box<KdTreeNode>,
        right: Box<KdTreeNode>,
    },
}

/// KD-Tree for efficient nearest neighbor search
struct KdTree {
    root: KdTreeNode,
    points: Vec<Vec<u8>>, // Copy of points for distance computation
}

/// Search result with distance
#[derive(Clone, Copy)]
struct SearchResult {
    index: usize,
    distance: u32,
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.cmp(&self.distance) // Reverse for min-heap
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl FlannIndex {
    /// Create a new FLANN index
    pub fn new(branching_factor: usize, num_trees: usize) -> Self {
        Self {
            trees: Vec::new(),
            branching_factor,
            num_trees,
            num_checks: 32, // Number of leaf nodes to check
        }
    }

    /// Set the number of checks during search (trade-off between speed and accuracy)
    pub fn with_checks(mut self, num_checks: usize) -> Self {
        self.num_checks = num_checks;
        self
    }

    /// Build the index from descriptors
    pub fn build(&mut self, descriptors: &Descriptors) {
        self.trees.clear();

        // Convert descriptors to byte vectors
        let points: Vec<Vec<u8>> = descriptors.iter().map(|d| d.data.clone()).collect();

        // Build multiple randomized KD-trees
        for tree_id in 0..self.num_trees {
            let tree = KdTree::build(&points, self.branching_factor, tree_id);
            self.trees.push(tree);
        }
    }

    /// Search for k nearest neighbors.
    ///
    /// A query whose length differs from the indexed descriptors is not a
    /// neighbor of any of them. Returning them at distance 0 is how a short
    /// query used to match a longer descriptor on the shared prefix alone.
    pub fn search_knn(&self, query: &[u8], k: usize) -> Vec<(usize, u32)> {
        if let Some(stored) = self.trees.first().and_then(|t| t.points.first())
            && stored.len() != query.len()
        {
            return Vec::new();
        }
        let mut candidates = BinaryHeap::new();

        // Search all trees
        for tree in &self.trees {
            tree.search_knn(&tree.root, query, k, self.num_checks, &mut candidates);
        }

        // Convert to sorted results (closest first). into_sorted_vec sorts
        // ascending by Ord, and SearchResult's Ord is distance-REVERSED
        // (min-heap semantics) — sorting by it yields the FARTHEST points.
        // Sort explicitly by raw distance (then index) instead.
        let mut results: Vec<SearchResult> = candidates.into_iter().collect();
        results.sort_by(|a, b| a.distance.cmp(&b.distance).then(a.index.cmp(&b.index)));
        // Every KD-tree indexes the same points, so a neighbor appears once per
        // tree in the merged heap. Collapse duplicate indices before truncating,
        // otherwise k-NN can return the same neighbor twice and the ratio test
        // then rejects an otherwise valid match.
        results.dedup_by_key(|s| s.index);
        results.truncate(k);

        results
            .into_iter()
            .map(|sr| (sr.index, sr.distance))
            .collect()
    }

    /// Find approximate nearest neighbors for all query descriptors
    pub fn knn_search(&self, queries: &Descriptors, k: usize) -> Vec<Vec<(usize, u32)>> {
        queries
            .iter()
            .map(|desc| self.search_knn(&desc.data, k))
            .collect()
    }
}

impl KdTree {
    /// Build a KD-tree from points
    fn build(points: &[Vec<u8>], branching_factor: usize, tree_id: usize) -> Self {
        let indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_recursive(points, &indices, 0, branching_factor, tree_id);

        Self {
            root,
            points: points.to_vec(),
        }
    }

    fn build_recursive(
        points: &[Vec<u8>],
        indices: &[usize],
        depth: usize,
        branching_factor: usize,
        tree_id: usize,
    ) -> KdTreeNode {
        if indices.len() <= branching_factor {
            return KdTreeNode::Leaf(indices.to_vec());
        }

        // `value < median` is an empty side whenever the median equals the
        // minimum (more than half the points share that value). The old
        // code then stored the whole set as one leaf, and search stopped
        // after `num_checks` points, missing an exact copy later in the
        // leaf. Walk dimensions until one splits both sides, and if the
        // median is the minimum pick the first strictly greater value so
        // the `<` side is non-empty.
        let num_dims = points[indices[0]].len();
        for attempt in 0..num_dims {
            let dimension = (depth + tree_id + attempt) % num_dims;
            let mut order: Vec<usize> = indices.to_vec();
            order.sort_by_key(|&i| points[i][dimension]);

            let min_val = points[order[0]][dimension];
            let max_val = points[order[order.len() - 1]][dimension];
            if min_val == max_val {
                continue;
            }

            let mid = order.len() / 2;
            let split_idx = if points[order[mid]][dimension] == min_val {
                order
                    .iter()
                    .position(|&i| points[i][dimension] > min_val)
                    .unwrap_or(mid)
            } else {
                mid
            };
            let threshold = points[order[split_idx]][dimension];
            // Keep the caller's index order. Sorting was only to find the cut.
            let left_indices: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| points[i][dimension] < threshold)
                .collect();
            let right_indices: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| points[i][dimension] >= threshold)
                .collect();
            if left_indices.is_empty() || right_indices.is_empty() {
                continue;
            }

            return KdTreeNode::Internal {
                dimension,
                threshold,
                left: Box::new(Self::build_recursive(
                    points,
                    &left_indices,
                    depth + 1,
                    branching_factor,
                    tree_id,
                )),
                right: Box::new(Self::build_recursive(
                    points,
                    &right_indices,
                    depth + 1,
                    branching_factor,
                    tree_id,
                )),
            };
        }

        KdTreeNode::Leaf(indices.to_vec())
    }

    fn search_knn(
        &self,
        node: &KdTreeNode,
        query: &[u8],
        k: usize,
        max_checks: usize,
        results: &mut BinaryHeap<SearchResult>,
    ) {
        self.search_knn_limited(node, query, k, max_checks, results, &mut 0);
    }

    fn search_knn_limited(
        &self,
        node: &KdTreeNode,
        query: &[u8],
        k: usize,
        max_checks: usize,
        results: &mut BinaryHeap<SearchResult>,
        checks: &mut usize,
    ) {
        // Enforce backtrack budget: stop searching once we have visited max_checks nodes.
        if *checks >= max_checks {
            return;
        }

        match node {
            KdTreeNode::Leaf(indices) => {
                for &idx in indices {
                    if *checks >= max_checks {
                        return;
                    }
                    *checks += 1;
                    let dist = hamming_distance(query, &self.points[idx]);
                    results.push(SearchResult {
                        index: idx,
                        distance: dist,
                    });
                }
            }
            KdTreeNode::Internal {
                dimension,
                threshold,
                left,
                right,
            } => {
                *checks += 1;

                // Choose which side to search first
                let query_val = query[*dimension];
                let (first, second) = if query_val < *threshold {
                    (left, right)
                } else {
                    (right, left)
                };

                // Search the closer side first
                self.search_knn_limited(first, query, k, max_checks, results, checks);

                // Check if we need to search the other side (within budget)
                if *checks < max_checks
                    && (results.len() < k
                        || results.peek().map(|r| r.distance).unwrap_or(u32::MAX) > 0)
                {
                    let diff = query_val.abs_diff(*threshold);

                    // Simple heuristic: if dimension difference is small, check other side
                    if diff < 128 {
                        self.search_knn_limited(second, query, k, max_checks, results, checks);
                    }
                }
            }
        }
    }
}

/// Hamming distance between two binary descriptors.
///
/// Different lengths are not comparable. `zip` would stop at the shorter
/// slice, so an 8-byte descriptor against a 32-byte one measured only the
/// shared prefix and could report 0. That is a confident match between
/// different kinds of descriptor, and a ratio test cannot repair a distance
/// that was already wrong. A mismatch returns the maximum distance.
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    if a.len() != b.len() {
        return u32::MAX;
    }
    // Eight bytes per popcount: the count of set bits in the XOR of two
    // little-endian words is the sum of the counts of their bytes, so this is
    // the byte-wise sum it replaces.
    let (a_words, a_tail) = a.as_chunks::<8>();
    let (b_words, b_tail) = b.as_chunks::<8>();
    let words: u32 = a_words
        .iter()
        .zip(b_words)
        .map(|(x, y)| (u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y)).count_ones())
        .sum();
    words
        + a_tail
            .iter()
            .zip(b_tail)
            .map(|(x, y)| (x ^ y).count_ones())
            .sum::<u32>()
}

/// FLANN-based matcher
pub struct FlannMatcher {
    index: FlannIndex,
    ratio_threshold: Option<f32>,
}

impl Default for FlannMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FlannMatcher {
    /// Create a new `FlannMatcher` with default FLANN index parameters (4 trees, branching factor 32).
    pub fn new() -> Self {
        Self {
            index: FlannIndex::new(32, 4), // 4 trees, branching factor 32
            ratio_threshold: None,
        }
    }

    /// Enable Lowe's ratio test: discard matches where `best / second_best > threshold`.
    pub fn with_ratio_test(mut self, threshold: f32) -> Self {
        self.ratio_threshold = Some(threshold);
        self
    }

    /// Train the matcher with reference descriptors
    pub fn train(&mut self, train_descriptors: &Descriptors) {
        self.index.build(train_descriptors);
    }

    /// Match query descriptors against trained index
    pub fn match_descriptors(&self, query_descriptors: &Descriptors) -> Matches {
        let mut matches = Matches::new();
        let knn_results = self.index.knn_search(query_descriptors, 2);

        for (query_idx, knn) in knn_results.iter().enumerate() {
            if knn.is_empty() {
                continue;
            }

            let best_match = knn[0];

            // Apply ratio test if enabled
            if let Some(ratio) = self.ratio_threshold
                && knn.len() >= 2
            {
                let second_match = knn[1];
                let ratio_calc = best_match.1 as f32 / second_match.1 as f32;
                if ratio_calc > ratio {
                    continue;
                }
            }

            let m = FeatureMatch::new(query_idx as i32, best_match.0 as i32, best_match.1 as f32);
            matches.push(m);
        }

        matches
    }
}

/// Match two sets of descriptors using FLANN
pub fn flann_match(query: &Descriptors, train: &Descriptors, k: usize) -> Vec<Vec<FeatureMatch>> {
    let mut index = FlannIndex::new(32, 4);
    index.build(train);

    let knn_results = index.knn_search(query, k);

    knn_results
        .into_iter()
        .enumerate()
        .map(|(query_idx, neighbors)| {
            neighbors
                .into_iter()
                .map(|(train_idx, distance)| {
                    FeatureMatch::new(query_idx as i32, train_idx as i32, distance as f32)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::Descriptor;
    use cv_core::KeyPoint;

    fn create_test_descriptors(n: usize) -> Descriptors {
        let mut descs = Descriptors::new();
        for i in 0..n {
            let data: Vec<u8> = (0..32).map(|j| ((i * 7 + j * 3) % 256) as u8).collect();
            descs.push(Descriptor::new(data, KeyPoint::new(i as f64, i as f64)));
        }
        descs
    }

    #[test]
    fn test_flann_index() {
        let train = create_test_descriptors(100);
        let mut index = FlannIndex::new(16, 2);
        index.build(&train);

        // Create a query that's identical to one in the training set
        let query: Vec<u8> = (0..32).map(|j| ((7 + j * 3) % 256) as u8).collect();
        let results = index.search_knn(&query, 3);

        assert!(!results.is_empty(), "Should find nearest neighbors");
        // Note: Due to approximate nature, might not find exact match.
        // The contract that must hold is the request size and the ordering.
        assert_eq!(results.len(), 3, "k-nearest must return exactly k results");
        assert!(
            results.windows(2).all(|w| w[0].1 <= w[1].1),
            "results must come back sorted by distance"
        );
    }

    #[test]
    fn test_search_knn_dedups_across_trees() {
        // All KD-trees index the same points, so the merged candidate heap can
        // contain one index per tree. Those duplicates must be collapsed before
        // truncating to k, otherwise the same neighbor is returned more than
        // once and the k=2 ratio test rejects valid matches.
        let train = create_test_descriptors(64);
        let mut index = FlannIndex::new(8, 4);
        index.build(&train);

        let query: Vec<u8> = (0..32).map(|j| ((7 + j * 3) % 256) as u8).collect();
        let results = index.search_knn(&query, 4);

        assert!(results.len() >= 2, "expected multiple neighbors");
        let mut indices: Vec<usize> = results.iter().map(|(i, _)| *i).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(
            indices.len(),
            results.len(),
            "duplicate neighbor indices leaked from the merged KD-tree candidates"
        );
        // Distances must be non-decreasing.
        for pair in results.windows(2) {
            assert!(pair[0].1 <= pair[1].1);
        }
    }

    #[test]
    fn test_flann_matcher() {
        let train = create_test_descriptors(100);
        let query = create_test_descriptors(20);

        // Use lower ratio threshold and without ratio test
        let mut matcher = FlannMatcher::new();
        matcher.train(&train);

        let matches = matcher.match_descriptors(&query);

        // Just verify the pipeline works - exact match count depends on data
        assert!(
            matches.len() <= query.len(),
            "Should not have more matches than queries"
        );
    }

    /// An 8-byte query must not match a 32-byte descriptor that shares its prefix.
    #[test]
    fn mismatched_descriptor_length_is_not_a_neighbor() {
        let train = create_test_descriptors(8);
        let mut index = FlannIndex::new(4, 2);
        index.build(&train);

        let short = vec![0u8; 8];
        assert!(
            index.search_knn(&short, 3).is_empty(),
            "a shorter query was returned as a neighbor"
        );
        assert_eq!(hamming_distance(&short, &[0u8; 32]), u32::MAX);
        assert_eq!(hamming_distance(&[0b1111_0000], &[0b1111_1111]), 4);
    }

    /// 32 copies of `[0, 255]` then one `[0, 0]`. Dim 0 is constant, so a
    /// `value < median` split is empty and the old tree stored every point
    /// in one leaf. With 4 checks the scan stopped on index 0 (distance 8)
    /// and never reached the exact copy at index 32.
    #[test]
    fn exact_copy_after_unbalanced_median_is_found() {
        let mut descs = Descriptors::new();
        for i in 0..32 {
            descs.push(Descriptor::new(vec![0, 255], KeyPoint::new(i as f64, 0.0)));
        }
        descs.push(Descriptor::new(vec![0, 0], KeyPoint::new(32.0, 0.0)));

        let mut index = FlannIndex::new(4, 1).with_checks(4);
        index.build(&descs);
        assert_eq!(index.search_knn(&[0, 0], 1), vec![(32, 0)]);
    }

    fn random_bytes(len: usize, state: &mut u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                *state as u8
            })
            .collect()
    }

    #[test]
    fn word_hamming_equals_the_byte_wise_count() {
        let mut state = 0x0123_4567_89ab_cdefu64;
        for len in 0..=70 {
            for _ in 0..20 {
                let a = random_bytes(len, &mut state);
                let b = random_bytes(len, &mut state);
                let expect: u32 = a.iter().zip(&b).map(|(x, y)| (x ^ y).count_ones()).sum();
                assert_eq!(hamming_distance(&a, &b), expect, "len {len}");
            }
            let zeros = vec![0u8; len];
            let ones = vec![255u8; len];
            assert_eq!(hamming_distance(&zeros, &ones), 8 * len as u32);
        }
    }

    /// The parallel batch search returns, per query and in query order, what a
    /// serial `search_knn` call returns.
    #[test]
    fn knn_search_equals_per_query_search() {
        let mut state = 0xfeed_beef_u64;
        let mut train = Descriptors::new();
        for i in 0..400 {
            train.push(Descriptor::new(
                random_bytes(32, &mut state),
                KeyPoint::new(i as f64, 0.0),
            ));
        }
        let mut queries = Descriptors::new();
        for i in 0..300 {
            // Half exact copies of training descriptors, half fresh.
            let data = if i % 2 == 0 {
                train.descriptors[i].data.clone()
            } else {
                random_bytes(32, &mut state)
            };
            queries.push(Descriptor::new(data, KeyPoint::new(i as f64, 1.0)));
        }
        queries.push(Descriptor::new(vec![0u8; 8], KeyPoint::new(0.0, 0.0)));
        let mut index = FlannIndex::new(32, 4);
        index.build(&train);
        for k in [1, 2, 5] {
            let batch = index.knn_search(&queries, k);
            let serial: Vec<_> = queries
                .iter()
                .map(|q| index.search_knn(&q.data, k))
                .collect();
            assert_eq!(batch, serial, "k = {k}");
        }
    }
}
