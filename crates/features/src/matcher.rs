use crate::descriptor::Descriptors;
use cv_core::{FeatureMatch, Matches};
use rayon::prelude::*;

/// Matching strategy used by [`Matcher`].
pub enum MatchType {
    /// Brute-force L2 distance matching (for float descriptors).
    BruteForce,
    /// Brute-force Hamming distance matching (for binary descriptors).
    BruteForceHamming,
}

/// Feature descriptor matcher with optional ratio test and cross-check filtering.
pub struct Matcher {
    match_type: MatchType,
    cross_check: bool,
    ratio_threshold: Option<f32>,
}

impl Matcher {
    /// Create a new matcher with the given matching strategy.
    pub fn new(match_type: MatchType) -> Self {
        Self {
            match_type,
            cross_check: false,
            ratio_threshold: None,
        }
    }

    /// Enable cross-check filtering (keep a match only if it is the best match in both directions).
    pub fn with_cross_check(mut self) -> Self {
        self.cross_check = true;
        self
    }

    /// Enable Lowe's ratio test: discard matches where `best / second_best > threshold`.
    pub fn with_ratio_test(mut self, threshold: f32) -> Self {
        self.ratio_threshold = Some(threshold);
        self
    }

    /// Match each query descriptor to the best training descriptor.
    pub fn match_descriptors(&self, query: &Descriptors, train: &Descriptors) -> Matches {
        match self.match_type {
            MatchType::BruteForce | MatchType::BruteForceHamming => {
                self.brute_force_match(query, train)
            }
        }
    }

    fn brute_force_match(&self, query: &Descriptors, train: &Descriptors) -> Matches {
        let matches: Vec<FeatureMatch> = (0..query.len())
            .into_par_iter()
            .filter_map(|query_idx| {
                let q_desc = &query.descriptors[query_idx];
                let mut best_match: Option<(usize, u32)> = None;
                let mut second_best: Option<u32> = None;

                for (train_idx, t_desc) in train.iter().enumerate() {
                    let distance = q_desc.hamming_distance(t_desc);

                    match best_match {
                        None => {
                            best_match = Some((train_idx, distance));
                        }
                        Some((_, best_dist)) => {
                            if distance < best_dist {
                                second_best = Some(best_dist);
                                best_match = Some((train_idx, distance));
                            } else if second_best.is_none() || distance < second_best.unwrap() {
                                second_best = Some(distance);
                            }
                        }
                    }
                }

                if let Some((train_idx, distance)) = best_match {
                    let mut keep_match = true;

                    if let Some(threshold) = self.ratio_threshold {
                        if let Some(second) = second_best {
                            let ratio = distance as f32 / second as f32;
                            if ratio > threshold {
                                keep_match = false;
                            }
                        }
                    }

                    if self.cross_check {
                        let reverse_match = self.find_best_match(train, query, train_idx);
                        if reverse_match != Some(query_idx) {
                            keep_match = false;
                        }
                    }

                    if keep_match {
                        return Some(FeatureMatch::new(
                            query_idx as i32,
                            train_idx as i32,
                            distance as f32,
                        ));
                    }
                }
                None
            })
            .collect();

        Matches {
            matches,
            mask: None,
        }
    }

    fn find_best_match(
        &self,
        query: &Descriptors,
        train: &Descriptors,
        query_idx: usize,
    ) -> Option<usize> {
        let q_desc = query.descriptors.get(query_idx)?;

        let mut best_idx = 0;
        let mut best_dist = q_desc.hamming_distance(&train.descriptors[0]);

        for (idx, t_desc) in train.iter().enumerate().skip(1) {
            let distance = q_desc.hamming_distance(t_desc);
            if distance < best_dist {
                best_dist = distance;
                best_idx = idx;
            }
        }

        Some(best_idx)
    }
}

/// Match descriptors using brute-force Hamming distance with an optional ratio test.
pub fn match_descriptors(
    query: &Descriptors,
    train: &Descriptors,
    ratio_threshold: Option<f32>,
) -> Matches {
    let mut matcher = Matcher::new(MatchType::BruteForceHamming);

    if let Some(threshold) = ratio_threshold {
        matcher = matcher.with_ratio_test(threshold);
    }

    matcher.match_descriptors(query, train)
}

/// K-nearest-neighbour matching: return the `k` closest training descriptors for each query.
pub fn knn_match(query: &Descriptors, train: &Descriptors, k: usize) -> Vec<Vec<FeatureMatch>> {
    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(query.len());

    for (query_idx, q_desc) in query.iter().enumerate() {
        let mut distances: Vec<(usize, u32)> = train
            .iter()
            .enumerate()
            .map(|(idx, t_desc)| (idx, q_desc.hamming_distance(t_desc)))
            .collect();

        distances.sort_by(|a, b| a.1.cmp(&b.1));

        let knn: Vec<FeatureMatch> = distances
            .into_iter()
            .take(k)
            .map(|(train_idx, distance)| {
                FeatureMatch::new(query_idx as i32, train_idx as i32, distance as f32)
            })
            .collect();

        all_matches.push(knn);
    }

    all_matches
}

/// Remove all matches with a distance greater than `max_distance`.
pub fn filter_matches_by_distance(matches: &mut Matches, max_distance: f32) {
    matches.filter_by_distance(max_distance);
}

/// Apply Lowe's ratio test to KNN match results.
///
/// Keeps a match from the `k=2` result only if `best.distance < ratio * second.distance`.
pub fn filter_matches_by_ratio_test(
    matches: &[Vec<FeatureMatch>],
    ratio: f32,
) -> Vec<FeatureMatch> {
    let mut good_matches = Vec::new();

    for knn in matches {
        if knn.len() >= 2 {
            let best = &knn[0];
            let second = &knn[1];

            if best.distance < ratio * second.distance {
                good_matches.push(*best);
            }
        }
    }

    good_matches
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;
    use cv_core::KeyPoint;

    fn make_descriptors_from_bytes(values: &[&[u8]]) -> Descriptors {
        let mut ds = Descriptors::new();
        for (i, &v) in values.iter().enumerate() {
            let kp = KeyPoint::new(i as f64, 0.0);
            ds.push(crate::descriptor::Descriptor::new(v.to_vec(), kp));
        }
        ds
    }

    #[test]
    fn matcher_new() {
        let m = Matcher::new(MatchType::BruteForce);
        assert!(!m.cross_check);
        assert!(m.ratio_threshold.is_none());
    }

    #[test]
    fn matcher_with_cross_check() {
        let m = Matcher::new(MatchType::BruteForce).with_cross_check();
        assert!(m.cross_check);
    }

    #[test]
    fn matcher_with_ratio_test() {
        let m = Matcher::new(MatchType::BruteForce).with_ratio_test(0.7);
        assert_eq!(m.ratio_threshold, Some(0.7));
    }

    #[test]
    fn match_descriptors_identical_sets() {
        let query = make_descriptors_from_bytes(&[&[0u8; 8], &[0xFFu8; 8]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 8], &[0xFFu8; 8]]);
        let result = match_descriptors(&query, &train, None);
        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.iter().all(|m| (m.distance as u32) == 0));
    }

    #[test]
    fn match_descriptors_empty_query() {
        let query = Descriptors::new();
        let train = make_descriptors_from_bytes(&[&[0u8; 8]]);
        let result = match_descriptors(&query, &train, None);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn match_descriptors_empty_train() {
        let query = make_descriptors_from_bytes(&[&[0u8; 8]]);
        let train = Descriptors::new();
        let result = match_descriptors(&query, &train, None);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn match_descriptors_with_ratio_filter() {
        let query = make_descriptors_from_bytes(&[&[0u8; 1]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 1], &[0xFFu8; 1]]);
        let result = match_descriptors(&query, &train, Some(0.75));
        assert_eq!(result.matches.len(), 1);
    }

    #[test]
    fn knn_match_returns_k_results() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 4], &[0x0Fu8; 4], &[0xFFu8; 4]]);
        let results = knn_match(&query, &train, 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);
        assert!(results[0][0].distance <= results[0][1].distance);
    }

    #[test]
    fn knn_match_fewer_than_k() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 4], &[0xFFu8; 4]]);
        let results = knn_match(&query, &train, 5);
        assert_eq!(results[0].len(), 2);
    }

    #[test]
    fn knn_match_empty_train() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = Descriptors::new();
        let results = knn_match(&query, &train, 2);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn filter_matches_by_distance_boundary() {
        let mut m = Matches {
            matches: vec![
                FeatureMatch::new(0, 0, 0.3),
                FeatureMatch::new(1, 1, 0.5),
                FeatureMatch::new(2, 2, 0.8),
            ],
            mask: None,
        };
        filter_matches_by_distance(&mut m, 0.5);
        assert_eq!(m.matches.len(), 2);
    }

    #[test]
    fn filter_matches_by_ratio_test_accepts_good() {
        let matches = vec![vec![
            FeatureMatch::new(0, 0, 0.3),
            FeatureMatch::new(0, 1, 1.0),
        ]];
        let result = filter_matches_by_ratio_test(&matches, 0.75);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn filter_matches_by_ratio_test_rejects_ambiguous() {
        let matches = vec![vec![
            FeatureMatch::new(0, 0, 0.8),
            FeatureMatch::new(0, 1, 0.9),
        ]];
        let result = filter_matches_by_ratio_test(&matches, 0.75);
        assert!(result.is_empty());
    }

    #[test]
    fn filter_matches_single_neighbor_rejected() {
        let matches = vec![vec![FeatureMatch::new(0, 0, 0.1)]];
        let result = filter_matches_by_ratio_test(&matches, 0.75);
        assert!(result.is_empty());
    }
}
