use crate::descriptor::Descriptors;
use cv_core::{FeatureMatch, Matches};
use rayon::prelude::*;

/// Matching strategy used by [`Matcher`].
pub enum MatchType {
    /// Brute-force Hamming distance matching (binary descriptors). Alias of
    /// [`MatchType::BruteForceHamming`]. Note this differs from OpenCV, whose
    /// `"BruteForce"` is L2; it is kept as Hamming because the workspace's ORB
    /// pipelines (sfm, localization) rely on it.
    BruteForce,
    /// Brute-force Hamming distance matching (for binary descriptors: ORB,
    /// BRIEF, AKAZE-MLDB).
    BruteForceHamming,
    /// Brute-force Euclidean (L2) distance over the descriptor bytes, read as
    /// unsigned values - OpenCV's `NORM_L2` on `CV_8U` descriptors. Use this
    /// for gradient-histogram descriptors such as SIFT: Hamming on those
    /// bytes is meaningless (query byte 128 is Hamming 8 from 127 but 1 from
    /// 0, so the far descriptor wins).
    BruteForceL2,
}

impl MatchType {
    fn distance(&self, a: &cv_core::Descriptor, b: &cv_core::Descriptor) -> f32 {
        match self {
            MatchType::BruteForce | MatchType::BruteForceHamming => a.hamming_distance(b) as f32,
            MatchType::BruteForceL2 => {
                // Unequal lengths are not a match. `zip` stops at the shorter
                // slice, so an 8-byte zero descriptor against a 128-byte zero
                // descriptor used to report distance 0 and win. Hamming already
                // returns this sentinel for the same reason.
                if a.data.len() != b.data.len() {
                    return u32::MAX as f32;
                }
                a.data
                    .iter()
                    .zip(&b.data)
                    .map(|(&x, &y)| {
                        let d = x as f32 - y as f32;
                        d * d
                    })
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }
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
        self.brute_force_match(query, train)
    }

    fn brute_force_match(&self, query: &Descriptors, train: &Descriptors) -> Matches {
        // Distances are evaluated on descriptors packed into flat buffers when
        // every descriptor has the same length (see `PackedDistance`), which
        // gives exactly the values `MatchType::distance` would.
        match PackedDistance::new(&self.match_type, query, train) {
            Some(PackedDistance::Hamming(p)) => {
                self.brute_force_match_with(query.len(), train.len(), |q, t| p.distance(q, t))
            }
            Some(PackedDistance::L2(p)) => {
                self.brute_force_match_with(query.len(), train.len(), |q, t| p.distance(q, t))
            }
            None => self.brute_force_match_with(query.len(), train.len(), |q, t| {
                self.match_type
                    .distance(&query.descriptors[q], &train.descriptors[t])
            }),
        }
    }

    /// Brute-force matching over `n_query` x `n_train` pairs, with
    /// `distance(query_idx, train_idx)` giving each pair's distance (which is
    /// symmetric for every [`MatchType`]).
    fn brute_force_match_with<D>(&self, n_query: usize, n_train: usize, distance: D) -> Matches
    where
        D: Fn(usize, usize) -> f32 + Sync,
    {
        // One reverse pass: the winning query index for each training
        // descriptor. Ties keep the lower index (replace only when the new
        // distance is strictly smaller).
        let reverse_best = if self.cross_check {
            best_query_per_train(n_query, n_train, &distance)
        } else {
            Vec::new()
        };

        let matches: Vec<FeatureMatch> = (0..n_query)
            .into_par_iter()
            .filter_map(|query_idx| {
                let mut best_match: Option<(usize, f32)> = None;
                let mut second_best: Option<f32> = None;

                for train_idx in 0..n_train {
                    let distance = distance(query_idx, train_idx);

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

                    if let Some(threshold) = self.ratio_threshold
                        && let Some(second) = second_best
                    {
                        // A zero second-best distance means the match is
                        // ambiguous: two train descriptors are identical, or
                        // as close as the encoding allows. That is the most
                        // ambiguous case the ratio test exists to reject, and
                        // it is exactly the one `0/0` fails on - IEEE-754
                        // gives NaN, and `NaN > threshold` is false, so the
                        // match would be *kept*.
                        //
                        // Such matches are common wherever an image has
                        // repeated texture, and each one that survives flows
                        // into track building and PnP looking confident.
                        let rejected = if second == 0.0 {
                            true
                        } else {
                            distance / second > threshold
                        };
                        if rejected {
                            keep_match = false;
                        }
                    }

                    if self.cross_check
                        && reverse_best.get(train_idx).copied().flatten() != Some(query_idx)
                    {
                        keep_match = false;
                    }

                    if keep_match {
                        return Some(FeatureMatch::new(
                            query_idx as i32,
                            train_idx as i32,
                            distance,
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
}

/// Winning query index for every training descriptor.
///
/// An empty query set leaves each entry `None`. Equal distances keep the
/// lower query index.
fn best_query_per_train<D>(n_query: usize, n_train: usize, distance: &D) -> Vec<Option<usize>>
where
    D: Fn(usize, usize) -> f32 + Sync,
{
    (0..n_train)
        .into_par_iter()
        .map(|train_idx| {
            let mut best: Option<(usize, f32)> = None;
            for query_idx in 0..n_query {
                let distance = distance(query_idx, train_idx);
                match best {
                    Some((_, best_dist)) if best_dist <= distance => {}
                    _ => best = Some((query_idx, distance)),
                }
            }
            best.map(|(idx, _)| idx)
        })
        .collect()
}

/// Query and train descriptors copied into flat, fixed-stride buffers.
///
/// Every `Descriptor` owns its own heap allocation, so a brute-force scan over
/// them chases a pointer per pair and reads bytes one at a time. When all the
/// descriptors on both sides have the same length the distances can instead be
/// computed on contiguous rows. Mixed lengths keep the per-descriptor path,
/// which owns the length-mismatch sentinel.
enum PackedDistance {
    Hamming(PackedHamming),
    L2(PackedL2),
}

impl PackedDistance {
    fn new(match_type: &MatchType, query: &Descriptors, train: &Descriptors) -> Option<Self> {
        let len = query.descriptors.first()?.data.len();
        let uniform = query
            .descriptors
            .iter()
            .chain(train.descriptors.iter())
            .all(|d| d.data.len() == len);
        if !uniform {
            return None;
        }
        match match_type {
            MatchType::BruteForce | MatchType::BruteForceHamming => {
                Some(Self::Hamming(PackedHamming::new(len, query, train)))
            }
            MatchType::BruteForceL2 => PackedL2::new(len, query, train).map(Self::L2),
        }
    }
}

/// Descriptors as little-endian `u64` words, the last word zero-padded.
///
/// The padding bytes are zero on both sides, so they XOR to zero and add
/// nothing: the popcount of the words is the byte-wise Hamming distance.
struct PackedHamming {
    words: usize,
    query: Vec<u64>,
    train: Vec<u64>,
}

impl PackedHamming {
    fn new(len: usize, query: &Descriptors, train: &Descriptors) -> Self {
        let words = len.div_ceil(8);
        let pack = |set: &Descriptors| -> Vec<u64> {
            let mut out = Vec::with_capacity(set.len() * words);
            for d in &set.descriptors {
                for chunk in d.data.chunks(8) {
                    let mut bytes = [0u8; 8];
                    bytes[..chunk.len()].copy_from_slice(chunk);
                    out.push(u64::from_le_bytes(bytes));
                }
            }
            out
        };
        Self {
            words,
            query: pack(query),
            train: pack(train),
        }
    }

    #[inline]
    fn distance_u32(&self, query_idx: usize, train_idx: usize) -> u32 {
        let q = &self.query[query_idx * self.words..(query_idx + 1) * self.words];
        let t = &self.train[train_idx * self.words..(train_idx + 1) * self.words];
        q.iter().zip(t).map(|(a, b)| (a ^ b).count_ones()).sum()
    }

    #[inline]
    fn distance(&self, query_idx: usize, train_idx: usize) -> f32 {
        self.distance_u32(query_idx, train_idx) as f32
    }
}

/// Descriptor bytes in one flat buffer, for the L2 norm in integer arithmetic.
struct PackedL2 {
    len: usize,
    query: Vec<u8>,
    train: Vec<u8>,
}

impl PackedL2 {
    /// Longest descriptor whose sum of squared byte differences stays below
    /// 2^24 (`258 * 255^2 = 16_776_450`).
    ///
    /// Up to that bound every partial sum `MatchType::distance` accumulates in
    /// `f32` is an exactly representable integer, so its `f32` sum equals the
    /// integer sum and the square roots agree bit for bit. Longer descriptors
    /// (the f32 sum could round) keep the `f32` path.
    const MAX_EXACT_LEN: usize = 258;

    fn new(len: usize, query: &Descriptors, train: &Descriptors) -> Option<Self> {
        // An empty descriptor is left to the f32 path, whose empty sum is -0.0.
        if len == 0 || len > Self::MAX_EXACT_LEN {
            return None;
        }
        let pack = |set: &Descriptors| -> Vec<u8> {
            set.descriptors
                .iter()
                .flat_map(|d| d.data.iter().copied())
                .collect()
        };
        Some(Self {
            len,
            query: pack(query),
            train: pack(train),
        })
    }

    #[inline]
    fn distance(&self, query_idx: usize, train_idx: usize) -> f32 {
        let q = &self.query[query_idx * self.len..(query_idx + 1) * self.len];
        let t = &self.train[train_idx * self.len..(train_idx + 1) * self.len];
        let sum: u32 = q
            .iter()
            .zip(t)
            .map(|(&x, &y)| {
                let d = x.abs_diff(y) as u32;
                d * d
            })
            .sum();
        (sum as f32).sqrt()
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

/// Largest `k` for which [`knn_match`] keeps a sorted insertion list rather
/// than collecting and selecting over every train distance.
const KNN_INSERTION_MAX_K: usize = 16;

/// K-nearest-neighbour matching: return the `k` closest training descriptors for each query.
///
/// Neighbours are ordered by Hamming distance, equal distances by ascending
/// train index.
pub fn knn_match(query: &Descriptors, train: &Descriptors, k: usize) -> Vec<Vec<FeatureMatch>> {
    let packed = match PackedDistance::new(&MatchType::BruteForceHamming, query, train) {
        Some(PackedDistance::Hamming(p)) => Some(p),
        _ => None,
    };

    // Queries are independent; the indexed parallel collect keeps query order.
    (0..query.len())
        .into_par_iter()
        .map(|query_idx| {
            let distance = |train_idx: usize| match &packed {
                Some(p) => p.distance_u32(query_idx, train_idx),
                None => {
                    query.descriptors[query_idx].hamming_distance(&train.descriptors[train_idx])
                }
            };

            // The first `k` by (distance, train index): the same neighbours in
            // the same order a stable sort on distance alone gives.
            let distances: Vec<(u32, usize)> = if k <= KNN_INSERTION_MAX_K {
                // Small k: keep a sorted list of the best `k` seen so far.
                // Train indices arrive ascending, so a candidate tying the
                // current k-th distance sorts after it and is not taken.
                let mut best: Vec<(u32, usize)> = Vec::with_capacity(k + 1);
                for train_idx in 0..train.len() {
                    let d = distance(train_idx);
                    if best.len() == k && best.last().is_none_or(|&(worst, _)| d >= worst) {
                        continue;
                    }
                    let at = best.partition_point(|&(bd, _)| bd <= d);
                    best.insert(at, (d, train_idx));
                    best.truncate(k);
                }
                best
            } else {
                // Large k: partial selection instead of sorting the whole set.
                let mut all: Vec<(u32, usize)> =
                    (0..train.len()).map(|t| (distance(t), t)).collect();
                if k < all.len() {
                    all.select_nth_unstable(k - 1);
                    all.truncate(k);
                }
                all.sort_unstable();
                all
            };

            distances
                .into_iter()
                .map(|(distance, train_idx)| {
                    FeatureMatch::new(query_idx as i32, train_idx as i32, distance as f32)
                })
                .collect()
        })
        .collect()
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
        let query = make_descriptors_from_bytes(&[&[0u8; 8], &[0xffu8; 8]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 8], &[0xffu8; 8]]);
        let result = match_descriptors(&query, &train, None);
        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.iter().all(|m| (m.distance as u32) == 0));
    }

    #[test]
    fn brute_force_variants_use_hamming() {
        // `BruteForce` and `BruteForceHamming` are documented aliases: both run
        // the Hamming matcher, so their result sets must be identical.
        let query = make_descriptors_from_bytes(&[&[0x0fu8; 4], &[0xf0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0x00u8; 4], &[0xffu8; 4]]);
        let a = Matcher::new(MatchType::BruteForce).match_descriptors(&query, &train);
        let b = Matcher::new(MatchType::BruteForceHamming).match_descriptors(&query, &train);
        let av: Vec<_> = a
            .matches
            .iter()
            .map(|m| (m.query_idx, m.train_idx, m.distance))
            .collect();
        let bv: Vec<_> = b
            .matches
            .iter()
            .map(|m| (m.query_idx, m.train_idx, m.distance))
            .collect();
        assert_eq!(av, bv);
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
        let train = make_descriptors_from_bytes(&[&[0u8; 1], &[0xffu8; 1]]);
        let result = match_descriptors(&query, &train, Some(0.75));
        assert_eq!(result.matches.len(), 1);
    }

    #[test]
    fn knn_match_returns_k_results() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 4], &[0x0fu8; 4], &[0xffu8; 4]]);
        let results = knn_match(&query, &train, 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);
        assert!(results[0][0].distance <= results[0][1].distance);
    }

    #[test]
    fn knn_match_fewer_than_k() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 4], &[0xffu8; 4]]);
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

    #[test]
    fn cross_check_keeps_mutual_best_and_drops_one_way() {
        // Query 1 is an exact copy of train 0. Query 0 prefers train 0 over
        // train 1, but train 0's nearest query is 1, so that one-way match
        // is dropped and the mutual match is kept.
        let query = make_descriptors_from_bytes(&[&[10u8], &[0u8]]);
        let train = make_descriptors_from_bytes(&[&[0u8], &[100u8]]);
        let matches = Matcher::new(MatchType::BruteForceL2)
            .with_cross_check()
            .match_descriptors(&query, &train);
        assert_eq!(matches.matches.len(), 1);
        let m = matches.matches[0];
        assert_eq!((m.query_idx, m.train_idx), (1, 0));
        assert!(m.distance.abs() < 1e-6);
    }

    #[test]
    fn cross_check_tie_keeps_the_lower_query_index() {
        let query = make_descriptors_from_bytes(&[&[0u8], &[0u8]]);
        let train = make_descriptors_from_bytes(&[&[0u8]]);
        let matches = Matcher::new(MatchType::BruteForceL2)
            .with_cross_check()
            .match_descriptors(&query, &train);
        assert_eq!(matches.matches.len(), 1);
        assert_eq!(matches.matches[0].query_idx, 0);
        assert_eq!(matches.matches[0].train_idx, 0);
    }

    fn random_descriptors(n: usize, len: usize, seed: u64) -> Descriptors {
        let mut state = seed;
        let mut ds = Descriptors::new();
        for i in 0..n {
            let data = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // Mix in the extremes so the L2 bound is exercised.
                    match state % 5 {
                        0 => 0,
                        1 => 255,
                        _ => (state >> 8) as u8,
                    }
                })
                .collect();
            ds.push(crate::descriptor::Descriptor::new(
                data,
                KeyPoint::new(i as f64, 0.0),
            ));
        }
        ds
    }

    /// The packed kernels give exactly `MatchType::distance` for every pair,
    /// for lengths that do and do not fill whole words, up to the longest
    /// descriptor the integer L2 path accepts (all-0 against all-255 there).
    #[test]
    fn packed_distances_equal_the_per_descriptor_distance() {
        for (len, seed) in [(1, 3u64), (32, 5), (61, 7), (128, 11), (258, 13)] {
            let mut q = random_descriptors(17, len, seed);
            let mut t = random_descriptors(23, len, seed * 31);
            q.push(crate::descriptor::Descriptor::new(
                vec![0; len],
                KeyPoint::new(0.0, 0.0),
            ));
            t.push(crate::descriptor::Descriptor::new(
                vec![255; len],
                KeyPoint::new(0.0, 0.0),
            ));
            let hamming = PackedHamming::new(len, &q, &t);
            let l2 = PackedL2::new(len, &q, &t).expect("length within the exact bound");
            for (qi, qd) in q.descriptors.iter().enumerate() {
                for (ti, td) in t.descriptors.iter().enumerate() {
                    let h = MatchType::BruteForceHamming.distance(qd, td);
                    assert_eq!(hamming.distance(qi, ti).to_bits(), h.to_bits());
                    let e = MatchType::BruteForceL2.distance(qd, td);
                    assert_eq!(l2.distance(qi, ti).to_bits(), e.to_bits(), "len {len}");
                }
            }
        }
        let q = random_descriptors(2, 259, 1);
        assert!(PackedL2::new(259, &q, &q).is_none());
    }

    /// Mixed descriptor lengths keep the per-descriptor path and its
    /// length-mismatch sentinel.
    #[test]
    fn mixed_lengths_are_not_packed() {
        let query = make_descriptors_from_bytes(&[&[0u8; 4]]);
        let train = make_descriptors_from_bytes(&[&[0u8; 4], &[0u8; 8]]);
        assert!(PackedDistance::new(&MatchType::BruteForceHamming, &query, &train).is_none());
        let m = Matcher::new(MatchType::BruteForceHamming).match_descriptors(&query, &train);
        assert_eq!(m.matches.len(), 1);
        assert_eq!(m.matches[0].train_idx, 0);
    }

    /// `knn_match` returns what sorting every train descriptor stably by
    /// Hamming distance and taking the first `k` does - including the order of
    /// tied distances, which short descriptors make plentiful.
    #[test]
    fn knn_match_equals_a_full_stable_sort() {
        let reference = |q: &Descriptors, t: &Descriptors, k: usize| -> Vec<Vec<(i32, i32, u32)>> {
            q.iter()
                .enumerate()
                .map(|(qi, qd)| {
                    let mut d: Vec<(usize, u32)> = t
                        .iter()
                        .enumerate()
                        .map(|(ti, td)| (ti, qd.hamming_distance(td)))
                        .collect();
                    d.sort_by_key(|a| a.1);
                    d.into_iter()
                        .take(k)
                        .map(|(ti, dist)| (qi as i32, ti as i32, dist))
                        .collect()
                })
                .collect()
        };
        let flatten = |r: Vec<Vec<FeatureMatch>>| -> Vec<Vec<(i32, i32, u32)>> {
            r.into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|m| (m.query_idx, m.train_idx, m.distance as u32))
                        .collect()
                })
                .collect()
        };
        for len in [1usize, 2, 32] {
            let q = random_descriptors(40, len, 99 + len as u64);
            let t = random_descriptors(300, len, 7 + len as u64);
            for k in [0usize, 1, 2, 5, 299, 300, 301] {
                assert_eq!(flatten(knn_match(&q, &t, k)), reference(&q, &t, k));
            }
        }
        // Mixed lengths take the per-descriptor path with its sentinel.
        let q = make_descriptors_from_bytes(&[&[0u8; 4], &[1u8; 4]]);
        let t = make_descriptors_from_bytes(&[&[0u8; 8], &[3u8; 4], &[0u8; 4]]);
        assert_eq!(flatten(knn_match(&q, &t, 3)), reference(&q, &t, 3));
    }
}
