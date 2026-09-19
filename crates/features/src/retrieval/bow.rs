//! Sparse bag-of-words vectors with TF-IDF weighting and L1/L2 scoring.

use crate::retrieval::Vocabulary;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Sparse bag-of-words vector.
///
/// `entries` holds `(word_id, weight)` pairs sorted ascending by word id. Weights
/// are TF-IDF values: the term frequency is the fraction of the descriptor set
/// assigned to the word and the IDF comes from the vocabulary's training
/// statistics.
#[derive(Debug, Clone, Default)]
pub struct BowVector {
    /// Word id and its weight, sorted ascending by word id.
    pub entries: Vec<(u32, f32)>,
}

impl BowVector {
    /// Build a TF-IDF bag-of-words vector for `descriptors` using `vocabulary`.
    ///
    /// Each descriptor is assigned to its nearest word. The weight of word `w`
    /// is `tf(w) * idf(w)` with `tf(w) = count(w) / total`. Entries are sorted by
    /// word id. Empty input (no descriptors, or an empty vocabulary) yields an
    /// empty vector.
    pub fn from_descriptors<D: AsRef<[u8]>>(vocabulary: &Vocabulary, descriptors: &[D]) -> Self {
        let total = descriptors.len();
        if total == 0 || vocabulary.k() == 0 {
            return Self {
                entries: Vec::new(),
            };
        }

        let mut counts: HashMap<u32, u32> = HashMap::new();
        for d in descriptors {
            let (word, _) = vocabulary.nearest_word(d.as_ref());
            *counts.entry(word).or_insert(0) += 1;
        }

        let mut entries: Vec<(u32, f32)> = counts
            .into_iter()
            .map(|(word, count)| (word, (count as f32 / total as f32) * vocabulary.idf(word)))
            .collect();
        entries.sort_unstable_by_key(|&(word, _)| word);

        Self { entries }
    }

    /// Number of distinct words in the vector.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vector contains no words.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Weight of `word`, or `0.0` if the word is absent.
    pub fn get(&self, word: u32) -> f32 {
        match self.entries.binary_search_by_key(&word, |&(w, _)| w) {
            Ok(i) => self.entries[i].1,
            Err(_) => 0.0,
        }
    }

    /// L1 norm (sum of absolute weights).
    pub fn norm_l1(&self) -> f32 {
        self.entries.iter().map(|&(_, w)| w.abs()).sum()
    }

    /// L2 norm (Euclidean length).
    pub fn norm_l2(&self) -> f32 {
        self.entries.iter().map(|&(_, w)| w * w).sum::<f32>().sqrt()
    }

    /// Rescale the vector in place so its L1 norm is `1` (no-op when zero).
    pub fn normalize_l1(&mut self) {
        let norm = self.norm_l1();
        if norm > 0.0 {
            for entry in &mut self.entries {
                entry.1 /= norm;
            }
        }
    }

    /// Rescale the vector in place so its L2 norm is `1` (no-op when zero).
    pub fn normalize_l2(&mut self) {
        let norm = self.norm_l2();
        if norm > 0.0 {
            for entry in &mut self.entries {
                entry.1 /= norm;
            }
        }
    }

    /// L1 distance `sum_w |a_w - b_w|`.
    pub fn l1_distance(a: &BowVector, b: &BowVector) -> f32 {
        merge_accumulate(a, b, |acc, x, y| acc + (x - y).abs())
    }

    /// L2 distance `sqrt(sum_w (a_w - b_w)^2)`.
    pub fn l2_distance(a: &BowVector, b: &BowVector) -> f32 {
        merge_accumulate(a, b, |acc, x, y| acc + (x - y) * (x - y)).sqrt()
    }

    /// L1-based similarity, **higher is more similar**.
    ///
    /// Returns `2 - l1_distance`; for L1-normalized non-negative vectors this
    /// lies in `[0, 2]` and identical vectors score `2`.
    pub fn score_l1(a: &BowVector, b: &BowVector) -> f32 {
        2.0 - Self::l1_distance(a, b)
    }

    /// L2-based similarity in `(0, 1]`, **higher is more similar**.
    ///
    /// Returns `1 / (1 + l2_distance)`; identical vectors score `1`.
    pub fn score_l2(a: &BowVector, b: &BowVector) -> f32 {
        1.0 / (1.0 + Self::l2_distance(a, b))
    }
}

/// Accumulate `f(weight_a, weight_b)` over the union of two sparse vectors,
/// treating absent words as weight `0`.
fn merge_accumulate(a: &BowVector, b: &BowVector, mut f: impl FnMut(f32, f32, f32) -> f32) -> f32 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut acc = 0.0f32;
    while i < a.entries.len() || j < b.entries.len() {
        match (a.entries.get(i), b.entries.get(j)) {
            (Some(&(wa, va)), Some(&(wb, vb))) => match wa.cmp(&wb) {
                Ordering::Equal => {
                    acc = f(acc, va, vb);
                    i += 1;
                    j += 1;
                }
                Ordering::Less => {
                    acc = f(acc, va, 0.0);
                    i += 1;
                }
                Ordering::Greater => {
                    acc = f(acc, 0.0, vb);
                    j += 1;
                }
            },
            (Some(&(_, va)), None) => {
                acc = f(acc, va, 0.0);
                i += 1;
            }
            (None, Some(&(_, vb))) => {
                acc = f(acc, 0.0, vb);
                j += 1;
            }
            (None, None) => break,
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::test_support::{random_descriptor, seeded};
    use rand::Rng;

    fn sample_vocabulary() -> (Vocabulary, Vec<Vec<u8>>) {
        let mut rng = seeded(11);
        let pool: Vec<Vec<u8>> = (0..40).map(|_| random_descriptor(&mut rng, 32)).collect();
        let vocab = Vocabulary::train(&pool, 40, 10, 5);
        (vocab, pool)
    }

    #[test]
    fn bow_entries_sorted_positive_and_normalizable() {
        let (vocab, pool) = sample_vocabulary();
        let bow = BowVector::from_descriptors(&vocab, &pool[..12]);

        assert!(!bow.is_empty());
        assert!(
            bow.entries.windows(2).all(|w| w[0].0 < w[1].0),
            "entries must be sorted and unique"
        );
        assert!(bow.entries.iter().all(|&(_, weight)| weight > 0.0));

        assert!(bow.norm_l1() > 0.0);
        assert!(bow.norm_l2() > 0.0);

        let mut l1 = bow.clone();
        l1.normalize_l1();
        assert!((l1.norm_l1() - 1.0).abs() < 1e-5);

        let mut l2 = bow.clone();
        l2.normalize_l2();
        assert!((l2.norm_l2() - 1.0).abs() < 1e-5);

        // Absent words read as zero weight.
        assert_eq!(bow.get(u32::MAX), 0.0);
        assert!(bow.get(bow.entries[0].0) > 0.0);
    }

    #[test]
    fn own_descriptors_score_highest() {
        let (vocab, pool) = sample_vocabulary();

        // Build a distinct bag-of-words for each synthetic "image".
        let mut rng = seeded(23);
        let mut bows = Vec::new();
        for _ in 0..8 {
            let mut idx: Vec<usize> = (0..pool.len()).collect();
            for a in 0..10 {
                let b = rng.random_range(a..pool.len());
                idx.swap(a, b);
            }
            let descs: Vec<Vec<u8>> = idx[..10].iter().map(|&j| pool[j].clone()).collect();
            bows.push(BowVector::from_descriptors(&vocab, &descs));
        }

        for (query_idx, query) in bows.iter().enumerate() {
            let best = bows
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    BowVector::score_l1(query, a)
                        .partial_cmp(&BowVector::score_l1(query, b))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(
                best, query_idx,
                "image {query_idx} should score itself highest"
            );
            // Identical vectors reach the maximum possible L1 similarity.
            assert!((BowVector::score_l1(query, query) - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn empty_input_is_sane() {
        let (vocab, _) = sample_vocabulary();
        let empty = BowVector::from_descriptors(&vocab, &[] as &[Vec<u8>]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.norm_l1(), 0.0);
        assert_eq!(empty.norm_l2(), 0.0);
        assert_eq!(BowVector::l1_distance(&empty, &empty), 0.0);
        assert_eq!(BowVector::score_l2(&empty, &empty), 1.0);

        let mut zero = empty.clone();
        zero.normalize_l1();
        assert!(zero.is_empty());

        // Empty vocabulary produces an empty vector without panicking.
        let no_vocab = Vocabulary::train::<Vec<u8>>(&[], 0, 0, 0);
        let bow = BowVector::from_descriptors(&no_vocab, &[vec![1u8, 2, 3]]);
        assert!(bow.is_empty());
    }
}
