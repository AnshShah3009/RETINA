//! Binary-descriptor vocabulary (visual words) trained with k-medians in
//! Hamming space.

use crate::retrieval::hamming_distance;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A trained vocabulary of binary visual words.
///
/// Each word is a `dim`-byte centroid produced by k-medians clustering of the
/// training descriptors under the Hamming distance. Centroids are re-estimated
/// by per-bit majority vote (`> 50%` of the assigned descriptors have the bit
/// set); a cluster that loses all of its members keeps its previous centroid.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    centroids: Vec<Vec<u8>>,
    dim: usize,
    df: Vec<u32>,
    n_train: u32,
}

impl Vocabulary {
    /// Train a vocabulary of `k` words over `iterations` k-medians iterations.
    ///
    /// `descriptors` is any slice of values coercible to `&[u8]` (all are
    /// assumed to share the same length, e.g. 32-byte ORB descriptors). `seed`
    /// deterministically fixes k-means++ initialization.
    ///
    /// Returns an empty vocabulary (zero words, `dim == 0`) when `descriptors`
    /// is empty or `k == 0`.
    pub fn train<D: AsRef<[u8]>>(
        descriptors: &[D],
        k: usize,
        iterations: usize,
        seed: u64,
    ) -> Self {
        let descs: Vec<&[u8]> = descriptors.iter().map(|d| d.as_ref()).collect();
        let n = descs.len();
        if n == 0 || k == 0 {
            return Self {
                centroids: Vec::new(),
                dim: 0,
                df: Vec::new(),
                n_train: 0,
            };
        }
        let dim = descs[0].len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = kmeans_plus_plus_init(&descs, k, &mut rng);
        let dim_bits = dim * 8;

        for _ in 0..iterations {
            let mut bit_sums = vec![vec![0u32; dim_bits]; centroids.len()];
            let mut counts = vec![0u32; centroids.len()];

            for d in &descs {
                let (w, _) = nearest(&centroids, d);
                let w = w as usize;
                counts[w] += 1;
                for (bit, sum) in bit_sums[w].iter_mut().enumerate() {
                    let byte = bit / 8;
                    if byte < d.len() && (d[byte] >> (bit % 8)) & 1 == 1 {
                        *sum += 1;
                    }
                }
            }

            for (c, (&count, sums)) in counts.iter().zip(bit_sums.iter()).enumerate() {
                if count == 0 {
                    continue; // Empty cluster: keep the previous centroid.
                }
                let mut new_c = vec![0u8; dim];
                for (bit, &sum) in sums.iter().enumerate() {
                    if (sum as u64) * 2 > count as u64 {
                        new_c[bit / 8] |= 1 << (bit % 8);
                    }
                }
                centroids[c] = new_c;
            }
        }

        // Document frequency: how many training descriptors fell into each word.
        let mut df = vec![0u32; centroids.len()];
        for d in &descs {
            let (w, _) = nearest(&centroids, d);
            df[w as usize] += 1;
        }

        Self {
            centroids,
            dim,
            df,
            n_train: n as u32,
        }
    }

    /// Number of words in the vocabulary.
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Descriptor length in bytes (`0` for an empty vocabulary).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Find the closest word to `descriptor`.
    ///
    /// Returns `(word_id, hamming_distance)`; ties resolve to the lowest word
    /// id. An empty vocabulary yields `(0, 0)`.
    pub fn nearest_word(&self, descriptor: &[u8]) -> (u32, u32) {
        nearest(&self.centroids, descriptor)
    }

    /// Inverse document frequency of `word`, derived from training statistics.
    ///
    /// `idf = ln((n_train + 1) / (df + 1)) + 1`, where `df` is the number of
    /// training descriptors assigned to the word. The value is always `>= 1`;
    /// out-of-range words and empty vocabularies return `1.0`.
    pub fn idf(&self, word: u32) -> f32 {
        let idx = word as usize;
        if self.n_train == 0 || idx >= self.df.len() {
            return 1.0;
        }
        let n = self.n_train as f32;
        let df = self.df[idx] as f32;
        ((n + 1.0) / (df + 1.0)).ln() + 1.0
    }

    /// Centroid bytes for `word`, or `None` if the word does not exist.
    pub fn word(&self, word: u32) -> Option<&[u8]> {
        self.centroids.get(word as usize).map(|c| c.as_slice())
    }
}

/// Return the nearest centroid index and Hamming distance, resolving ties to the
/// lowest index. An empty centroid set yields `(0, 0)`.
fn nearest(centroids: &[Vec<u8>], descriptor: &[u8]) -> (u32, u32) {
    if centroids.is_empty() {
        return (0, 0);
    }
    let mut best = 0usize;
    let mut best_dist = u32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = hamming_distance(c, descriptor);
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    (best as u32, best_dist)
}

/// Deterministic k-means++ seeding: spread initial centroids across the data
/// proportional to their squared distance from the nearest chosen centroid.
fn kmeans_plus_plus_init(descs: &[&[u8]], k: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let n = descs.len();
    let mut centroids = Vec::with_capacity(k);

    let first = rng.random_range(0..n);
    centroids.push(descs[first].to_vec());
    let mut dist2: Vec<f64> = descs
        .iter()
        .map(|d| {
            let h = hamming_distance(d, &centroids[0]) as f64;
            h * h
        })
        .collect();

    while centroids.len() < k {
        let total: f64 = dist2.iter().sum();
        let pick = if total <= 0.0 {
            // All remaining points coincide with existing centroids.
            rng.random_range(0..n)
        } else {
            let r = rng.random::<f64>() * total;
            let mut acc = 0.0;
            let mut idx = n - 1;
            for (i, &w) in dist2.iter().enumerate() {
                acc += w;
                if acc >= r {
                    idx = i;
                    break;
                }
            }
            idx
        };

        let new_c = descs[pick].to_vec();
        for (i, d) in descs.iter().enumerate() {
            let h = hamming_distance(d, &new_c) as f64;
            let nd = h * h;
            if nd < dist2[i] {
                dist2[i] = nd;
            }
        }
        centroids.push(new_c);
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::test_support::{flip_bits, seeded};

    /// Build three well-separated synthetic clusters around fixed bit patterns.
    fn make_clusters(dim: usize, per_cluster: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let centers = [vec![0x00u8; dim], vec![0xFFu8; dim], vec![0xAAu8; dim]];
        let mut rng = seeded(seed);
        let mut descs = Vec::new();
        let mut labels = Vec::new();
        for (label, center) in centers.iter().enumerate() {
            for _ in 0..per_cluster {
                descs.push(flip_bits(&mut rng, center, 2));
                labels.push(label);
            }
        }
        (descs, labels)
    }

    #[test]
    fn train_separates_well_separated_clusters() {
        let (descs, labels) = make_clusters(16, 10, 42);
        let vocab = Vocabulary::train(&descs, 3, 20, 7);

        assert_eq!(vocab.k(), 3);
        assert_eq!(vocab.dim(), 16);

        let mut words_per_cluster: Vec<Vec<u32>> = vec![Vec::new(); 3];
        for (d, &label) in descs.iter().zip(labels.iter()) {
            let (w, _) = vocab.nearest_word(d);
            words_per_cluster[label].push(w);
        }

        for words in &words_per_cluster {
            assert!(
                words.iter().all(|&w| w == words[0]),
                "cluster split across words: {words:?}"
            );
        }
        let (w0, w1, w2) = (
            words_per_cluster[0][0],
            words_per_cluster[1][0],
            words_per_cluster[2][0],
        );
        assert_ne!(w0, w1);
        assert_ne!(w1, w2);
        assert_ne!(w0, w2);
    }

    #[test]
    fn nearest_word_returns_centroid_for_exact_member() {
        let (descs, _) = make_clusters(16, 10, 42);
        let vocab = Vocabulary::train(&descs, 3, 20, 7);

        // The first cluster's center is a majority of its (lightly perturbed)
        // members, so the trained centroid must equal it exactly.
        let center0 = vec![0x00u8; 16];
        let (word, dist) = vocab.nearest_word(&center0);
        assert_eq!(dist, 0, "centroid should coincide with the cluster center");
        assert_eq!(vocab.word(word), Some(center0.as_slice()));
    }

    #[test]
    fn empty_training_is_sane() {
        let vocab = Vocabulary::train::<Vec<u8>>(&[], 5, 5, 1);
        assert_eq!(vocab.k(), 0);
        assert_eq!(vocab.dim(), 0);
        assert_eq!(vocab.nearest_word(&[]), (0, 0));
        assert_eq!(vocab.idf(0), 1.0);
        assert_eq!(vocab.word(0), None);

        let zero_k = Vocabulary::train::<Vec<u8>>(&[vec![0u8; 4]], 0, 5, 1);
        assert_eq!(zero_k.k(), 0);
    }
}
