//! Locality-sensitive hashing over binary descriptors.
//!
//! Each of the `tables` hash tables samples `bits_per_key` bit positions of the
//! descriptor to form a compact bucket key. Descriptors that agree on those bits
//! collide into the same bucket, giving fast approximate nearest-neighbour
//! candidates under the Hamming distance.

use crate::retrieval::hamming_distance;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

/// A multi-table LSH index for fixed-length binary descriptors.
///
/// Keys are built from sampled bit positions (up to 64 per table), so the index
/// is deterministic for a given `seed` and returns candidates in a stable order.
pub struct LshIndex {
    dim: usize,
    tables: usize,
    bits_per_key: usize,
    positions: Vec<Vec<usize>>,
    buckets: Vec<HashMap<u64, Vec<usize>>>,
    descriptors: HashMap<usize, Vec<u8>>,
}

impl LshIndex {
    /// Create an index for `dim`-byte descriptors.
    ///
    /// `tables` hash tables are created, each hashing on `bits_per_key`
    /// distinct bit positions (`bits_per_key` is capped at 64 and at `dim * 8`).
    /// `seed` deterministically fixes the sampled positions for every table.
    pub fn new(dim: usize, tables: usize, bits_per_key: usize, seed: u64) -> Self {
        let total_bits = dim * 8;
        let bits = bits_per_key.min(64).min(total_bits);
        let mut rng = StdRng::seed_from_u64(seed);

        let positions = (0..tables)
            .map(|_| sample_positions(total_bits, bits, &mut rng))
            .collect();

        Self {
            dim,
            tables,
            bits_per_key: bits,
            positions,
            buckets: (0..tables).map(|_| HashMap::new()).collect(),
            descriptors: HashMap::new(),
        }
    }

    /// Descriptor length in bytes this index was configured for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of hash tables.
    pub fn tables(&self) -> usize {
        self.tables
    }

    /// Number of sampled bits per table key (after capping).
    pub fn bits_per_key(&self) -> usize {
        self.bits_per_key
    }

    /// Number of distinct ids inserted.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether the index holds no ids.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Insert `descriptor` under `id`, adding it to every table's bucket.
    ///
    /// Re-inserting the same `id` overwrites its stored descriptor; query results
    /// are always deduplicated by id.
    pub fn insert<D: AsRef<[u8]>>(&mut self, id: usize, descriptor: D) {
        let desc = descriptor.as_ref().to_vec();
        for (table, positions) in self.positions.iter().enumerate() {
            let key = bucket_key(&desc, positions);
            self.buckets[table].entry(key).or_default().push(id);
        }
        self.descriptors.insert(id, desc);
    }

    /// Return up to `k` `(id, hamming_distance)` pairs, best first.
    ///
    /// Candidates are gathered from every table's bucket and deduplicated so the
    /// same id is never returned twice. Ordering is by ascending Hamming distance
    /// with ties broken by ascending id. Returns an empty vector for `k == 0`, an
    /// index with no tables, or an empty index.
    pub fn query<D: AsRef<[u8]>>(&self, descriptor: D, k: usize) -> Vec<(usize, u32)> {
        if k == 0 || self.tables == 0 || self.descriptors.is_empty() {
            return Vec::new();
        }

        let desc = descriptor.as_ref();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for (table, positions) in self.positions.iter().enumerate() {
            let key = bucket_key(desc, positions);
            if let Some(ids) = self.buckets[table].get(&key) {
                for &id in ids {
                    if seen.insert(id) {
                        candidates.push(id);
                    }
                }
            }
        }

        let mut results: Vec<(usize, u32)> = candidates
            .into_iter()
            .filter_map(|id| {
                self.descriptors
                    .get(&id)
                    .map(|stored| (id, hamming_distance(desc, stored)))
            })
            .collect();
        results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        results
    }

    /// Number of distinct candidate ids sharing a bucket with `descriptor`.
    ///
    /// Useful for inspecting bucket occupancy; does not compute distances.
    pub fn candidate_count<D: AsRef<[u8]>>(&self, descriptor: D) -> usize {
        let desc = descriptor.as_ref();
        let mut seen = HashSet::new();
        for (table, positions) in self.positions.iter().enumerate() {
            let key = bucket_key(desc, positions);
            if let Some(ids) = self.buckets[table].get(&key) {
                seen.extend(ids.iter().copied());
            }
        }
        seen.len()
    }
}

/// Sample `m` distinct bit positions from `0..total_bits` using a partial
/// Fisher-Yates shuffle so the selection is deterministic for a given RNG state.
fn sample_positions(total_bits: usize, m: usize, rng: &mut StdRng) -> Vec<usize> {
    if m == 0 || total_bits == 0 {
        return Vec::new();
    }
    let m = m.min(total_bits);
    let mut all: Vec<usize> = (0..total_bits).collect();
    for i in 0..m {
        let j = rng.random_range(i..total_bits);
        all.swap(i, j);
    }
    all.truncate(m);
    all
}

/// Build a `u64` bucket key from the descriptor's bits at `positions`.
///
/// Missing bytes read as zero; at most 64 positions are encoded.
fn bucket_key(desc: &[u8], positions: &[usize]) -> u64 {
    let mut key = 0u64;
    for (slot, &pos) in positions.iter().enumerate() {
        if slot >= 64 {
            break;
        }
        let byte = pos / 8;
        if byte < desc.len() && (desc[byte] >> (pos % 8)) & 1 == 1 {
            key |= 1u64 << slot;
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::test_support::{flip_bits, random_descriptor, seeded};

    const DIM: usize = 32;
    const COUNT: usize = 60;

    fn populated_index(seed: u64, tables: usize, bits: usize) -> (LshIndex, Vec<Vec<u8>>) {
        let mut rng = seeded(seed);
        let descs: Vec<Vec<u8>> = (0..COUNT)
            .map(|_| random_descriptor(&mut rng, DIM))
            .collect();
        let mut index = LshIndex::new(DIM, tables, bits, 7);
        for (id, d) in descs.iter().enumerate() {
            index.insert(id, d);
        }
        (index, descs)
    }

    #[test]
    fn exact_matches_recall_at_1_is_full() {
        let (index, descs) = populated_index(2024, 16, 16);
        assert_eq!(index.len(), COUNT);

        let mut hits = 0;
        for (id, d) in descs.iter().enumerate() {
            let top = index.query(d, 1);
            if top.first().map(|&(top_id, _)| top_id) == Some(id) {
                hits += 1;
            }
        }
        assert_eq!(hits, COUNT, "exact-match recall@1 must be 100%");
    }

    #[test]
    fn one_bit_flip_returns_original_in_top_three() {
        let (index, descs) = populated_index(2024, 16, 8);
        let mut rng = seeded(555);
        let sample = 40usize;

        let mut hits = 0;
        for id in 0..sample {
            let query = flip_bits(&mut rng, &descs[id], 1);
            if index.query(&query, 3).iter().any(|&(rid, _)| rid == id) {
                hits += 1;
            }
        }
        let recall = hits as f64 / sample as f64;
        assert!(
            recall >= 0.9,
            "1-bit-flip recall@3 = {recall} (want >= 0.9)"
        );
    }

    #[test]
    fn query_results_are_unique() {
        let mut index = LshIndex::new(4, 8, 8, 3);
        for id in 0..12 {
            index.insert(id, vec![id as u8; 4]);
        }
        let results = index.query(vec![0u8; 4], 12);

        let mut ids: Vec<usize> = results.iter().map(|&(id, _)| id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "ids must not repeat across tables");
    }

    #[test]
    fn empty_index_query_is_sane() {
        let index = LshIndex::new(DIM, 4, 8, 1);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.query(vec![0u8; DIM], 5).is_empty());
        assert_eq!(index.candidate_count(vec![0u8; DIM]), 0);

        // Degenerate configurations must not panic.
        let zero_tables = LshIndex::new(DIM, 0, 8, 1);
        assert!(zero_tables.query(vec![0u8; DIM], 5).is_empty());
        let zero_dim = LshIndex::new(0, 2, 8, 1);
        assert_eq!(zero_dim.bits_per_key(), 0);
        assert!(zero_dim.query(&[] as &[u8], 5).is_empty());
    }
}
