//! Inverted-file image database over bag-of-words vectors.

use crate::retrieval::{BowVector, Vocabulary};
use std::cmp::Ordering;
use std::collections::HashMap;

/// An image database backed by an inverted file over bag-of-words vectors.
///
/// Images are added as descriptor sets and converted to TF-IDF bag-of-words
/// vectors using the database vocabulary. [`BowDatabase::build`] constructs the
/// inverted lists; [`BowDatabase::query`] then accumulates an inner-product
/// score over only the images that share a word with the query.
pub struct BowDatabase {
    vocabulary: Vocabulary,
    entries: Vec<(usize, BowVector)>,
    inverted: Vec<Vec<(usize, f32)>>,
    built: bool,
}

impl BowDatabase {
    /// Create an empty database that uses `vocabulary` to build word vectors.
    pub fn new(vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            entries: Vec::new(),
            inverted: Vec::new(),
            built: false,
        }
    }

    /// Add an image's descriptors under `image_id`.
    ///
    /// The descriptors are converted to a bag-of-words vector immediately. Any
    /// previously built index is invalidated until [`BowDatabase::build`] is
    /// called again.
    pub fn add<D: AsRef<[u8]>>(&mut self, image_id: usize, descriptors: &[D]) {
        let bow = BowVector::from_descriptors(&self.vocabulary, descriptors);
        self.entries.push((image_id, bow));
        self.built = false;
    }

    /// Build the inverted lists from every image added so far.
    ///
    /// Must be called (again after new `add`s) before [`BowDatabase::query`]
    /// returns results.
    pub fn build(&mut self) {
        let mut inverted = vec![Vec::new(); self.vocabulary.k()];
        for (image_id, bow) in &self.entries {
            for &(word, weight) in &bow.entries {
                if let Some(list) = inverted.get_mut(word as usize) {
                    list.push((*image_id, weight));
                }
            }
        }
        self.inverted = inverted;
        self.built = true;
    }

    /// Number of images in the database (built or not).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database holds no images.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the inverted index has been built since the last modification.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// The vocabulary backing this database.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Query the database with `query`, returning up to `k` best matches.
    ///
    /// Results are `(image_id, score)` pairs sorted by descending inner-product
    /// score, with ties broken deterministically by ascending image id. Scores
    /// accumulate only over images that share at least one word with the query,
    /// so images with no overlap are never returned. Returns an empty vector when
    /// the index has not been built, `k == 0`, or the query is empty.
    pub fn query(&self, query: &BowVector, k: usize) -> Vec<(usize, f32)> {
        if !self.built || k == 0 || query.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for &(word, query_weight) in &query.entries {
            if let Some(list) = self.inverted.get(word as usize) {
                for &(image_id, weight) in list {
                    *scores.entry(image_id).or_insert(0.0) += query_weight * weight;
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::test_support::{flip_bits, random_descriptor, seeded};
    use rand::Rng;

    const DIM: usize = 32;
    const POOL_SIZE: usize = 50;
    const IMAGES: usize = 20;
    const WORDS_PER_IMAGE: usize = 15;

    /// Build a database of synthetic images drawn from a shared descriptor pool,
    /// returning the database and each image's descriptors.
    fn build_database(seed: u64) -> (BowDatabase, Vec<Vec<Vec<u8>>>) {
        let mut rng = seeded(seed);
        let pool: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| random_descriptor(&mut rng, DIM))
            .collect();

        // One word per pool descriptor keeps the shared structure explicit.
        let vocab = Vocabulary::train(&pool, POOL_SIZE, 10, 99);
        let mut db = BowDatabase::new(vocab);

        let mut images = Vec::new();
        for image_id in 0..IMAGES {
            let mut idx: Vec<usize> = (0..POOL_SIZE).collect();
            for a in 0..WORDS_PER_IMAGE {
                let b = rng.random_range(a..POOL_SIZE);
                idx.swap(a, b);
            }
            let descs: Vec<Vec<u8>> = idx[..WORDS_PER_IMAGE]
                .iter()
                .map(|&j| pool[j].clone())
                .collect();
            db.add(image_id, &descs);
            images.push(descs);
        }
        db.build();
        (db, images)
    }

    #[test]
    fn exact_copy_retrieves_image_first() {
        let (db, images) = build_database(7);
        assert_eq!(db.len(), IMAGES);
        assert!(db.is_built());

        for &target in &[0usize, 5, 13, IMAGES - 1] {
            let query = BowVector::from_descriptors(db.vocabulary(), &images[target]);
            let results = db.query(&query, 3);
            assert_eq!(
                results.first().map(|&(id, _)| id),
                Some(target),
                "exact query for image {target} must rank it first: {results:?}"
            );
        }
    }

    #[test]
    fn perturbed_copy_retrieves_image_in_top_three() {
        let (db, images) = build_database(7);
        let target = 9usize;

        let mut rng = seeded(321);
        let mut perturbed = images[target].clone();
        for (i, desc) in perturbed.iter_mut().enumerate() {
            if i % 5 == 0 {
                // Simulate mismatches/outliers by replacing a few descriptors.
                *desc = random_descriptor(&mut rng, DIM);
            } else {
                *desc = flip_bits(&mut rng, desc, 3);
            }
        }

        let query = BowVector::from_descriptors(db.vocabulary(), &perturbed);
        let results = db.query(&query, 3);
        assert!(
            results.iter().any(|&(id, _)| id == target),
            "perturbed query for image {target} must appear in top-3: {results:?}"
        );
    }

    #[test]
    fn ties_break_by_image_id() {
        let (db, images) = build_database(7);
        // Rebuild a small database where three images share identical content.
        let mut tied = BowDatabase::new(db.vocabulary().clone());
        let descs = images[2].clone();
        for id in [7usize, 2, 5] {
            tied.add(id, &descs);
        }
        tied.build();

        let query = BowVector::from_descriptors(tied.vocabulary(), &descs);
        let ids: Vec<usize> = tied
            .query(&query, 3)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, vec![2, 5, 7]);
    }

    #[test]
    fn empty_database_and_empty_query_are_sane() {
        let mut db = BowDatabase::new(Vocabulary::train::<Vec<u8>>(&[], 0, 0, 1));
        assert!(db.is_empty());
        assert!(!db.is_built());

        let empty_query = BowVector::default();
        assert!(db.query(&empty_query, 5).is_empty());

        // Building an empty database is safe and queries still return nothing.
        db.build();
        assert!(db.query(&empty_query, 5).is_empty());

        // Adding an image with no descriptors keeps it queryable but empty.
        db.add(0, &[] as &[Vec<u8>]);
        db.build();
        assert_eq!(db.len(), 1);
        assert!(db.query(&empty_query, 5).is_empty());
    }
}
