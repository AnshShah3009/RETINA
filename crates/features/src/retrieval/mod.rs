//! Visual-localization retrieval layer.
//!
//! This module adds the database-search half of a visual-localization system on
//! top of the existing ORB/BRIEF descriptors and matching code. It provides:
//!
//! - [`Vocabulary`]: a binary-descriptor *vocabulary* (a set of visual words)
//!   trained with k-medians under the Hamming distance.
//! - [`BowVector`]: sparse, TF-IDF weighted bag-of-words vectors with L1/L2
//!   similarity scoring.
//! - [`BowDatabase`]: an inverted-file image database that answers top-`k`
//!   similarity queries.
//! - [`LshIndex`]: a multi-table locality-sensitive hash index for fast
//!   approximate nearest-neighbour search over binary descriptors.
//!
//! All randomised operations take an explicit `seed: u64` and use a local
//! [`rand::rngs::StdRng`], so results are fully reproducible. The public
//! routines accept any descriptor type that coerces to `&[u8]` (a `Vec<u8>`,
//! `&[u8]`, or `[u8; N]`), e.g. 32-byte ORB descriptors, and never panic on
//! empty input.

mod bow;
mod database;
mod lsh;
mod vocabulary;

pub use bow::BowVector;
pub use database::BowDatabase;
pub use lsh::LshIndex;
pub use vocabulary::Vocabulary;

use cv_core::Descriptors;

/// Compute the Hamming distance between two binary descriptors.
///
/// Bits are compared over the overlapping prefix; trailing bytes present in only
/// one of the slices are ignored.
pub(crate) fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Flatten a cv-core [`Descriptors`] collection into plain byte vectors.
///
/// Convenience bridge for feeding descriptors extracted by the ORB/BRIEF
/// extractors into the retrieval layer.
pub fn descriptors_to_bytes(descriptors: &Descriptors) -> Vec<Vec<u8>> {
    descriptors.iter().map(|d| d.data.clone()).collect()
}

/// Shared deterministic helpers used by the module's tests.
#[cfg(test)]
pub(crate) mod test_support {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Create a deterministically seeded RNG.
    pub(crate) fn seeded(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Generate a uniformly random `dim`-byte descriptor.
    pub(crate) fn random_descriptor(rng: &mut StdRng, dim: usize) -> Vec<u8> {
        (0..dim).map(|_| rng.random::<u8>()).collect()
    }

    /// Flip `n` random bits of `desc`, returning a perturbed copy.
    pub(crate) fn flip_bits(rng: &mut StdRng, desc: &[u8], n: usize) -> Vec<u8> {
        let mut out = desc.to_vec();
        let total_bits = out.len() * 8;
        if total_bits == 0 {
            return out;
        }
        for _ in 0..n {
            let bit = rng.random_range(0..total_bits);
            out[bit / 8] ^= 1 << (bit % 8);
        }
        out
    }
}
