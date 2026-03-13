//! Line Matching functionality

use crate::hough::LineSegment;

/// Binary descriptor for a detected line segment.
#[derive(Debug, Clone)]
pub struct LineDescriptor {
    /// Raw descriptor bytes (256-bit / 32-byte binary pattern).
    pub data: Vec<u8>,
    /// The line segment this descriptor was computed for.
    pub segment: LineSegment,
}

/// A match between two line descriptors.
#[derive(Debug, Clone, Copy)]
pub struct LineMatch {
    /// Index into the query descriptor list.
    pub query_idx: usize,
    /// Index into the training descriptor list.
    pub train_idx: usize,
    /// Hamming distance between the matched descriptors.
    pub distance: f32,
}

/// Nearest-neighbour line descriptor matcher.
pub struct LineMatcher {
    /// Maximum Hamming distance for a match to be accepted.
    pub threshold: f32,
}

impl LineMatcher {
    /// Create a new matcher with the given Hamming-distance threshold.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Match each query descriptor to the nearest training descriptor.
    ///
    /// Returns only matches whose Hamming distance is below the configured threshold.
    pub fn match_lines(
        &self,
        query: &[LineDescriptor],
        train: &[LineDescriptor],
    ) -> Vec<LineMatch> {
        let mut matches = Vec::new();

        for (qi, q) in query.iter().enumerate() {
            let mut best_dist = u32::MAX;
            let mut best_idx = None;

            for (ti, t) in train.iter().enumerate() {
                let dist = hamming_dist(&q.data, &t.data);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(ti);
                }
            }

            if let Some(ti) = best_idx {
                if (best_dist as f32) < self.threshold {
                    matches.push(LineMatch {
                        query_idx: qi,
                        train_idx: ti,
                        distance: best_dist as f32,
                    });
                }
            }
        }

        matches
    }
}

fn hamming_dist(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0;
    for i in 0..a.len().min(b.len()) {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}
