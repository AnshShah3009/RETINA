//! Spatial data structures and distance computations.
//!
//! Provides a KD-tree for efficient nearest-neighbor queries and pairwise
//! distance functions (`cdist`, `pdist`) with multiple distance metrics.
//!
//! # Example
//!
//! ```rust
//! use cv_scientific::spatial::{KDTree, cdist, DistanceMetric};
//!
//! let points = vec![
//!     vec![0.0, 0.0],
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//! ];
//! let tree = KDTree::new(&points).unwrap();
//!
//! // 2 nearest neighbors of (0.1, 0.1)
//! let neighbors = tree.query(&[0.1, 0.1], 2);
//! assert_eq!(neighbors[0].0, 0); // closest is origin
//! ```

use nalgebra::DMatrix;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Distance metric for spatial queries.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance.
    Euclidean,
    /// L1 (Manhattan / city-block) distance.
    Manhattan,
    /// L-infinity (Chebyshev) distance.
    Chebyshev,
    /// Lp (Minkowski) distance with exponent p.
    Minkowski(f64),
}

/// Compute the distance between two points using the given metric.
fn distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt(),
        DistanceMetric::Manhattan => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>(),
        DistanceMetric::Chebyshev => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max),
        DistanceMetric::Minkowski(p) => {
            let s: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs().powf(p))
                .sum();
            s.powf(1.0 / p)
        }
    }
}

/// Compute pairwise distances between two sets of points.
///
/// Returns an `m x n` matrix where `D[i,j] = distance(xa[i], xb[j])`.
pub fn cdist(xa: &[Vec<f64>], xb: &[Vec<f64>], metric: DistanceMetric) -> DMatrix<f64> {
    let m = xa.len();
    let n = xb.len();
    DMatrix::from_fn(m, n, |i, j| distance(&xa[i], &xb[j], metric))
}

/// Compute condensed pairwise distances within a single set of points.
///
/// Returns a vector of `n*(n-1)/2` distances corresponding to the upper
/// triangle of the full distance matrix, in row-major order:
/// `[(0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)]`.
pub fn pdist(x: &[Vec<f64>], metric: DistanceMetric) -> Vec<f64> {
    let n = x.len();
    let mut result = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            result.push(distance(&x[i], &x[j], metric));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// KD-Tree
// ---------------------------------------------------------------------------

/// Internal KD-tree node.
#[derive(Debug)]
enum KDNode {
    Leaf {
        indices: Vec<usize>,
        points: Vec<Vec<f64>>,
    },
    Internal {
        split_dim: usize,
        split_val: f64,
        left: usize,  // index into nodes vec
        right: usize, // index into nodes vec
    },
}

/// KD-tree for efficient nearest-neighbor and radius queries.
///
/// Supports arbitrary dimensionality. Built using the median-split strategy
/// with a leaf size of 16 for efficiency.
#[derive(Debug)]
pub struct KDTree {
    nodes: Vec<KDNode>,
    dimension: usize,
    points: Vec<Vec<f64>>,
}

const LEAF_SIZE: usize = 16;

/// Helper for the k-NN priority queue: (negative distance, index) so that
/// `BinaryHeap` (max-heap) gives us the farthest neighbor on top.
#[derive(Debug, Clone)]
struct HeapItem {
    dist: f64,
    index: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance (farthest on top)
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl KDTree {
    /// Build a KD-tree from a set of points.
    ///
    /// All points must have the same dimensionality. Returns an error if the
    /// input is empty or dimensions are inconsistent.
    pub fn new(points: &[Vec<f64>]) -> Result<Self, String> {
        if points.is_empty() {
            return Err("Cannot build KD-tree from empty point set".into());
        }
        let dimension = points[0].len();
        if dimension == 0 {
            return Err("Points must have at least 1 dimension".into());
        }
        for (i, p) in points.iter().enumerate() {
            if p.len() != dimension {
                return Err(format!(
                    "Point {} has dimension {} but expected {}",
                    i,
                    p.len(),
                    dimension
                ));
            }
        }

        let indices: Vec<usize> = (0..points.len()).collect();
        let mut tree = KDTree {
            nodes: Vec::new(),
            dimension,
            points: points.to_vec(),
        };

        tree.build_node(&indices, 0);
        Ok(tree)
    }

    /// Recursively build a subtree and return its node index.
    fn build_node(&mut self, indices: &[usize], depth: usize) -> usize {
        let node_idx = self.nodes.len();

        if indices.len() <= LEAF_SIZE {
            self.nodes.push(KDNode::Leaf {
                indices: indices.to_vec(),
                points: indices.iter().map(|&i| self.points[i].clone()).collect(),
            });
            return node_idx;
        }

        let split_dim = depth % self.dimension;

        // Find median along split_dim
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            self.points[a][split_dim]
                .partial_cmp(&self.points[b][split_dim])
                .unwrap_or(Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let split_val = self.points[sorted_indices[mid]][split_dim];

        let left_indices = &sorted_indices[..mid];
        let right_indices = &sorted_indices[mid..];

        // Reserve slot for this node
        self.nodes.push(KDNode::Internal {
            split_dim,
            split_val,
            left: 0,
            right: 0,
        });

        let left_idx = self.build_node(left_indices, depth + 1);
        let right_idx = self.build_node(right_indices, depth + 1);

        if let KDNode::Internal {
            ref mut left,
            ref mut right,
            ..
        } = self.nodes[node_idx]
        {
            *left = left_idx;
            *right = right_idx;
        }

        node_idx
    }

    /// Find the `k` nearest neighbors of `point`.
    ///
    /// Returns `(index, distance)` pairs sorted by ascending distance.
    pub fn query(&self, point: &[f64], k: usize) -> Vec<(usize, f64)> {
        assert_eq!(
            point.len(),
            self.dimension,
            "Query point dimension mismatch"
        );
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        self.knn_search(0, point, k, &mut heap);

        let mut result: Vec<(usize, f64)> = heap.into_iter().map(|h| (h.index, h.dist)).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }

    fn knn_search(
        &self,
        node_idx: usize,
        point: &[f64],
        k: usize,
        heap: &mut BinaryHeap<HeapItem>,
    ) {
        match &self.nodes[node_idx] {
            KDNode::Leaf { indices, points } => {
                for (i, pt) in indices.iter().zip(points.iter()) {
                    let d = euclidean_dist(point, pt);
                    if heap.len() < k {
                        heap.push(HeapItem { dist: d, index: *i });
                    } else if let Some(top) = heap.peek() {
                        if d < top.dist {
                            heap.pop();
                            heap.push(HeapItem { dist: d, index: *i });
                        }
                    }
                }
            }
            KDNode::Internal {
                split_dim,
                split_val,
                left,
                right,
            } => {
                let diff = point[*split_dim] - split_val;
                let (near, far) = if diff <= 0.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };

                self.knn_search(near, point, k, heap);

                // Check if we need to search the far side
                let should_search_far =
                    heap.len() < k || diff.abs() < heap.peek().map_or(f64::INFINITY, |h| h.dist);

                if should_search_far {
                    self.knn_search(far, point, k, heap);
                }
            }
        }
    }

    /// Find all point indices within Euclidean `radius` of `point`.
    pub fn query_ball(&self, point: &[f64], radius: f64) -> Vec<usize> {
        assert_eq!(
            point.len(),
            self.dimension,
            "Query point dimension mismatch"
        );
        let mut result = Vec::new();
        if !self.nodes.is_empty() {
            self.ball_search(0, point, radius, &mut result);
        }
        result
    }

    fn ball_search(&self, node_idx: usize, point: &[f64], radius: f64, result: &mut Vec<usize>) {
        match &self.nodes[node_idx] {
            KDNode::Leaf { indices, points } => {
                for (i, pt) in indices.iter().zip(points.iter()) {
                    if euclidean_dist(point, pt) <= radius {
                        result.push(*i);
                    }
                }
            }
            KDNode::Internal {
                split_dim,
                split_val,
                left,
                right,
            } => {
                let diff = point[*split_dim] - split_val;
                let (near, far) = if diff <= 0.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };

                self.ball_search(near, point, radius, result);

                if diff.abs() <= radius {
                    self.ball_search(far, point, radius, result);
                }
            }
        }
    }

    /// Find all pairs of points `(i, j)` with `i < j` within Euclidean
    /// distance `radius` of each other.
    ///
    /// Uses the KD-tree for efficient pruning; for each point queries the
    /// tree for neighbors within `radius`.
    pub fn query_pairs(&self, radius: f64) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.points.len() {
            let neighbors = self.query_ball(&self.points[i], radius);
            for j in neighbors {
                if j > i {
                    pairs.push((i, j));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }
}

/// Euclidean distance between two slices.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_kdtree_knn_basic() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let tree = KDTree::new(&points).unwrap();
        let neighbors = tree.query(&[0.0, 0.0], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!(approx_eq(neighbors[0].1, 0.0, 1e-12));
    }

    #[test]
    fn test_kdtree_knn_multiple() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
        ];
        let tree = KDTree::new(&points).unwrap();
        let neighbors = tree.query(&[0.1, 0.1], 3);
        assert_eq!(neighbors.len(), 3);
        // Closest should be (0,0), then (1,0) or (0,1)
        assert_eq!(neighbors[0].0, 0);
        // The 3 nearest should NOT include (10,10)
        let indices: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
        assert!(!indices.contains(&3));
    }

    #[test]
    fn test_kdtree_query_ball() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![2.0, 0.0],
            vec![5.0, 5.0],
        ];
        let tree = KDTree::new(&points).unwrap();
        let mut result = tree.query_ball(&[0.0, 0.0], 1.0);
        result.sort();
        // Should include point 0 (dist=0) and point 1 (dist=0.5)
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(!result.contains(&2)); // dist=2
        assert!(!result.contains(&3)); // dist=~7
    }

    #[test]
    fn test_kdtree_query_pairs() {
        let points = vec![vec![0.0], vec![0.5], vec![1.0], vec![10.0]];
        let tree = KDTree::new(&points).unwrap();
        let pairs = tree.query_pairs(0.6);
        // Within distance 0.6: (0,1) dist=0.5, (1,2) dist=0.5
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(1, 2)));
        assert!(!pairs.contains(&(0, 2))); // dist=1.0
        assert!(!pairs.contains(&(2, 3))); // dist=9.0
    }

    #[test]
    fn test_cdist_known() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = cdist(&xa, &xb, DistanceMetric::Euclidean);
        assert_eq!(d.shape(), (2, 2));
        assert!(approx_eq(d[(0, 0)], 1.0, 1e-12)); // (0,0)->(0,1) = 1
        assert!(approx_eq(d[(0, 1)], 2.0_f64.sqrt(), 1e-12)); // (0,0)->(1,1) = sqrt(2)
        assert!(approx_eq(d[(1, 0)], 2.0_f64.sqrt(), 1e-12)); // (1,0)->(0,1) = sqrt(2)
        assert!(approx_eq(d[(1, 1)], 1.0, 1e-12)); // (1,0)->(1,1) = 1
    }

    #[test]
    fn test_pdist() {
        let x = vec![vec![0.0], vec![1.0], vec![3.0]];
        let d = pdist(&x, DistanceMetric::Euclidean);
        // (0,1)=1, (0,2)=3, (1,2)=2
        assert_eq!(d.len(), 3);
        assert!(approx_eq(d[0], 1.0, 1e-12));
        assert!(approx_eq(d[1], 3.0, 1e-12));
        assert!(approx_eq(d[2], 2.0, 1e-12));
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        assert!(approx_eq(
            distance(&a, &b, DistanceMetric::Euclidean),
            5.0,
            1e-12
        ));
        assert!(approx_eq(
            distance(&a, &b, DistanceMetric::Manhattan),
            7.0,
            1e-12
        ));
        assert!(approx_eq(
            distance(&a, &b, DistanceMetric::Chebyshev),
            4.0,
            1e-12
        ));
        assert!(approx_eq(
            distance(&a, &b, DistanceMetric::Minkowski(2.0)),
            5.0,
            1e-12
        ));
    }

    #[test]
    fn test_kdtree_3d() {
        let points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![0.1, 0.1, 0.1],
        ];
        let tree = KDTree::new(&points).unwrap();
        let nn = tree.query(&[0.0, 0.0, 0.0], 2);
        assert_eq!(nn.len(), 2);
        assert_eq!(nn[0].0, 0); // exact match
        assert_eq!(nn[1].0, 3); // (0.1, 0.1, 0.1) is next closest
    }

    #[test]
    fn test_kdtree_empty_error() {
        let result = KDTree::new(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kdtree_large() {
        // Build tree with 1000 points and verify k-NN correctness via brute force
        let n = 1000;
        let _dim = 3;
        let points: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let f = i as f64;
                vec![f.sin(), f.cos(), f * 0.01]
            })
            .collect();
        let tree = KDTree::new(&points).unwrap();

        let query = vec![0.5, 0.5, 0.05];
        let k = 5;
        let kd_result = tree.query(&query, k);

        // Brute force
        let mut dists: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean_dist(&query, p)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for i in 0..k {
            assert_eq!(kd_result[i].0, dists[i].0, "Mismatch at neighbor {}", i);
            assert!(approx_eq(kd_result[i].1, dists[i].1, 1e-10));
        }
    }
}
