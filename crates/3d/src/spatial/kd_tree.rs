use nalgebra::Point3;
use std::collections::BinaryHeap;

/// KDTree for nearest neighbor queries.
///
/// For best performance, use [`KDTree::build`] for bulk construction (O(n log n),
/// produces a balanced tree) rather than repeated [`KDTree::insert`] calls.
pub struct KDTree<T: Clone> {
    root: Option<Box<KDNode<T>>>,
    dim: usize,
}

struct KDNode<T: Clone> {
    point: Point3<f32>,
    data: T,
    left: Option<Box<KDNode<T>>>,
    right: Option<Box<KDNode<T>>>,
    axis: usize,
}

/// Max-heap entry for kNN: ordered by distance so the farthest neighbor is at the top.
struct KnnEntry<T: Clone> {
    dist_sq: f32,
    point: Point3<f32>,
    data: T,
}

impl<T: Clone> PartialEq for KnnEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}
impl<T: Clone> Eq for KnnEntry<T> {}
impl<T: Clone> PartialOrd for KnnEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Clone> Ord for KnnEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn axis_coord(p: &Point3<f32>, axis: usize) -> f32 {
    match axis {
        0 => p.x,
        1 => p.y,
        _ => p.z,
    }
}

fn squared_distance(a: &Point3<f32>, b: &Point3<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

impl<T: Clone> KDTree<T> {
    pub fn new() -> Self {
        Self { root: None, dim: 3 }
    }

    pub fn with_capacity(_capacity: usize) -> Self {
        Self::new()
    }

    /// Bulk-build a balanced KDTree from a slice of (point, data) pairs.
    /// O(n log n) construction, produces an optimally balanced tree.
    pub fn build(items: &mut [(Point3<f32>, T)]) -> Self {
        let root = Self::build_recursive(items, 0, 3);
        Self { root, dim: 3 }
    }

    fn build_recursive(
        items: &mut [(Point3<f32>, T)],
        depth: usize,
        dim: usize,
    ) -> Option<Box<KDNode<T>>> {
        if items.is_empty() {
            return None;
        }
        let axis = depth % dim;
        // Partition around median using select_nth_unstable_by
        let mid = items.len() / 2;
        items.select_nth_unstable_by(mid, |a, b| {
            axis_coord(&a.0, axis)
                .partial_cmp(&axis_coord(&b.0, axis))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (left_slice, rest) = items.split_at_mut(mid);
        let (median, right_slice) = rest.split_first_mut().unwrap();
        Some(Box::new(KDNode {
            point: median.0,
            data: median.1.clone(),
            left: Self::build_recursive(left_slice, depth + 1, dim),
            right: Self::build_recursive(right_slice, depth + 1, dim),
            axis,
        }))
    }

    /// Insert a single point. For bulk data, prefer [`KDTree::build`].
    pub fn insert(&mut self, point: Point3<f32>, data: T) {
        self.root = Self::insert_recursive(self.root.take(), point, data, 0, self.dim);
    }

    fn insert_recursive(
        node: Option<Box<KDNode<T>>>,
        point: Point3<f32>,
        data: T,
        depth: usize,
        dim: usize,
    ) -> Option<Box<KDNode<T>>> {
        match node {
            None => Some(Box::new(KDNode {
                point,
                data,
                left: None,
                right: None,
                axis: depth % dim,
            })),
            Some(mut n) => {
                if axis_coord(&point, n.axis) < axis_coord(&n.point, n.axis) {
                    n.left = Self::insert_recursive(n.left, point, data, depth + 1, dim);
                } else {
                    n.right = Self::insert_recursive(n.right, point, data, depth + 1, dim);
                }
                Some(n)
            }
        }
    }

    pub fn nearest_neighbor(&self, query: &Point3<f32>) -> Option<(Point3<f32>, T, f32)> {
        self.root.as_ref().map(|root| {
            let mut best = (
                root.point,
                root.data.clone(),
                squared_distance(&root.point, query),
            );
            Self::nearest_recursive(root, query, &mut best);
            best
        })
    }

    fn nearest_recursive(node: &KDNode<T>, query: &Point3<f32>, best: &mut (Point3<f32>, T, f32)) {
        let dist = squared_distance(&node.point, query);
        if dist < best.2 {
            *best = (node.point, node.data.clone(), dist);
        }

        let diff = axis_coord(query, node.axis) - axis_coord(&node.point, node.axis);
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::nearest_recursive(child, query, best);
        }
        if diff * diff < best.2 {
            if let Some(ref child) = second {
                Self::nearest_recursive(child, query, best);
            }
        }
    }

    pub fn search_radius(&self, query: &Point3<f32>, radius: f32) -> Vec<(Point3<f32>, T, f32)> {
        let mut results = Vec::new();
        let radius_sq = radius * radius;
        if let Some(ref root) = self.root {
            Self::radius_recursive(root, query, radius_sq, &mut results);
        }
        results
    }

    fn radius_recursive(
        node: &KDNode<T>,
        query: &Point3<f32>,
        radius_sq: f32,
        results: &mut Vec<(Point3<f32>, T, f32)>,
    ) {
        let dist = squared_distance(&node.point, query);
        if dist <= radius_sq {
            results.push((node.point, node.data.clone(), dist));
        }

        let diff = axis_coord(query, node.axis) - axis_coord(&node.point, node.axis);
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::radius_recursive(child, query, radius_sq, results);
        }
        if diff * diff < radius_sq {
            if let Some(ref child) = second {
                Self::radius_recursive(child, query, radius_sq, results);
            }
        }
    }

    /// K nearest neighbors using a max-heap for efficient pruning.
    /// Only explores branches that could contain closer points than the current k-th best.
    pub fn k_nearest_neighbors(&self, query: &Point3<f32>, k: usize) -> Vec<(Point3<f32>, T, f32)> {
        let mut heap: BinaryHeap<KnnEntry<T>> = BinaryHeap::with_capacity(k + 1);
        if let Some(ref root) = self.root {
            Self::knn_recursive(root, query, k, &mut heap);
        }
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|e| (e.point, e.data, e.dist_sq))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn knn_recursive(
        node: &KDNode<T>,
        query: &Point3<f32>,
        k: usize,
        heap: &mut BinaryHeap<KnnEntry<T>>,
    ) {
        let dist = squared_distance(&node.point, query);

        if heap.len() < k {
            heap.push(KnnEntry {
                dist_sq: dist,
                point: node.point,
                data: node.data.clone(),
            });
        } else if dist < heap.peek().unwrap().dist_sq {
            heap.pop();
            heap.push(KnnEntry {
                dist_sq: dist,
                point: node.point,
                data: node.data.clone(),
            });
        }

        let diff = axis_coord(query, node.axis) - axis_coord(&node.point, node.axis);
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::knn_recursive(child, query, k, heap);
        }

        // Prune: only explore the other side if it could contain closer points
        let worst = if heap.len() < k {
            f32::MAX
        } else {
            heap.peek().unwrap().dist_sq
        };
        if diff * diff < worst {
            if let Some(ref child) = second {
                Self::knn_recursive(child, query, k, heap);
            }
        }
    }
}

impl<T: Clone> Default for KDTree<T> {
    fn default() -> Self {
        Self::new()
    }
}
