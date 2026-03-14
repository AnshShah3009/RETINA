//! Spatial Data Structures
//!
//! Implements:
//! - KDTree for fast nearest neighbor search
//! - Octree for spatial partitioning
//! - VoxelGrid for voxelization

use nalgebra::Point3;
use std::collections::{BinaryHeap, HashMap};

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

fn squared_distance(a: &Point3<f32>, b: &Point3<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

/// Octree for spatial partitioning
pub struct Octree<T: Clone> {
    root: Option<Box<OctreeNode<T>>>,
    bounds: (Point3<f32>, Point3<f32>),
    max_depth: usize,
    max_points_per_node: usize,
}

struct OctreeNode<T: Clone> {
    bounds: (Point3<f32>, Point3<f32>),
    center: Point3<f32>,
    children: Option<Box<[OctreeNode<T>; 8]>>,
    points: Vec<(Point3<f32>, T)>,
    depth: usize,
}

impl<T: Clone> Octree<T> {
    pub fn new(
        bounds: (Point3<f32>, Point3<f32>),
        max_depth: usize,
        max_points_per_node: usize,
    ) -> Self {
        Self {
            root: None,
            bounds,
            max_depth,
            max_points_per_node,
        }
    }

    pub fn insert(&mut self, point: Point3<f32>, data: T) {
        if self.root.is_none() {
            let center = Point3::new(
                (self.bounds.0.x + self.bounds.1.x) * 0.5,
                (self.bounds.0.y + self.bounds.1.y) * 0.5,
                (self.bounds.0.z + self.bounds.1.z) * 0.5,
            );
            self.root = Some(Box::new(OctreeNode::new(self.bounds, center, 0)));
        }

        if let Some(ref mut root) = self.root {
            Self::insert_recursive(root, point, data, self.max_depth, self.max_points_per_node);
        }
    }

    fn insert_recursive(
        node: &mut OctreeNode<T>,
        point: Point3<f32>,
        data: T,
        max_depth: usize,
        max_points: usize,
    ) {
        // Check if point is inside bounds
        if !point_in_bounds(&point, &node.bounds) {
            return;
        }

        // If leaf node and not full, add point
        if node.children.is_none() && (node.points.len() < max_points || node.depth >= max_depth) {
            node.points.push((point, data));
            return;
        }

        // Subdivide if necessary
        if node.children.is_none() {
            node.subdivide();
        }

        // Calculate child index first before borrowing children
        let child_idx = node.get_child_index(&point);

        // Insert into child
        if let Some(ref mut children) = node.children {
            Self::insert_recursive(&mut children[child_idx], point, data, max_depth, max_points);
        }
    }

    pub fn search_radius(&self, query: &Point3<f32>, radius: f32) -> Vec<(Point3<f32>, T, f32)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::search_radius_recursive(root, query, radius * radius, &mut results);
        }
        results
    }

    fn search_radius_recursive(
        node: &OctreeNode<T>,
        query: &Point3<f32>,
        radius_sq: f32,
        results: &mut Vec<(Point3<f32>, T, f32)>,
    ) {
        // Check if node intersects search sphere
        let closest = closest_point_in_bounds(query, &node.bounds);
        let dist_sq = squared_distance(query, &closest);

        if dist_sq > radius_sq {
            return;
        }

        // Check points in this node
        for (point, data) in &node.points {
            let d = squared_distance(query, point);
            if d <= radius_sq {
                results.push((*point, data.clone(), d));
            }
        }

        // Recurse into children
        if let Some(ref children) = node.children {
            for child in children.iter() {
                Self::search_radius_recursive(child, query, radius_sq, results);
            }
        }
    }
}

impl<T: Clone> OctreeNode<T> {
    fn new(bounds: (Point3<f32>, Point3<f32>), center: Point3<f32>, depth: usize) -> Self {
        Self {
            bounds,
            center,
            children: None,
            points: Vec::new(),
            depth,
        }
    }

    fn subdivide(&mut self) {
        let (min, max) = self.bounds;
        let mid = self.center;

        // Create 8 children (octants)
        let children: Vec<OctreeNode<T>> = (0..8)
            .map(|i| {
                let (min_x, max_x) = if i & 1 == 0 {
                    (min.x, mid.x)
                } else {
                    (mid.x, max.x)
                };
                let (min_y, max_y) = if i & 2 == 0 {
                    (min.y, mid.y)
                } else {
                    (mid.y, max.y)
                };
                let (min_z, max_z) = if i & 4 == 0 {
                    (min.z, mid.z)
                } else {
                    (mid.z, max.z)
                };

                let child_min = Point3::new(min_x, min_y, min_z);
                let child_max = Point3::new(max_x, max_y, max_z);
                let child_center = Point3::new(
                    (min_x + max_x) * 0.5,
                    (min_y + max_y) * 0.5,
                    (min_z + max_z) * 0.5,
                );

                OctreeNode::new((child_min, child_max), child_center, self.depth + 1)
            })
            .collect();

        assert_eq!(children.len(), 8, "Expected exactly 8 children for octree");
        let children_array: [OctreeNode<T>; 8] = children
            .try_into()
            .ok()
            .expect("Failed to convert Vec to array - this should never happen");
        self.children = Some(Box::new(children_array));

        // Get center for child index calculation
        let center = self.center;

        // Redistribute points
        if let Some(ref mut children) = self.children {
            for (point, data) in self.points.drain(..) {
                let idx = get_child_index(&center, &point);
                children[idx].points.push((point, data));
            }
        }
    }

    fn get_child_index(&self, point: &Point3<f32>) -> usize {
        get_child_index(&self.center, point)
    }
}

fn get_child_index(center: &Point3<f32>, point: &Point3<f32>) -> usize {
    let mut idx = 0;
    if point.x >= center.x {
        idx |= 1;
    }
    if point.y >= center.y {
        idx |= 2;
    }
    if point.z >= center.z {
        idx |= 4;
    }
    idx
}

/// VoxelGrid for voxelization
pub struct VoxelGrid {
    pub origin: Point3<f32>,
    pub voxel_size: f32,
    pub grid: HashMap<(i32, i32, i32), Voxel>,
}

#[derive(Debug, Clone)]
pub struct Voxel {
    pub indices: Vec<usize>,
    pub centroid: Option<Point3<f32>>,
}

impl VoxelGrid {
    pub fn new(origin: Point3<f32>, voxel_size: f32) -> Self {
        Self {
            origin,
            voxel_size,
            grid: HashMap::new(),
        }
    }

    pub fn insert(&mut self, point: Point3<f32>, index: usize) {
        let key = self.point_to_voxel(&point);
        self.grid
            .entry(key)
            .or_insert_with(|| Voxel {
                indices: Vec::new(),
                centroid: None,
            })
            .indices
            .push(index);
    }

    pub fn point_to_voxel(&self, point: &Point3<f32>) -> (i32, i32, i32) {
        (
            ((point.x - self.origin.x) / self.voxel_size).floor() as i32,
            ((point.y - self.origin.y) / self.voxel_size).floor() as i32,
            ((point.z - self.origin.z) / self.voxel_size).floor() as i32,
        )
    }

    pub fn voxel_to_point(&self, voxel: (i32, i32, i32)) -> Point3<f32> {
        Point3::new(
            voxel.0 as f32 * self.voxel_size + self.origin.x,
            voxel.1 as f32 * self.voxel_size + self.origin.y,
            voxel.2 as f32 * self.voxel_size + self.origin.z,
        )
    }

    pub fn compute_centroids(&mut self, points: &[Point3<f32>]) {
        for voxel in self.grid.values_mut() {
            if !voxel.indices.is_empty() {
                let mut centroid = Point3::origin();
                for &idx in &voxel.indices {
                    centroid += points[idx].coords;
                }
                centroid /= voxel.indices.len() as f32;
                voxel.centroid = Some(centroid);
            }
        }
    }

    pub fn downsample(&self, points: &[Point3<f32>]) -> Vec<Point3<f32>> {
        self.grid
            .values()
            .filter_map(|voxel| {
                if voxel.indices.is_empty() {
                    None
                } else {
                    let mut centroid = Point3::origin();
                    for &idx in &voxel.indices {
                        centroid += points[idx].coords;
                    }
                    Some(centroid / voxel.indices.len() as f32)
                }
            })
            .collect()
    }
}

fn point_in_bounds(point: &Point3<f32>, bounds: &(Point3<f32>, Point3<f32>)) -> bool {
    point.x >= bounds.0.x
        && point.x <= bounds.1.x
        && point.y >= bounds.0.y
        && point.y <= bounds.1.y
        && point.z >= bounds.0.z
        && point.z <= bounds.1.z
}

fn closest_point_in_bounds(
    query: &Point3<f32>,
    bounds: &(Point3<f32>, Point3<f32>),
) -> Point3<f32> {
    Point3::new(
        query.x.clamp(bounds.0.x, bounds.1.x),
        query.y.clamp(bounds.0.y, bounds.1.y),
        query.z.clamp(bounds.0.z, bounds.1.z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_nearest_neighbor() {
        let mut tree = KDTree::<usize>::new();

        // Insert 5 known points with their index as data
        let points = [
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(-2.0, -2.0, -2.0),
        ];
        for (i, &p) in points.iter().enumerate() {
            tree.insert(p, i);
        }

        // Query a point closest to (0.9, 0.1, 0.1) -- should match points[0] = (1,0,0)
        let query = Point3::new(0.9, 0.1, 0.1);
        let result = tree.nearest_neighbor(&query);
        assert!(result.is_some(), "KDTree should find a nearest neighbor");

        let (nearest_point, data, dist_sq) = result.unwrap();
        assert_eq!(data, 0, "Nearest neighbor should be point index 0");
        assert!(
            (nearest_point.x - 1.0).abs() < 1e-6
                && (nearest_point.y - 0.0).abs() < 1e-6
                && (nearest_point.z - 0.0).abs() < 1e-6,
            "Nearest point should be (1, 0, 0), got {:?}",
            nearest_point
        );

        // Also verify distance is correct: ||(0.9,0.1,0.1) - (1,0,0)|| = sqrt(0.01+0.01+0.01)
        let expected_dist_sq = 0.03;
        assert!(
            (dist_sq - expected_dist_sq).abs() < 1e-5,
            "Expected squared distance ~{}, got {}",
            expected_dist_sq,
            dist_sq
        );

        // Query near (5,5,5) should match points[3]
        let query2 = Point3::new(4.9, 5.1, 5.0);
        let (_, data2, _) = tree.nearest_neighbor(&query2).unwrap();
        assert_eq!(data2, 3, "Nearest neighbor should be point index 3");
    }
}
