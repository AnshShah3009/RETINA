//! Spatial Data Structures
//!
//! Implements:
//! - KDTree for fast nearest neighbor search
//! - Octree for spatial partitioning
//! - VoxelGrid for voxelization

use nalgebra::Point3;
use std::collections::HashMap;

/// KDTree for nearest neighbor queries
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

impl<T: Clone> KDTree<T> {
    pub fn new() -> Self {
        Self { root: None, dim: 3 }
    }

    pub fn with_capacity(_capacity: usize) -> Self {
        Self::new()
    }

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
                let axis = n.axis;
                let coord = match axis {
                    0 => point.x,
                    1 => point.y,
                    _ => point.z,
                };
                let node_coord = match axis {
                    0 => n.point.x,
                    1 => n.point.y,
                    _ => n.point.z,
                };

                if coord < node_coord {
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

        let diff = match node.axis {
            0 => query.x - node.point.x,
            1 => query.y - node.point.y,
            _ => query.z - node.point.z,
        };

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::nearest_recursive(child, query, best);
        }

        // Check if we need to explore the other side
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

        let diff = match node.axis {
            0 => query.x - node.point.x,
            1 => query.y - node.point.y,
            _ => query.z - node.point.z,
        };

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

    pub fn k_nearest_neighbors(&self, query: &Point3<f32>, k: usize) -> Vec<(Point3<f32>, T, f32)> {
        let mut results = Vec::with_capacity(k);

        if let Some(ref root) = self.root {
            Self::knn_recursive(root, query, k, &mut results);
        }

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results.truncate(k);
        results
    }

    fn knn_recursive(
        node: &KDNode<T>,
        query: &Point3<f32>,
        k: usize,
        results: &mut Vec<(Point3<f32>, T, f32)>,
    ) {
        let dist = squared_distance(&node.point, query);
        results.push((node.point, node.data.clone(), dist));

        let diff = match node.axis {
            0 => query.x - node.point.x,
            1 => query.y - node.point.y,
            _ => query.z - node.point.z,
        };

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::knn_recursive(child, query, k, results);
        }

        // Check if we need to explore the other side
        let max_dist = results.iter().map(|r| r.2).fold(0.0, f32::max);
        if diff * diff < max_dist || results.len() < k {
            if let Some(ref child) = second {
                Self::knn_recursive(child, query, k, results);
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
