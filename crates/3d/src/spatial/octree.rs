use nalgebra::Point3;

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

fn squared_distance(a: &Point3<f32>, b: &Point3<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}
