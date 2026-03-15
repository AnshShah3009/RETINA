//! Bounding Volume Hierarchy (BVH) for accelerated ray-triangle intersection.
//!
//! Builds a binary tree of axis-aligned bounding boxes (AABBs) over triangle
//! meshes, enabling O(log N) ray intersection instead of O(N) brute force.

use nalgebra::{Point3, Vector3};

/// Axis-aligned bounding box.
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    pub fn from_points(pts: &[Point3<f32>]) -> Self {
        let mut b = Self::empty();
        for p in pts {
            b.expand_point(p);
        }
        b
    }

    pub fn expand_point(&mut self, p: &Point3<f32>) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn centroid(&self) -> Point3<f32> {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Slab-based ray-AABB intersection. Returns true if ray hits the box.
    pub fn intersect_ray(&self, origin: &Point3<f32>, inv_dir: &Vector3<f32>) -> bool {
        let t1 = (self.min.x - origin.x) * inv_dir.x;
        let t2 = (self.max.x - origin.x) * inv_dir.x;
        let t3 = (self.min.y - origin.y) * inv_dir.y;
        let t4 = (self.max.y - origin.y) * inv_dir.y;
        let t5 = (self.min.z - origin.z) * inv_dir.z;
        let t6 = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        tmax >= tmin.max(0.0)
    }

    fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }
}

/// BVH node — either a leaf with triangle indices or an internal node with children.
enum BvhNode {
    Leaf {
        bounds: Aabb,
        start: usize,
        count: usize,
    },
    Internal {
        bounds: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

/// A BVH built over triangle mesh faces for fast ray intersection.
pub struct Bvh {
    root: BvhNode,
    /// Reordered triangle indices (BVH construction reorders for locality).
    tri_indices: Vec<usize>,
}

impl Bvh {
    /// Build a BVH from mesh vertices and faces.
    pub fn build(vertices: &[Point3<f32>], faces: &[[usize; 3]]) -> Self {
        let mut tri_data: Vec<(usize, Aabb, Point3<f32>)> = faces
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let aabb = Aabb::from_points(&[vertices[f[0]], vertices[f[1]], vertices[f[2]]]);
                let c = aabb.centroid();
                (i, aabb, c)
            })
            .collect();

        let n = tri_data.len();
        let root = Self::build_recursive(&mut tri_data, 0, n);
        let tri_indices = tri_data.iter().map(|(i, _, _)| *i).collect();
        Self { root, tri_indices }
    }

    fn build_recursive(
        data: &mut [(usize, Aabb, Point3<f32>)],
        start: usize,
        end: usize,
    ) -> BvhNode {
        let count = end - start;
        let slice = &data[start..end];

        // Compute bounds
        let mut bounds = Aabb::empty();
        for (_, aabb, _) in slice {
            bounds = bounds.merge(aabb);
        }

        // Leaf threshold
        if count <= 4 {
            return BvhNode::Leaf {
                bounds,
                start,
                count,
            };
        }

        // Split along longest axis at centroid midpoint
        let axis = bounds.longest_axis();
        let mid_val = match axis {
            0 => (bounds.min.x + bounds.max.x) * 0.5,
            1 => (bounds.min.y + bounds.max.y) * 0.5,
            _ => (bounds.min.z + bounds.max.z) * 0.5,
        };

        // Partition
        let mid = {
            let slice = &mut data[start..end];
            let mut i = 0;
            let mut j = slice.len();
            while i < j {
                let c = match axis {
                    0 => slice[i].2.x,
                    1 => slice[i].2.y,
                    _ => slice[i].2.z,
                };
                if c < mid_val {
                    i += 1;
                } else {
                    j -= 1;
                    slice.swap(i, j);
                }
            }
            start + i
        };

        // Avoid degenerate splits
        let mid = if mid == start || mid == end {
            start + count / 2
        } else {
            mid
        };

        let left = Self::build_recursive(data, start, mid);
        let right = Self::build_recursive(data, mid, end);

        BvhNode::Internal {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Cast a ray against the BVH. Returns the closest hit: (t, face_index, u, v).
    pub fn intersect_ray(
        &self,
        origin: &Point3<f32>,
        dir: &Vector3<f32>,
        vertices: &[Point3<f32>],
        faces: &[[usize; 3]],
    ) -> Option<(f32, usize, f32, f32)> {
        let inv_dir = Vector3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        let mut best: Option<(f32, usize, f32, f32)> = None;
        self.intersect_recursive(
            &self.root, origin, dir, &inv_dir, vertices, faces, &mut best,
        );
        best
    }

    fn intersect_recursive(
        &self,
        node: &BvhNode,
        origin: &Point3<f32>,
        dir: &Vector3<f32>,
        inv_dir: &Vector3<f32>,
        vertices: &[Point3<f32>],
        faces: &[[usize; 3]],
        best: &mut Option<(f32, usize, f32, f32)>,
    ) {
        match node {
            BvhNode::Leaf {
                bounds,
                start,
                count,
            } => {
                if !bounds.intersect_ray(origin, inv_dir) {
                    return;
                }
                for i in *start..(*start + *count) {
                    let fi = self.tri_indices[i];
                    let f = &faces[fi];
                    if let Some((t, u, v)) = moller_trumbore(
                        origin,
                        dir,
                        &vertices[f[0]],
                        &vertices[f[1]],
                        &vertices[f[2]],
                    ) {
                        if t > 1e-6 {
                            let replace = match best {
                                None => true,
                                Some((bt, _, _, _)) => t < *bt,
                            };
                            if replace {
                                *best = Some((t, fi, u, v));
                            }
                        }
                    }
                }
            }
            BvhNode::Internal {
                bounds,
                left,
                right,
            } => {
                if !bounds.intersect_ray(origin, inv_dir) {
                    return;
                }
                self.intersect_recursive(left, origin, dir, inv_dir, vertices, faces, best);
                self.intersect_recursive(right, origin, dir, inv_dir, vertices, faces, best);
            }
        }
    }
}

/// Möller-Trumbore ray-triangle intersection.
fn moller_trumbore(
    origin: &Point3<f32>,
    dir: &Vector3<f32>,
    v0: &Point3<f32>,
    v1: &Point3<f32>,
    v2: &Point3<f32>,
) -> Option<(f32, f32, f32)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(&e2);
    let a = e1.dot(&h);
    if a.abs() < 1e-9 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(&e1);
    let v = f * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(&q);
    Some((t, u, v))
}
