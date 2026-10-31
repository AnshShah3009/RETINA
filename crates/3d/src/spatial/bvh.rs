//! Bounding Volume Hierarchy (BVH) for accelerated ray-triangle intersection.
//!
//! Builds a binary tree of axis-aligned bounding boxes (AABBs) over triangle
//! meshes, enabling O(log N) ray intersection instead of O(N) brute force.

use nalgebra::{Point3, Vector3};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
    /// `false` until a point or another non-empty Aabb is merged in.
    ///
    /// This field exists because the obvious sentinel - `min = f32::MAX`,
    /// `max = f32::MIN` - cannot express "empty". `f32::MIN` is the most
    /// *negative* finite f32, so `max` could never rise above it, and an Aabb
    /// that never received a point reported
    ///
    /// ```text
    /// min = (3.403e38, 3.403e38, 3.403e38)
    /// max = (-3.403e38, -3.403e38, -3.403e38)
    /// ```
    ///
    /// with min greater than max on every axis. Two consequences, both measured:
    /// `Aabb::from_points(&[])` returned exactly that, and `longest_axis()`
    /// computed `max - min = -6.8e38`, which **overflows to -inf** in f32, so
    /// both `d.x > d.y` and `d.y > d.z` were false and it returned axis 2 for
    /// every input. In `build_recursive` that means a degenerate face - one with
    /// a repeated index, so all three of its vertices coincide - silently
    /// determines the split axis of the whole tree.
    ///
    /// Emptiness is now tracked explicitly, so an empty Aabb can be recognised
    /// instead of being inferred from a corrupt range.
    pub is_empty: bool,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
            is_empty: true,
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
        self.is_empty = false;
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        // Merging an empty Aabb must leave the other untouched. Without this the
        // sentinel values participated in the min/max, so merging two empties
        // produced a third that still looked empty but could no longer be told
        // apart from a real degenerate box.
        if other.is_empty {
            return self.clone();
        }
        if self.is_empty {
            return other.clone();
        }
        Aabb {
            is_empty: false,
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

    /// Midpoint of the box.
    ///
    /// An empty Aabb has no midpoint; the sentinel bounds would overflow here
    /// too, so callers that can receive one must check [`Aabb::is_empty`] first.
    pub fn centroid(&self) -> Point3<f32> {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Slab-based ray-AABB intersection. Returns true if ray hits the box.
    pub fn intersect_ray(&self, origin: &Point3<f32>, inv_dir: &Vector3<f32>) -> bool {
        // An empty box contains nothing, so no ray hits it.
        //
        // Measured before this guard: `intersect_ray` returned `true` for an
        // empty Aabb from *every* origin and direction. The sentinel bounds make
        // `max - min` overflow to -inf, so `tmin` and `tmax` both became
        // non-finite and the final `tmax >= tmin.max(0.0)` comparison was
        // decided by NaN ordering rather than by geometry. In a BVH traversal
        // that is a subtree reported as hit for rays that pass nowhere near it,
        // so the traversal descends into boxes it should have culled.
        if self.is_empty {
            return false;
        }
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
        // An empty Aabb has no extent. Returning a fixed axis rather than
        // subtracting the sentinel pair, which overflows to -inf and makes every
        // comparison below false - that picked axis 2 for every input, including
        // meshes whose geometry lies entirely in x.
        if self.is_empty {
            return 0;
        }
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

/// Angular tolerance for "this ray is parallel to this triangle", as a
/// dimensionless bound on `|cos(angle between ray and triangle normal)|`.
///
/// Chosen to match the absolute constant this test used to be, so the fix
/// changes *scale* behaviour only and leaves the angular sharpness alone.
const PARALLEL_EPS: f32 = 1e-9;

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
    // `a` is twice the projected triangle area, so for a genuine hit it scales
    // as L^2 - the *square* of the edge length. Comparing it against a fixed
    // constant is therefore not an angular test, it is a size test, and the
    // size it rejects moves with the world units the mesh happens to be stored
    // in. Measured: an equilateral triangle of edge 1 hit dead-on gives
    // |a| = 0.866; of edge 1e-3, |a| = 8.66e-7, below the 1e-6 threshold used
    // to be in `raycasting::ray_triangle_intersection`; of edge 1e-5,
    // |a| = 8.66e-11, below this function's own 1e-9. A metre-unit mesh of
    // 1 mm triangles - exactly what a surface reconstruction emits - was
    // rejected as "parallel to triangle".
    //
    // Normalising by |e1||e2| turns it back into what it was meant to be: a
    // dimensionless bound on the angle between the ray and the normal.
    let denom = e1.norm() * e2.norm();
    if denom == 0.0 || a.abs() < PARALLEL_EPS * denom {
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
