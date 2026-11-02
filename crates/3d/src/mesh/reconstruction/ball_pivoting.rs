//! Ball Pivoting Algorithm (BPA)
//!
//! Implements the Ball Pivoting Algorithm by Bernardini et al. (1999)
//! for surface reconstruction from point clouds.
//!
//! A triangle is accepted only when a ball of radius `ball_radius` touches its
//! three vertices **and contains no other point** (the defining BPA condition),
//! with the ball on the side the point normals face. Triangles are wound so
//! their normals agree with the point normals, and every shared edge is
//! traversed in opposite directions by its two triangles, so the result is a
//! consistently oriented, non-overlapping surface. On a plane the empty-ball
//! condition is the empty-circumcircle condition, so the triangles are Delaunay
//! triangles with circumradius at most `ball_radius`.

use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

/// Ball Pivoting Algorithm
///
/// Uses the cloud's normals to decide which side of the surface the ball
/// rolls on (as Open3D's `create_from_point_cloud_ball_pivoting` requires).
/// A cloud without normals gets them from
/// [`super::compute_point_normals`], which orients them into the positive-z
/// half-space - adequate for a height field, not for a closed object, so pass
/// oriented normals for those.
pub fn ball_pivoting(cloud: &PointCloud, ball_radius: f32) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();

    if cloud.points.len() < 3 {
        return mesh;
    }

    mesh.vertices = cloud.points.clone();
    if !(ball_radius > 0.0) || !ball_radius.is_finite() {
        return mesh;
    }

    let normals: Vec<Vector3<f64>> = match &cloud.normals {
        Some(n) if n.len() == cloud.points.len() => n.iter().map(|v| v.cast::<f64>()).collect(),
        _ => super::compute_point_normals(cloud, 10)
            .iter()
            .map(|v| v.cast::<f64>())
            .collect(),
    };
    let points: Vec<Point3<f64>> = cloud.points.iter().map(|p| p.cast::<f64>()).collect();

    let mut bpa = Bpa {
        points: &points,
        normals: &normals,
        radius: ball_radius as f64,
        index: SpatialIndex::new(cloud, ball_radius * 2.0),
        edge_count: HashMap::new(),
        directed: HashSet::new(),
        used: vec![false; points.len()],
        faces: Vec::new(),
    };

    // The previous version accepted any triangle whose circumradius fitted the
    // ball, never asked whether the ball was empty, and placed the ball on
    // whichever side the vertex order happened to give: on a jittered 10x10
    // planar grid at radius 0.2 it produced 845 triangles covering 19.27 m^2
    // of a 0.86 m^2 patch, half of them facing down.
    for i in 0..points.len() {
        if bpa.used[i] || !is_finite(&points[i]) {
            continue;
        }
        if let Some((seed, center)) = bpa.find_seed(i) {
            bpa.add_face(seed);
            bpa.expand(seed, center);
        }
    }

    mesh.faces = bpa.faces;
    mesh.compute_vertex_normals();

    mesh
}

fn is_finite(p: &Point3<f64>) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

pub(crate) struct SpatialIndex {
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
}

impl SpatialIndex {
    pub(crate) fn new(cloud: &PointCloud, cell_size: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (i, point) in cloud.points.iter().enumerate() {
            if !(point.x.is_finite() && point.y.is_finite() && point.z.is_finite()) {
                continue;
            }
            grid.entry(Self::cell(point, cell_size))
                .or_default()
                .push(i);
        }

        Self { grid, cell_size }
    }

    fn cell(point: &Point3<f32>, cell_size: f32) -> (i32, i32, i32) {
        (
            (point.x / cell_size).floor() as i32,
            (point.y / cell_size).floor() as i32,
            (point.z / cell_size).floor() as i32,
        )
    }

    /// Every indexed point in the cells within `radius` of `point` (a
    /// superset of the points within `radius`; callers filter by distance).
    pub(crate) fn find_neighbors(&self, point: &Point3<f32>, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let r = (radius / self.cell_size).ceil() as i32;
        let (cx, cy, cz) = Self::cell(point, self.cell_size);

        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some(indices) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        neighbors.extend(indices);
                    }
                }
            }
        }

        neighbors
    }
}

/// Centre of the ball of `radius` resting on the triangle `(p1, p2, p3)`, on
/// the side its winding normal `(p2 - p1) x (p3 - p1)` points to, or `None`
/// when the triangle is degenerate or its circumradius exceeds `radius`.
#[cfg(test)]
fn find_ball_center(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
    radius: f32,
) -> Option<Point3<f32>> {
    ball_center(
        &p1.cast::<f64>(),
        &p2.cast::<f64>(),
        &p3.cast::<f64>(),
        radius as f64,
    )
    .map(|c| c.cast::<f32>())
}

fn ball_center(
    p1: &Point3<f64>,
    p2: &Point3<f64>,
    p3: &Point3<f64>,
    radius: f64,
) -> Option<Point3<f64>> {
    let a = p2 - p1;
    let b = p3 - p1;
    let normal = a.cross(&b);
    let normal_len_sq = normal.norm_squared();
    // Relative to the edge lengths: an absolute floor on this length^2 quantity
    // called every small triangle degenerate.
    if !(normal_len_sq > 1e-24 * a.norm_squared() * b.norm_squared()) {
        return None;
    }

    // Circumcentre relative to p1: (|a|^2 (b x n) + |b|^2 (n x a)) / (2 |n|^2).
    let offset = (b.cross(&normal) * a.norm_squared() + normal.cross(&a) * b.norm_squared())
        / (2.0 * normal_len_sq);
    let h_sq = radius * radius - offset.norm_squared();
    if h_sq < 0.0 {
        return None;
    }
    let n_hat = normal / normal_len_sq.sqrt();
    Some(p1 + offset + n_hat * h_sq.sqrt())
}

/// Canonical (undirected) edge key: smaller index first.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// A directed edge on the expansion front, in the winding of the triangle it
/// belongs to; `opposite` is that triangle's third vertex and `ball_center`
/// the centre of the ball resting on it.
struct FrontEdge {
    a: usize,
    b: usize,
    opposite: usize,
    ball_center: Point3<f64>,
}

struct Bpa<'a> {
    points: &'a [Point3<f64>],
    normals: &'a [Vector3<f64>],
    radius: f64,
    index: SpatialIndex,
    /// Triangles per undirected edge.
    edge_count: HashMap<(usize, usize), u8>,
    /// Directed edges already used by a triangle.
    directed: HashSet<(usize, usize)>,
    used: Vec<bool>,
    faces: Vec<[usize; 3]>,
}

impl Bpa<'_> {
    fn neighbors(&self, center: &Point3<f64>, radius: f64) -> Vec<usize> {
        let r2 = radius * radius;
        self.index
            .find_neighbors(&center.cast::<f32>(), radius as f32)
            .into_iter()
            .filter(|&j| (self.points[j] - center).norm_squared() <= r2)
            .collect()
    }

    /// No point other than the triangle's own lies strictly inside the ball.
    /// Points on the sphere (co-circular samples) are allowed, with a relative
    /// tolerance for rounding.
    fn ball_is_empty(&self, center: &Point3<f64>, tri: [usize; 3]) -> bool {
        let inner = self.radius * (1.0 - 1e-6);
        self.neighbors(center, self.radius)
            .into_iter()
            .all(|j| tri.contains(&j) || (self.points[j] - center).norm_squared() >= inner * inner)
    }

    /// The winding normal of `tri` agrees with the normal of each vertex.
    fn compatible(&self, tri: [usize; 3]) -> bool {
        let [i, j, k] = tri;
        let n = (self.points[j] - self.points[i]).cross(&(self.points[k] - self.points[i]));
        tri.iter().all(|&v| n.dot(&self.normals[v]) > 0.0)
    }

    /// The ball resting on `tri` when it is a valid BPA triangle: compatible
    /// with the normals, ball on the normal side, ball empty.
    fn valid_ball(&self, tri: [usize; 3]) -> Option<Point3<f64>> {
        if !self.compatible(tri) {
            return None;
        }
        let center = ball_center(
            &self.points[tri[0]],
            &self.points[tri[1]],
            &self.points[tri[2]],
            self.radius,
        )?;
        self.ball_is_empty(&center, tri).then_some(center)
    }

    fn find_seed(&self, i: usize) -> Option<([usize; 3], Point3<f64>)> {
        let p = self.points[i];
        let mut nbrs: Vec<usize> = self
            .neighbors(&p, 2.0 * self.radius)
            .into_iter()
            .filter(|&j| j != i && !self.used[j])
            .collect();
        nbrs.sort_by(|&a, &b| {
            (self.points[a] - p)
                .norm_squared()
                .total_cmp(&(self.points[b] - p).norm_squared())
                .then(a.cmp(&b))
        });
        for x in 0..nbrs.len() {
            for y in (x + 1)..nbrs.len() {
                let (j, k) = (nbrs[x], nbrs[y]);
                // Wind the triangle so its normal faces the point normals.
                let n = (self.points[j] - p).cross(&(self.points[k] - p));
                let facing = n.dot(&(self.normals[i] + self.normals[j] + self.normals[k]));
                let tri = if facing >= 0.0 { [i, j, k] } else { [i, k, j] };
                if let Some(center) = self.valid_ball(tri) {
                    return Some((tri, center));
                }
            }
        }
        None
    }

    /// Whether `tri` can join the mesh without overlapping it: no undirected
    /// edge already has two triangles and no directed edge is used twice
    /// (which would put two triangles on the same side of an edge).
    fn can_add(&self, tri: [usize; 3]) -> bool {
        (0..3).all(|e| {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            !self.directed.contains(&(a, b))
                && *self.edge_count.get(&edge_key(a, b)).unwrap_or(&0) < 2
        })
    }

    fn add_face(&mut self, tri: [usize; 3]) {
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            self.directed.insert((a, b));
            *self.edge_count.entry(edge_key(a, b)).or_insert(0) += 1;
        }
        for &v in &tri {
            self.used[v] = true;
        }
        self.faces.push(tri);
    }

    /// Roll the ball over the directed edge `a -> b` and return the first point
    /// it touches, with the new ball centre: the candidate reached by the
    /// smallest rotation about the edge (Bernardini et al. 1999).
    fn pivot(&self, edge: &FrontEdge) -> Option<(usize, Point3<f64>)> {
        let pa = self.points[edge.a];
        let pb = self.points[edge.b];
        let mid = Point3::from((pa.coords + pb.coords) * 0.5);
        let axis = (pb - pa).try_normalize(0.0)?;
        let project = |c: &Point3<f64>| {
            let v = c - mid;
            v - axis * v.dot(&axis)
        };
        let u = project(&edge.ball_center);

        let mut best: Option<(f64, usize, Point3<f64>)> = None;
        for c in self.neighbors(&mid, 2.0 * self.radius) {
            if c == edge.a || c == edge.b || c == edge.opposite {
                continue;
            }
            // The new triangle crosses the shared edge as b -> a.
            let tri = [edge.b, edge.a, c];
            let Some(center) = self.valid_ball(tri) else {
                continue;
            };
            let v = project(&center);
            // Rotation about +axis carries the old ball over the edge to the
            // outside of its triangle.
            let mut angle = u.cross(&v).dot(&axis).atan2(u.dot(&v));
            if angle < -1e-6 {
                angle += std::f64::consts::TAU;
            }
            let angle = angle.max(0.0);
            if best.as_ref().is_none_or(|b| angle < b.0) {
                best = Some((angle, c, center));
            }
        }
        best.map(|(_, c, center)| (c, center))
    }

    fn expand(&mut self, seed: [usize; 3], center: Point3<f64>) {
        let mut front: VecDeque<FrontEdge> = (0..3)
            .map(|e| FrontEdge {
                a: seed[e],
                b: seed[(e + 1) % 3],
                opposite: seed[(e + 2) % 3],
                ball_center: center,
            })
            .collect();

        while let Some(edge) = front.pop_front() {
            if *self.edge_count.get(&edge_key(edge.a, edge.b)).unwrap_or(&0) >= 2 {
                continue;
            }
            let Some((c, new_center)) = self.pivot(&edge) else {
                continue; // a boundary edge
            };
            let tri = [edge.b, edge.a, c];
            if !self.can_add(tri) {
                continue;
            }
            self.add_face(tri);
            for (a, b, opposite) in [(edge.a, c, edge.b), (c, edge.b, edge.a)] {
                if *self.edge_count.get(&edge_key(a, b)).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a,
                        b,
                        opposite,
                        ball_center: new_center,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_find_ball_center_is_equidistant() {
        // Right triangle with legs 1: circumcenter is (0.5, 0.5, 0),
        // circumradius sqrt(2)/2 ~= 0.7071.
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let radius = 1.0;

        let center = find_ball_center(&a, &b, &c, radius).expect("ball center exists");

        // The returned center is a ball of `radius` resting on the triangle,
        // so it must be exactly `radius` away from all three vertices.
        assert!(((center - a).norm() - radius).abs() < 1e-4);
        assert!(((center - b).norm() - radius).abs() < 1e-4);
        assert!(((center - c).norm() - radius).abs() < 1e-4);

        // Its in-plane projection is the circumcenter (0.5, 0.5, 0).
        assert!((center.x - 0.5).abs() < 1e-4, "x = {}", center.x);
        assert!((center.y - 0.5).abs() < 1e-4, "y = {}", center.y);
        assert!(
            center.z.abs() > 0.5,
            "should be off-plane, z = {}",
            center.z
        );
    }
}
