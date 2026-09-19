//! Ball Pivoting Algorithm (BPA)
//!
//! Implements the Ball Pivoting Algorithm by Bernardini et al. (1999)
//! for surface reconstruction from point clouds.

use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::Point3;
use std::collections::{HashMap, HashSet, VecDeque};

/// Ball Pivoting Algorithm
pub fn ball_pivoting(cloud: &PointCloud, ball_radius: f32) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();

    if cloud.points.len() < 3 {
        return mesh;
    }

    let spatial_index = SpatialIndex::new(cloud, ball_radius * 2.0);

    let mut edge_count: HashMap<(usize, usize), u8> = HashMap::new();
    let mut mesh_faces: Vec<[usize; 3]> = Vec::new();
    let mut used_in_mesh: HashSet<usize> = HashSet::new();

    for i in 0..cloud.points.len() {
        // Skip vertices already incorporated into the mesh
        if used_in_mesh.contains(&i) {
            continue;
        }
        if let Some(seed) = find_seed_triangle(i, cloud, &spatial_index, ball_radius) {
            // Only start a new seed if none of the seed vertices have all edges saturated
            if try_add_triangle(seed, &mut edge_count, &mut mesh_faces) {
                for &v in &seed {
                    used_in_mesh.insert(v);
                }
                expand_front(
                    seed,
                    cloud,
                    &spatial_index,
                    ball_radius,
                    &mut edge_count,
                    &mut mesh_faces,
                    &mut used_in_mesh,
                );
            }
        }
    }

    mesh.vertices = cloud.points.clone();
    mesh.faces = mesh_faces;
    mesh.compute_vertex_normals();

    mesh
}

pub(crate) struct SpatialIndex {
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
}

impl SpatialIndex {
    pub(crate) fn new(cloud: &PointCloud, cell_size: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (i, point) in cloud.points.iter().enumerate() {
            let key = (
                (point.x / cell_size) as i32,
                (point.y / cell_size) as i32,
                (point.z / cell_size) as i32,
            );
            grid.entry(key).or_default().push(i);
        }

        Self { grid, cell_size }
    }

    pub(crate) fn find_neighbors(&self, point: &Point3<f32>, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let r = (radius / self.cell_size).ceil() as i32;
        let cx = (point.x / self.cell_size) as i32;
        let cy = (point.y / self.cell_size) as i32;
        let cz = (point.z / self.cell_size) as i32;

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

fn find_seed_triangle(
    start_idx: usize,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
) -> Option<[usize; 3]> {
    let point = &cloud.points[start_idx];
    let neighbors = spatial_index.find_neighbors(point, ball_radius * 2.0);

    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            let idx1 = neighbors[i];
            let idx2 = neighbors[j];

            if find_ball_center(
                &cloud.points[start_idx],
                &cloud.points[idx1],
                &cloud.points[idx2],
                ball_radius,
            )
            .is_some()
            {
                return Some([start_idx, idx1, idx2]);
            }
        }
    }

    None
}

fn find_ball_center(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
    radius: f32,
) -> Option<Point3<f32>> {
    let a = p2 - p1;
    let b = p3 - p1;
    let normal = a.cross(&b);
    let normal_len = normal.norm();

    if normal_len < 1e-6 {
        return None;
    }

    let normal = normal / normal_len;

    let a_len = a.norm();
    let b_len = b.norm();
    let c_len = (p3 - p2).norm();

    let circumradius = (a_len * b_len * c_len) / (2.0 * normal_len);

    if circumradius > radius {
        return None;
    }

    let a_len_sq = a_len * a_len;
    let b_len_sq = b_len * b_len;
    let c_len_sq = c_len * c_len;

    let denom = 2.0 * normal_len * normal_len;
    let alpha = b_len_sq * c_len_sq * (a_len_sq + b_len_sq - c_len_sq) / denom;
    let beta = a_len_sq * c_len_sq * (a_len_sq - b_len_sq + c_len_sq) / denom;
    let gamma = a_len_sq * b_len_sq * (-a_len_sq + b_len_sq + c_len_sq) / denom;

    let circumcenter = p1 * alpha + p2.coords * beta + p3.coords * gamma;

    let height = (radius * radius - circumradius * circumradius).sqrt();
    let center1 = circumcenter + normal * height;

    Some(center1)
}

/// Canonical (undirected) edge key: smaller index first.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn try_add_triangle(
    face: [usize; 3],
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
) -> bool {
    let e0 = edge_key(face[0], face[1]);
    let e1 = edge_key(face[1], face[2]);
    let e2 = edge_key(face[2], face[0]);

    // An edge that already has 2 uses is complete; cannot add another triangle to it.
    if *edge_count.get(&e0).unwrap_or(&0) >= 2
        || *edge_count.get(&e1).unwrap_or(&0) >= 2
        || *edge_count.get(&e2).unwrap_or(&0) >= 2
    {
        return false;
    }

    *edge_count.entry(e0).or_insert(0) += 1;
    *edge_count.entry(e1).or_insert(0) += 1;
    *edge_count.entry(e2).or_insert(0) += 1;
    mesh_faces.push(face);

    true
}

/// A directed edge on the expansion front.
/// `a` and `b` are the edge vertices (in winding order), `opposite` is the
/// vertex of the triangle on the other side of this edge, and `ball_center`
/// is the centre of the ball that rests on that triangle.
struct FrontEdge {
    a: usize,
    b: usize,
    opposite: usize,
    ball_center: Point3<f32>,
}

/// Attempt to find the point `c` that the ball of the given `radius` touches
/// when pivoting around the directed edge (a -> b).
///
/// Among all candidate points in the neighborhood, we choose the one with the
/// smallest positive pivot angle (Bernardini et al. 1999).
fn pivot_ball(
    edge: &FrontEdge,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &HashMap<(usize, usize), u8>,
) -> Option<(usize, Point3<f32>)> {
    let pa = &cloud.points[edge.a];
    let pb = &cloud.points[edge.b];

    // Edge midpoint and axis
    let mid = Point3::from((pa.coords + pb.coords) * 0.5);
    let edge_vec = pb - pa;
    let edge_len = edge_vec.norm();
    if edge_len < 1e-10 {
        return None;
    }
    let edge_axis = edge_vec / edge_len;

    // Reference direction: from edge midpoint towards the old ball centre,
    // projected onto the plane perpendicular to the edge axis.
    let old_center_vec = edge.ball_center - mid;
    let old_center_proj = old_center_vec - edge_axis * old_center_vec.dot(&edge_axis);
    let proj_len = old_center_proj.norm();
    if proj_len < 1e-10 {
        return None;
    }

    let candidates = spatial_index.find_neighbors(&mid, ball_radius * 2.0);

    let mut best_angle = f32::MAX;
    let mut best_candidate: Option<(usize, Point3<f32>)> = None;

    for &idx in &candidates {
        // Skip the edge endpoints and the opposite vertex of the current triangle
        if idx == edge.a || idx == edge.b || idx == edge.opposite {
            continue;
        }

        // Skip if both edges from the candidate are already complete (count >= 2)
        let ek_ca = edge_key(idx, edge.a);
        let ek_cb = edge_key(idx, edge.b);
        if *edge_count.get(&ek_ca).unwrap_or(&0) >= 2 || *edge_count.get(&ek_cb).unwrap_or(&0) >= 2
        {
            continue;
        }

        // Try to place a ball on (a, b, candidate)
        if let Some(new_center) = find_ball_center(pa, pb, &cloud.points[idx], ball_radius) {
            // The new ball centre must be on the correct side. We check this
            // by computing the pivot angle around the edge axis.
            let new_center_vec = new_center - mid;
            let new_center_proj = new_center_vec - edge_axis * new_center_vec.dot(&edge_axis);
            let new_proj_len = new_center_proj.norm();
            if new_proj_len < 1e-10 {
                continue;
            }

            // Pivot angle: angle from old_center_proj to new_center_proj
            // around the edge axis, measured in the positive rotation direction.
            let cos_angle = old_center_proj.dot(&new_center_proj) / (proj_len * new_proj_len);
            let sin_angle =
                old_center_proj.cross(&new_center_proj).dot(&edge_axis) / (proj_len * new_proj_len);
            let mut angle = sin_angle.atan2(cos_angle);

            // We want the angle in (0, 2*PI] — the smallest positive pivot.
            if angle <= 1e-6 {
                angle += std::f32::consts::TAU;
            }

            if angle < best_angle {
                best_angle = angle;
                best_candidate = Some((idx, new_center));
            }
        }
    }

    best_candidate
}

fn expand_front(
    seed: [usize; 3],
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
    used_in_mesh: &mut HashSet<usize>,
) {
    // Compute ball centre for the seed triangle
    let seed_center = match find_ball_center(
        &cloud.points[seed[0]],
        &cloud.points[seed[1]],
        &cloud.points[seed[2]],
        ball_radius,
    ) {
        Some(c) => c,
        None => return,
    };

    // Initialise the front with the three directed edges of the seed triangle.
    // For triangle (a, b, c), the three directed edges are:
    //   (a->b, opposite=c), (b->c, opposite=a), (c->a, opposite=b)
    let mut front: VecDeque<FrontEdge> = VecDeque::new();
    front.push_back(FrontEdge {
        a: seed[0],
        b: seed[1],
        opposite: seed[2],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[1],
        b: seed[2],
        opposite: seed[0],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[2],
        b: seed[0],
        opposite: seed[1],
        ball_center: seed_center,
    });

    // Safety limit to avoid infinite loops on degenerate inputs.
    let max_iterations = cloud.points.len() * 3;
    let mut iterations = 0;

    while let Some(edge) = front.pop_front() {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        // If the edge is already complete (used twice), skip it.
        let ek = edge_key(edge.a, edge.b);
        if *edge_count.get(&ek).unwrap_or(&0) >= 2 {
            continue;
        }

        // Pivot the ball around this edge to find the next point.
        if let Some((c, new_center)) =
            pivot_ball(&edge, cloud, spatial_index, ball_radius, edge_count)
        {
            let face = [edge.a, edge.b, c];

            if try_add_triangle(face, edge_count, mesh_faces) {
                used_in_mesh.insert(c);

                // Add the two new edges to the front.
                // New edge (b -> c): opposite is a
                let ek_bc = edge_key(edge.b, c);
                if *edge_count.get(&ek_bc).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: edge.b,
                        b: c,
                        opposite: edge.a,
                        ball_center: new_center,
                    });
                }

                // New edge (c -> a): opposite is b
                let ek_ca = edge_key(c, edge.a);
                if *edge_count.get(&ek_ca).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: c,
                        b: edge.a,
                        opposite: edge.b,
                        ball_center: new_center,
                    });
                }
            }
        }
    }
}
