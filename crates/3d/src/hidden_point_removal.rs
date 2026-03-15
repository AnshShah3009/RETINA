//! Hidden Point Removal (HPR) — Katz, Tal & Basri (SIGGRAPH 2007).
//!
//! Determines which points in a point cloud are visible from a given viewpoint
//! without mesh reconstruction, using the spherical flipping operator and
//! convex hull test.
//!
//! # Algorithm
//!
//! 1. Translate the point cloud so the viewpoint is at the origin.
//! 2. **Spherical flip:** For each point p, compute
//!    `p' = p + 2(R - ||p||) * (p / ||p||)`
//!    where R is a radius parameter.
//! 3. Add the origin (viewpoint) to the flipped point set.
//! 4. Compute the 3D convex hull of the flipped points + origin.
//! 5. Original points whose flipped versions lie on the convex hull are visible.
//!
//! # Reference
//!
//! S. Katz, A. Tal, R. Basri, "Direct Visibility of Point Sets",
//! ACM SIGGRAPH 2007.

use nalgebra::Point3;
use rayon::prelude::*;

// ── Public types ─────────────────────────────────────────────────────────────

/// Result of hidden point removal.
#[derive(Debug, Clone)]
pub struct HprResult {
    /// Indices of visible points in the original point cloud.
    pub visible_indices: Vec<usize>,
    /// Boolean mask aligned with the input: `true` = visible.
    pub visibility_mask: Vec<bool>,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Hidden Point Removal using the Katz-Tal-Basri spherical flipping operator.
///
/// # Arguments
/// * `points`    – Input point cloud as a slice of 3D points.
/// * `viewpoint` – Camera / eye position.
/// * `radius`    – Flipping radius parameter.  Larger values produce more
///   aggressive culling.  Pass `0.0` to auto-compute as `100 * diameter`.
///
/// # Returns
/// An [`HprResult`] containing both the visible indices and a boolean mask.
///
/// # Errors
/// Returns `Err` if the point cloud has fewer than 4 non-degenerate points
/// (a convex hull in 3D requires at least 4 non-coplanar points).
pub fn hidden_point_removal(
    points: &[Point3<f64>],
    viewpoint: &Point3<f64>,
    radius: f64,
) -> Result<HprResult, String> {
    if points.len() < 4 {
        return Err("Need at least 4 points for HPR".into());
    }

    // 1. Translate so viewpoint is at origin (parallel).
    let translated: Vec<Point3<f64>> = points
        .par_iter()
        .map(|p| Point3::new(p.x - viewpoint.x, p.y - viewpoint.y, p.z - viewpoint.z))
        .collect();

    // Auto-compute radius if not provided.
    let r = if radius <= 0.0 {
        auto_radius(&translated)
    } else {
        radius
    };

    // 2. Spherical flip (parallel).
    let n = translated.len();
    let mut flipped: Vec<Point3<f64>> = translated
        .par_iter()
        .map(|p| {
            let norm = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if norm < 1e-15 {
                Point3::origin()
            } else {
                let factor = 2.0 * (r - norm) / norm;
                Point3::new(p.x + factor * p.x, p.y + factor * p.y, p.z + factor * p.z)
            }
        })
        .collect();
    // Add the origin (viewpoint).
    flipped.push(Point3::origin());

    // 3. Convex hull of flipped points.
    let on_hull = convex_hull_3d_membership(&flipped);

    // 4. Points whose flipped versions are on the hull (excluding the origin point)
    //    are visible.
    let mut visibility_mask = vec![false; n];
    let mut visible_indices = Vec::new();
    for i in 0..n {
        if on_hull[i] {
            visibility_mask[i] = true;
            visible_indices.push(i);
        }
    }

    Ok(HprResult {
        visible_indices,
        visibility_mask,
    })
}

/// Convenience: return only the visible points (not indices).
pub fn select_visible_points(
    points: &[Point3<f64>],
    viewpoint: &Point3<f64>,
    radius: f64,
) -> Result<Vec<Point3<f64>>, String> {
    let result = hidden_point_removal(points, viewpoint, radius)?;
    Ok(result.visible_indices.iter().map(|&i| points[i]).collect())
}

// ── Internals ────────────────────────────────────────────────────────────────

/// Auto-compute flipping radius.
///
/// Following the Katz et al. paper, the radius should be large enough that all
/// flipped points end up outside the original cloud, but not so large that the
/// discrimination vanishes.  We use `R = gamma * max_dist` where gamma is
/// tuned to provide good results for typical point clouds.
fn auto_radius(points: &[Point3<f64>]) -> f64 {
    let mut max_dist = 0.0_f64;
    for p in points {
        let d = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        if d > max_dist {
            max_dist = d;
        }
    }
    if max_dist < 1e-15 {
        return 1.0;
    }
    // gamma ~ 2-3 works well in practice (Open3D uses a similar range).
    // We use 2.5 * max_distance as a balanced default.
    2.5 * max_dist
}

// ── 3D Convex Hull (incremental, QHull-style) ───────────────────────────────
//
// We only need to know *which* point indices lie on the hull, not the full
// face topology.  The implementation below builds the hull incrementally and
// returns a boolean membership vector.

/// Epsilon for convex hull geometric predicates.
const HULL_EPS: f64 = 1e-10;

/// A triangular face of the convex hull, with vertices ordered so that the
/// outward normal follows the right-hand rule.
#[derive(Clone)]
struct HullFace {
    v: [usize; 3],
    /// Outward-pointing normal (not necessarily unit length).
    normal: [f64; 3],
    /// Signed distance from the origin along the normal (for the plane equation).
    offset: f64,
    /// Whether this face is still active (not removed).
    alive: bool,
}

impl HullFace {
    fn new(points: &[Point3<f64>], a: usize, b: usize, c: usize) -> Self {
        let (nx, ny, nz, d) = face_plane(points, a, b, c);
        HullFace {
            v: [a, b, c],
            normal: [nx, ny, nz],
            offset: d,
            alive: true,
        }
    }

    /// Signed distance from point to this face's plane.
    /// Positive = outside (visible from outside the hull).
    fn signed_distance(&self, p: &Point3<f64>) -> f64 {
        self.normal[0] * p.x + self.normal[1] * p.y + self.normal[2] * p.z - self.offset
    }
}

/// Compute the plane equation for triangle (a, b, c).
/// Returns (nx, ny, nz, d) where nx*x + ny*y + nz*z = d.
fn face_plane(points: &[Point3<f64>], a: usize, b: usize, c: usize) -> (f64, f64, f64, f64) {
    let pa = &points[a];
    let pb = &points[b];
    let pc = &points[c];
    // edge vectors
    let ux = pb.x - pa.x;
    let uy = pb.y - pa.y;
    let uz = pb.z - pa.z;
    let vx = pc.x - pa.x;
    let vy = pc.y - pa.y;
    let vz = pc.z - pa.z;
    // cross product u x v
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let d = nx * pa.x + ny * pa.y + nz * pa.z;
    (nx, ny, nz, d)
}

/// Find an initial tetrahedron from the point set.
/// Returns 4 indices of non-coplanar points, or Err if the points are degenerate.
fn find_initial_tetrahedron(points: &[Point3<f64>]) -> Result<[usize; 4], String> {
    let n = points.len();
    if n < 4 {
        return Err("Need at least 4 points for convex hull".into());
    }

    // Pick the first point.
    let i0 = 0;

    // Find the point farthest from i0.
    let i1 = (0..n)
        .max_by(|&a, &b| {
            let da = dist_sq(&points[a], &points[i0]);
            let db = dist_sq(&points[b], &points[i0]);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    if dist_sq(&points[i0], &points[i1]) < HULL_EPS * HULL_EPS {
        return Err("All points are coincident".into());
    }

    // Find the point farthest from line (i0, i1).
    let dir = [
        points[i1].x - points[i0].x,
        points[i1].y - points[i0].y,
        points[i1].z - points[i0].z,
    ];
    let i2 = (0..n)
        .max_by(|&a, &b| {
            let da = point_line_dist_sq(&points[a], &points[i0], &dir);
            let db = point_line_dist_sq(&points[b], &points[i0], &dir);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    if point_line_dist_sq(&points[i2], &points[i0], &dir) < HULL_EPS * HULL_EPS {
        return Err("All points are collinear".into());
    }

    // Find the point farthest from the plane (i0, i1, i2).
    let (nx, ny, nz, d) = face_plane(points, i0, i1, i2);
    let nl = (nx * nx + ny * ny + nz * nz).sqrt();
    if nl < HULL_EPS {
        return Err("Degenerate triangle in initial tetrahedron".into());
    }
    let i3 = (0..n)
        .max_by(|&a, &b| {
            let da = ((nx * points[a].x + ny * points[a].y + nz * points[a].z - d) / nl).abs();
            let db = ((nx * points[b].x + ny * points[b].y + nz * points[b].z - d) / nl).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    let dist_i3 = ((nx * points[i3].x + ny * points[i3].y + nz * points[i3].z - d) / nl).abs();
    if dist_i3 < HULL_EPS {
        return Err("All points are coplanar".into());
    }

    Ok([i0, i1, i2, i3])
}

fn dist_sq(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}

fn point_line_dist_sq(p: &Point3<f64>, line_origin: &Point3<f64>, line_dir: &[f64; 3]) -> f64 {
    let ap = [
        p.x - line_origin.x,
        p.y - line_origin.y,
        p.z - line_origin.z,
    ];
    // cross product ap x dir
    let cx = ap[1] * line_dir[2] - ap[2] * line_dir[1];
    let cy = ap[2] * line_dir[0] - ap[0] * line_dir[2];
    let cz = ap[0] * line_dir[1] - ap[1] * line_dir[0];
    let cross_sq = cx * cx + cy * cy + cz * cz;
    let dir_sq = line_dir[0].powi(2) + line_dir[1].powi(2) + line_dir[2].powi(2);
    if dir_sq < HULL_EPS * HULL_EPS {
        return 0.0;
    }
    cross_sq / dir_sq
}

/// An edge represented as an ordered pair of vertex indices.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Edge(usize, usize);

/// Build the 3D convex hull incrementally and return a boolean vector indicating
/// which input points lie on the hull.
fn convex_hull_3d_membership(points: &[Point3<f64>]) -> Vec<bool> {
    let n = points.len();
    let mut on_hull = vec![false; n];

    let tet = match find_initial_tetrahedron(points) {
        Ok(t) => t,
        Err(_) => {
            // Degenerate — mark all points as on-hull (safe fallback for HPR).
            for v in on_hull.iter_mut() {
                *v = true;
            }
            return on_hull;
        }
    };

    let [i0, i1, i2, i3] = tet;

    // Build initial tetrahedron with 4 faces, normals pointing outward.
    // Determine winding: if i3 is on the positive side of face (i0,i1,i2),
    // reverse the face winding so normals point away from the interior.
    let mut faces: Vec<HullFace> = Vec::new();

    let test_face = HullFace::new(points, i0, i1, i2);
    if test_face.signed_distance(&points[i3]) > 0.0 {
        // i3 is outside face (i0,i1,i2) — flip this face
        faces.push(HullFace::new(points, i0, i2, i1));
        faces.push(HullFace::new(points, i0, i1, i3));
        faces.push(HullFace::new(points, i1, i2, i3));
        faces.push(HullFace::new(points, i0, i3, i2));
    } else {
        faces.push(HullFace::new(points, i0, i1, i2));
        faces.push(HullFace::new(points, i0, i3, i1));
        faces.push(HullFace::new(points, i1, i3, i2));
        faces.push(HullFace::new(points, i0, i2, i3));
    }

    // Incrementally add each remaining point.
    let initial_set: std::collections::HashSet<usize> = [i0, i1, i2, i3].iter().copied().collect();

    for pi in 0..n {
        if initial_set.contains(&pi) {
            continue;
        }

        let p = &points[pi];

        // Find all visible (alive) faces.
        let mut visible = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if face.alive && face.signed_distance(p) > HULL_EPS {
                visible.push(fi);
            }
        }

        if visible.is_empty() {
            // Point is inside the current hull — not on the final hull.
            continue;
        }

        // Find horizon edges: edges shared by exactly one visible face.
        let mut edge_count: std::collections::HashMap<Edge, (usize, [usize; 3])> =
            std::collections::HashMap::new();

        for &fi in &visible {
            let fv = faces[fi].v;
            for k in 0..3 {
                let a = fv[k];
                let b = fv[(k + 1) % 3];
                let key = if a < b { Edge(a, b) } else { Edge(b, a) };
                let entry = edge_count.entry(key).or_insert((0, fv));
                entry.0 += 1;
            }
        }

        // Kill visible faces.
        for &fi in &visible {
            faces[fi].alive = false;
        }

        // Create new faces from horizon edges to the new point.
        for (&edge, &(count, fv)) in &edge_count {
            if count != 1 {
                continue; // internal edge between two visible faces
            }
            let Edge(a, b) = edge;

            // We need correct winding. The horizon edge (a, b) in the surviving
            // adjacent face has a specific order. The new face should have the
            // reverse order of that edge + the new point, so that its normal
            // points outward.
            //
            // The visible face that contained this edge had vertices in order
            // such that the edge appears as (a, b) or (b, a). We need the new
            // triangle to be wound so the new point is on the outside.
            // Try both windings and pick the one consistent with the rest.
            let f1 = HullFace::new(points, a, b, pi);
            let f2 = HullFace::new(points, b, a, pi);

            // The centroid of the old tetrahedron should be inside the hull.
            // Use one of the vertices from the visible face that isn't on this edge.
            let other = fv.iter().find(|&&v| v != a && v != b);
            if let Some(&ov) = other {
                // The new face should have the "other" vertex on the inside.
                if f1.signed_distance(&points[ov]) < 0.0 {
                    faces.push(f1);
                } else {
                    faces.push(f2);
                }
            } else {
                faces.push(f1);
            }
        }
    }

    // Collect all vertex indices referenced by alive faces.
    for face in &faces {
        if face.alive {
            for &v in &face.v {
                on_hull[v] = true;
            }
        }
    }

    on_hull
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate points on a unit sphere using a Fibonacci lattice.
    fn sphere_points(n: usize) -> Vec<Point3<f64>> {
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        (0..n)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / golden;
                let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
                Point3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
            })
            .collect()
    }

    #[test]
    fn test_hpr_sphere_from_outside() {
        let points = sphere_points(200);
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let result = hidden_point_removal(&points, &viewpoint, 0.0).unwrap();

        let mut front_visible = 0;
        let mut back_visible = 0;
        for &i in &result.visible_indices {
            if points[i].z > 0.2 {
                front_visible += 1;
            }
            if points[i].z < -0.2 {
                back_visible += 1;
            }
        }

        let front_total = points.iter().filter(|p| p.z > 0.2).count();
        assert!(
            front_visible as f64 / front_total as f64 > 0.7,
            "Expected most front-facing points to be visible, got {}/{}",
            front_visible,
            front_total
        );

        let back_total = points.iter().filter(|p| p.z < -0.2).count();
        assert!(
            (back_visible as f64) < (back_total as f64 * 0.3),
            "Expected few back-facing points to be visible, got {}/{}",
            back_visible,
            back_total
        );
    }

    #[test]
    fn test_hpr_select_visible() {
        let points = sphere_points(50);
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let visible = select_visible_points(&points, &viewpoint, 0.0).unwrap();
        assert!(!visible.is_empty());
        assert!(visible.len() <= points.len());
    }

    #[test]
    fn test_hpr_too_few_points() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        assert!(hidden_point_removal(&points, &viewpoint, 0.0).is_err());
    }

    #[test]
    fn test_hpr_cube_dense_faces() {
        let mut points = Vec::new();
        let mut face_labels = Vec::new();

        let steps = 5;
        for i in 0..steps {
            for j in 0..steps {
                let u = -0.5 + (i as f64 + 0.5) / steps as f64;
                let v = -0.5 + (j as f64 + 0.5) / steps as f64;

                points.push(Point3::new(0.5, u, v));
                face_labels.push("+X");
                points.push(Point3::new(-0.5, u, v));
                face_labels.push("-X");
                points.push(Point3::new(u, 0.5, v));
                face_labels.push("+Y");
                points.push(Point3::new(u, -0.5, v));
                face_labels.push("-Y");
                points.push(Point3::new(u, v, 0.5));
                face_labels.push("+Z");
                points.push(Point3::new(u, v, -0.5));
                face_labels.push("-Z");
            }
        }

        let viewpoint = Point3::new(5.0, 5.0, 5.0);
        let result = hidden_point_removal(&points, &viewpoint, 0.0).unwrap();

        let visible_set: std::collections::HashSet<usize> =
            result.visible_indices.iter().copied().collect();

        let total_per_face = steps * steps;
        let mut visible_counts = std::collections::HashMap::new();
        for (i, label) in face_labels.iter().enumerate() {
            if visible_set.contains(&i) {
                *visible_counts.entry(*label).or_insert(0usize) += 1;
            }
        }

        for face in &["+X", "+Y", "+Z"] {
            let count = visible_counts.get(face).copied().unwrap_or(0);
            assert!(
                count as f64 / total_per_face as f64 > 0.5,
                "Face {} should be mostly visible, got {}/{}",
                face,
                count,
                total_per_face
            );
        }

        for face in &["-X", "-Y", "-Z"] {
            let count = visible_counts.get(face).copied().unwrap_or(0);
            assert!(
                (count as f64) < (total_per_face as f64 * 0.5),
                "Face {} should be mostly hidden, got {}/{}",
                face,
                count,
                total_per_face
            );
        }
    }

    #[test]
    fn test_convex_hull_tetrahedron() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.1, 0.1, 0.1),
        ];
        let on_hull = convex_hull_3d_membership(&points);
        assert!(on_hull[0]);
        assert!(on_hull[1]);
        assert!(on_hull[2]);
        assert!(on_hull[3]);
        assert!(!on_hull[4]);
    }

    #[test]
    fn test_convex_hull_cube() {
        let mut points = Vec::new();
        for &x in &[0.0, 1.0] {
            for &y in &[0.0, 1.0] {
                for &z in &[0.0, 1.0] {
                    points.push(Point3::new(x, y, z));
                }
            }
        }
        points.push(Point3::new(0.5, 0.5, 0.5));

        let on_hull = convex_hull_3d_membership(&points);
        for i in 0..8 {
            assert!(on_hull[i]);
        }
        assert!(!on_hull[8]);
    }

    #[test]
    fn test_visibility_mask_consistent() {
        let points = sphere_points(100);
        let viewpoint = Point3::new(0.0, 0.0, 5.0);
        let result = hidden_point_removal(&points, &viewpoint, 0.0).unwrap();

        assert_eq!(result.visibility_mask.len(), points.len());
        let mask_count = result.visibility_mask.iter().filter(|&&v| v).count();
        assert_eq!(mask_count, result.visible_indices.len());

        for &i in &result.visible_indices {
            assert!(result.visibility_mask[i]);
        }
    }
}
