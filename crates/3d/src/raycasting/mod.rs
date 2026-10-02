//! Ray Casting Module
//!
//! Ray-mesh and ray-pointcloud intersection queries.

use crate::mesh::TriangleMesh;
use crate::spatial::bvh::Bvh;
use nalgebra::{Point3, Vector3};

use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::RuntimeRunner;

/// Ray representation
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

impl Ray {
    /// Build a ray from an origin and a direction.
    ///
    /// The direction is normalised. A **zero** direction has no normalisation,
    /// and `Vector3::normalize` returns NaN for it - so every later `point_at`,
    /// dot product and intersection returned NaN with no error anywhere, and a
    /// NaN propagates silently through any transform that touches it.
    ///
    /// There is no correct direction to invent, so a degenerate input is reported
    /// rather than papered over. `try_new` returns `None`; `new` panics with a
    /// clear message rather than constructing a ray that cannot be used.
    pub fn try_new(origin: Point3<f32>, direction: Vector3<f32>) -> Option<Self> {
        // The only invalid norms are non-finite ones and exactly zero. A
        // *small* norm is still a direction: `1e-7` is a perfectly good unit
        // vector after normalisation, and rejecting it would refuse legitimate
        // input. An earlier version of this guard used `<= f32::EPSILON`,
        // which threw away directions that normalise perfectly well.
        let norm = direction.norm();
        if !norm.is_finite() || norm == 0.0 {
            return None;
        }
        Some(Self {
            origin,
            direction: direction / norm,
        })
    }

    /// # Panics
    ///
    /// If `direction` is zero-length or non-finite. Use [`Ray::try_new`] to
    /// handle that case.
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self::try_new(origin, direction).unwrap_or_else(|| {
            panic!(
                "Ray::new given a degenerate direction {direction:?}: a ray with no \
                 direction cannot be normalised, and normalising it anyway yields NaN"
            )
        })
    }

    pub fn point_at(&self, t: f32) -> Point3<f32> {
        Point3::from(self.origin.coords + self.direction * t)
    }
}

/// Ray hit information
#[derive(Debug, Clone)]
pub struct RayHit {
    pub distance: f32,
    pub point: Point3<f32>,
    pub normal: Vector3<f32>,
    pub triangle_index: usize,
    pub barycentric: (f32, f32, f32),
}

/// Cast a ray against a triangle mesh using BVH acceleration — O(log N).
pub fn cast_ray_mesh(ray: &Ray, mesh: &TriangleMesh) -> Option<RayHit> {
    let bvh = Bvh::build(&mesh.vertices, &mesh.faces);
    cast_ray_mesh_bvh(ray, mesh, &bvh)
}

/// Cast a ray against a triangle mesh using a pre-built BVH.
pub fn cast_ray_mesh_bvh(ray: &Ray, mesh: &TriangleMesh, bvh: &Bvh) -> Option<RayHit> {
    bvh.intersect_ray(&ray.origin, &ray.direction, &mesh.vertices, &mesh.faces)
        .map(|(t, tri_idx, u, v)| {
            let face = &mesh.faces[tri_idx];
            let v0 = mesh.vertices[face[0]];
            let v1 = mesh.vertices[face[1]];
            let v2 = mesh.vertices[face[2]];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let mut normal = e1.cross(&e2);
            let len = normal.norm();
            // Same scale trap as the Möller-Trumbore parallel guard, in its
            // `|a| < eps` form: `e1.cross(&e2)` is the *unnormalised* normal, so
            // its length is the triangle area and goes as L². Measured: an
            // equilateral triangle of edge 1 gives len = 0.866 and is
            // normalised; of edge 1e-5, len = 8.66e-11, which is below 1e-9, so
            // the division was skipped and the caller received a "normal" of
            // [[0, 0, 8.66e-11]] — a unit vector of length 1e-10, not a
            // direction. `RayHit::normal` is documented as a normal, and every
            // lighting, back-face and shading consumer normalises it itself.
            //
            // `len > 0.0` is the correct test: a non-zero cross product is
            // already a direction, however short. It degenerates to the same
            // refusal on a zero-area triangle, where `len` is exactly 0.
            if len > 0.0 {
                normal /= len;
            }
            RayHit {
                distance: t,
                point: Point3::from(ray.origin.coords + ray.direction * t),
                normal,
                triangle_index: tri_idx,
                barycentric: (1.0 - u - v, u, v),
            }
        })
}

/// Cast multiple rays against a mesh using best available runner
pub fn cast_rays_mesh(rays: &[Ray], mesh: &TriangleMesh) -> Vec<Option<RayHit>> {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback to CPU registry on error
        cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
    });
    cast_rays_mesh_ctx(rays, mesh, &runner)
}

/// Cast multiple rays against a mesh with explicit context
pub fn cast_rays_mesh_ctx(
    rays: &[Ray],
    mesh: &TriangleMesh,
    group: &RuntimeRunner,
) -> Vec<Option<RayHit>> {
    // GPU Path
    if let Ok(ComputeDevice::Gpu(gpu)) = group.device() {
        let rays_tuples: Vec<(Point3<f32>, Vector3<f32>)> =
            rays.iter().map(|r| (r.origin, r.direction)).collect();
        let gpu_faces: Vec<[u32; 3]> = mesh
            .faces
            .iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        if let Ok(gpu_hits) = cv_hal::gpu_kernels::raycasting_gpu::cast_rays(
            gpu,
            &rays_tuples,
            &mesh.vertices,
            &gpu_faces,
        ) {
            return gpu_hits
                .into_iter()
                .map(|hit| {
                    hit.map(|(dist, point, normal)| {
                        RayHit {
                            distance: dist,
                            point,
                            normal,
                            triangle_index: 0, // GPU doesn't return index yet
                            barycentric: (0.0, 0.0, 0.0), // GPU doesn't return barycentric yet
                        }
                    })
                })
                .collect();
        }
    }

    use rayon::prelude::*;
    let bvh = Bvh::build(&mesh.vertices, &mesh.faces);
    group.run(|| {
        rays.par_iter()
            .map(|ray| cast_ray_mesh_bvh(ray, mesh, &bvh))
            .collect()
    })
}

/// Ray-triangle intersection using Möller-Trumbore algorithm
fn ray_triangle_intersection(
    ray: &Ray,
    v0: Point3<f32>,
    v1: Point3<f32>,
    v2: Point3<f32>,
) -> Option<RayHitInfo> {
    let epsilon = 1e-6;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(&edge2);
    let a = edge1.dot(&h);

    // `a` is twice the projected triangle area, so for a genuine hit it scales
    // as L^2 - the *square* of the edge length. `a.abs() < epsilon` was
    // therefore never an angular tolerance; it was a test for "edge length
    // below about 1e-3", and the length it rejected moved with the world units
    // the mesh happened to be stored in. Measured on an equilateral triangle
    // hit dead-on: edge 1.0 gives |a| = 8.66e-1 (hit), edge 1e-3 gives
    // |a| = 8.66e-7 (MISS), edge 1e-5 gives |a| = 8.66e-11 (MISS). A
    // metre-unit mesh of 1 mm triangles is exactly what a surface
    // reconstruction emits, and every ray through it was rejected as
    // "parallel to triangle".
    //
    // Dividing by |e1||e2| makes the test what it says it is: a dimensionless
    // bound on the angle between the ray and the triangle normal. `epsilon` is
    // kept at its previous value, so the angular sharpness is unchanged and
    // only the scale dependence is removed.
    let denom = edge1.norm() * edge2.norm();
    if denom == 0.0 || a.abs() < epsilon * denom {
        return None; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(&h);

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(&edge1);
    let v = f * ray.direction.dot(&q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(&q);

    if t > epsilon {
        let w = 1.0 - u - v;
        let point = ray.point_at(t);
        let normal = edge1.cross(&edge2).normalize();

        Some(RayHitInfo {
            distance: t,
            point,
            normal,
            barycentric: (u, v, w),
        })
    } else {
        None
    }
}

#[allow(dead_code)]
struct RayHitInfo {
    distance: f32,
    point: Point3<f32>,
    normal: Vector3<f32>,
    barycentric: (f32, f32, f32),
}

/// Triangle index returned by [`closest_point_on_mesh`] when the mesh has no
/// surface to report on — it holds no faces, or no vertices at all.
///
/// This is deliberately **not** `0`. Zero is a perfectly valid index into
/// `mesh.faces`, so returning it for a faceless mesh handed callers a triangle
/// that did not exist; `usize::MAX` cannot be dereferenced into `mesh.faces` at
/// all and so fails loudly instead of lying quietly.
pub const NO_TRIANGLE: usize = usize::MAX;

/// Distance query: closest point on mesh to query point.
///
/// # Returns
/// `(closest_point, distance, triangle_index)`.
///
/// # The absent-geometry case
///
/// A `TriangleMesh` can hold vertices without holding any *faces*: the struct's
/// two fields are public and independent. There is then no surface to project
/// onto, so no distance exists and no triangle index can be meaningful. The
/// function used to guard only `mesh.vertices.is_empty()`, which meant a
/// vertices-without-faces mesh fell straight through: the search loop over
/// `mesh.faces` never ran and the function returned
/// `(vertices[0], distance_to_vertex_0, 0)` — naming triangle 0 of a mesh that
/// has no triangles. A caller using the index to look up a face got a wrong face
/// or panicked.
///
/// Absence is now reported the way this module already reports it: an
/// **infinite** distance, the query point echoed back unchanged, and
/// `NO_TRIANGLE` as the index. `NO_TRIANGLE` is deliberately not a valid index
/// into `mesh.faces`, so a caller that forgets to check the distance cannot
/// silently dereference a face that does not exist.
pub fn closest_point_on_mesh(
    query: &Point3<f32>,
    mesh: &TriangleMesh,
) -> (Point3<f32>, f32, usize) {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        // No geometry to project onto; report an infinite distance instead of
        // panicking on `mesh.vertices[0]` or fabricating a triangle index.
        return (*query, f32::INFINITY, NO_TRIANGLE);
    }

    // Seed from the first real triangle rather than from `vertices[0]`: with no
    // faces there is nothing to seed from, and with faces the seed is replaced
    // by the loop below anyway.
    let first = mesh.faces[0];
    let (mut closest_point, mut closest_dist) = closest_point_on_triangle(
        query,
        mesh.vertices[first[0]],
        mesh.vertices[first[1]],
        mesh.vertices[first[2]],
    );
    let mut closest_tri = 0usize;

    for (tri_idx, face) in mesh.faces.iter().enumerate().skip(1) {
        let v0 = mesh.vertices[face[0]];
        let v1 = mesh.vertices[face[1]];
        let v2 = mesh.vertices[face[2]];

        let (point, dist) = closest_point_on_triangle(query, v0, v1, v2);

        if dist < closest_dist {
            closest_dist = dist;
            closest_point = point;
            closest_tri = tri_idx;
        }
    }

    (closest_point, closest_dist, closest_tri)
}

/// Closest point on triangle to query point
fn closest_point_on_triangle(
    query: &Point3<f32>,
    v0: Point3<f32>,
    v1: Point3<f32>,
    v2: Point3<f32>,
) -> (Point3<f32>, f32) {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = query.coords - v0.coords;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);

    if d1 <= 0.0 && d2 <= 0.0 {
        return (v0, (query.coords - v0.coords).norm());
    }

    let bp = query.coords - v1.coords;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);

    if d3 >= 0.0 && d4 <= d3 {
        return (v1, (query.coords - v1.coords).norm());
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let point = v0 + ab * v;
        return (point, (query.coords - point.coords).norm());
    }

    let cp = query.coords - v2.coords;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);

    if d6 >= 0.0 && d5 <= d6 {
        return (v2, (query.coords - v2.coords).norm());
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let point = v0 + ac * w;
        return (point, (query.coords - point.coords).norm());
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let point = v1 + (v2.coords - v1.coords) * w;
        return (point, (query.coords - point.coords).norm());
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let point = v0 + ab * v + ac * w;
    (point, (query.coords - point.coords).norm())
}

/// Batch distance queries using best available runner
pub fn closest_points_on_mesh(
    queries: &[Point3<f32>],
    mesh: &TriangleMesh,
) -> Vec<(Point3<f32>, f32, usize)> {
    let runner = cv_runtime::best_runner().unwrap_or_else(|_| {
        // Fallback to CPU registry on error
        cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
    });
    closest_points_on_mesh_ctx(queries, mesh, &runner)
}

/// Batch distance queries with explicit context
pub fn closest_points_on_mesh_ctx(
    queries: &[Point3<f32>],
    mesh: &TriangleMesh,
    group: &RuntimeRunner,
) -> Vec<(Point3<f32>, f32, usize)> {
    use rayon::prelude::*;

    group.run(|| {
        queries
            .par_iter()
            .map(|query| closest_point_on_mesh(query, mesh))
            .collect()
    })
}

/// Whether a mesh carries a *surface* — the thing the distance queries in this
/// module measure against. Vertices alone are not a surface: with no faces there
/// is nothing to project a query point onto, and reporting a finite distance
/// would be reporting against a point cloud while the API says "mesh".
fn has_surface(mesh: &TriangleMesh) -> bool {
    !mesh.vertices.is_empty() && !mesh.faces.is_empty()
}

/// Compute mesh distance to another mesh (Hausdorff distance).
///
/// Returns `(hausdorff, forward_mean)`.
///
/// # The absent-geometry case
///
/// The previous guard was
/// ```ignore
/// if source.vertices.is_empty() || target.vertices.is_empty() {
///     return (0.0, 0.0);   // "finite zeros"
/// }
/// ```
/// which avoided the NaN from a `0/0` mean but replaced it with a worse lie:
/// `(0.0, 0.0)` is the exact value two **identical, fully overlapping** meshes
/// produce, so "there is no geometry here" was indistinguishable from "perfect
/// match". `mesh_to_mesh_distance(&TriangleMesh::new(), &cube)` returned
/// `(0.0, 0.0)`.
///
/// The NaN-avoidance intent was right; the replacement was not. Distance is
/// measured between *surfaces*, so a mesh with no faces has no surface and no
/// distance exists — which is precisely the `f32::INFINITY` convention
/// [`closest_point_on_mesh`] already uses one function above. Both components
/// are now `f32::INFINITY`, so absence propagates through `max` and averages as
/// absence rather than collapsing to the perfect-match value.
pub fn mesh_to_mesh_distance(source: &TriangleMesh, target: &TriangleMesh) -> (f32, f32) {
    if !has_surface(source) || !has_surface(target) {
        // No geometry on one side: the distances are undefined, not zero. An
        // infinite value keeps this distinguishable from a perfect match and
        // cannot be mistaken for a small real distance.
        return (f32::INFINITY, f32::INFINITY);
    }

    // Forward distance: source -> target
    let forward_dists: Vec<f32> = source
        .vertices
        .iter()
        .map(|v| closest_point_on_mesh(v, target).1)
        .collect();

    let forward_max = forward_dists.iter().cloned().fold(0.0, f32::max);
    let forward_mean = forward_dists.iter().sum::<f32>() / forward_dists.len() as f32;

    // Backward distance: target -> source
    let backward_dists: Vec<f32> = target
        .vertices
        .iter()
        .map(|v| closest_point_on_mesh(v, source).1)
        .collect();

    let backward_max = backward_dists.iter().cloned().fold(0.0, f32::max);

    // Symmetric Hausdorff distance
    let hausdorff = forward_max.max(backward_max);

    (hausdorff, forward_mean)
}

/// Check if point is inside mesh (ray casting method, BVH-accelerated)
pub fn point_inside_mesh(query: &Point3<f32>, mesh: &TriangleMesh) -> bool {
    let bvh = Bvh::build(&mesh.vertices, &mesh.faces);
    point_inside_mesh_bvh(query, mesh, &bvh)
}

/// Check if point is inside mesh using a pre-built BVH.
pub fn point_inside_mesh_bvh(query: &Point3<f32>, mesh: &TriangleMesh, _bvh: &Bvh) -> bool {
    // Cast ray in +X direction and count intersections via BVH
    let ray = Ray::new(*query, Vector3::x());
    // BVH gives us only the closest hit. To count all intersections,
    // we need to walk through all hits. For now, fall back to brute force
    // since BVH only returns closest. This is still correct.
    let mut intersection_count = 0;
    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0]];
        let v1 = mesh.vertices[face[1]];
        let v2 = mesh.vertices[face[2]];
        if let Some(hit) = ray_triangle_intersection(&ray, v0, v1, v2) {
            if hit.distance > 0.0 {
                intersection_count += 1;
            }
        }
    }
    intersection_count % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_triangle_hit() {
        // Ray from (0,0,-1) in direction (0,0,1) at a triangle on the XY-plane
        let ray = Ray::new(Point3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let v0 = Point3::new(-1.0, -1.0, 0.0);
        let v1 = Point3::new(1.0, -1.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        let hit = ray_triangle_intersection(&ray, v0, v1, v2);
        assert!(hit.is_some(), "Ray should hit the triangle");

        let hit = hit.unwrap();
        assert!(
            (hit.distance - 1.0).abs() < 1e-5,
            "Expected hit distance ~1.0, got {}",
            hit.distance
        );
    }

    #[test]
    fn test_ray_triangle_miss() {
        // Ray that misses the triangle (offset in X so it passes beside the triangle)
        let ray = Ray::new(Point3::new(5.0, 5.0, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let v0 = Point3::new(-1.0, -1.0, 0.0);
        let v1 = Point3::new(1.0, -1.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        let hit = ray_triangle_intersection(&ray, v0, v1, v2);
        assert!(hit.is_none(), "Ray should miss the triangle");
    }

    #[test]
    fn test_closest_point_on_empty_mesh() {
        let mesh = TriangleMesh::new();
        let query = Point3::new(1.0, 2.0, 3.0);
        let (p, d, tri) = closest_point_on_mesh(&query, &mesh);
        assert_eq!(p, query);
        assert!(d.is_infinite());
        assert_eq!(tri, NO_TRIANGLE, "there is no triangle to name");
    }

    /// A mesh can hold vertices without holding faces; the index must not be
    /// fabricated.
    #[test]
    fn test_closest_point_on_mesh_without_faces() {
        let mesh = TriangleMesh::with_vertices_and_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![],
        );
        let query = Point3::new(1.0, 2.0, 3.0);
        let (p, d, tri) = closest_point_on_mesh(&query, &mesh);
        assert_eq!(p, query);
        assert!(
            d.is_infinite(),
            "no faces means no surface to measure against, got {d}"
        );
        assert!(
            tri >= mesh.faces.len(),
            "returned triangle {tri} for a mesh with {} faces",
            mesh.faces.len()
        );
    }

    /// Absence must not look like a perfect match. This test used to assert
    /// only `is_finite()`, which pinned the fabricated `(0.0, 0.0)` in place.
    #[test]
    fn test_mesh_to_mesh_distance_empty_mesh_is_not_a_perfect_match() {
        let empty = TriangleMesh::new();
        let (hausdorff, mean) = mesh_to_mesh_distance(&empty, &empty);

        assert!(
            !hausdorff.is_nan() && !mean.is_nan(),
            "absence must be reported, not a NaN from 0/0"
        );
        assert!(
            hausdorff.is_infinite() && mean.is_infinite(),
            "an empty mesh has no surface, so the distance is undefined; expected \
             INFINITY (the closest_point_on_mesh convention), got \
             hausdorff={hausdorff} mean={mean}"
        );

        // CONTROL: a genuine perfect match is 0.0 and must stay 0.0, so the two
        // remain distinguishable.
        let cube = TriangleMesh::with_vertices_and_faces(
            vec![
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(1.0, 1.0, -1.0),
            ],
            vec![[0, 1, 2]],
        );
        let (h, m) = mesh_to_mesh_distance(&cube, &cube);
        assert_eq!(
            (h, m),
            (0.0, 0.0),
            "a mesh measured against itself is a perfect match, not absence"
        );
    }
}
