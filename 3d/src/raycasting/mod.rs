//! Ray Casting Module
//!
//! Ray-mesh and ray-pointcloud intersection queries.

use crate::mesh::TriangleMesh;
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
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
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

/// Cast a ray against a triangle mesh
pub fn cast_ray_mesh(ray: &Ray, mesh: &TriangleMesh) -> Option<RayHit> {
    let mut closest_hit: Option<RayHit> = None;
    let mut closest_dist = f32::MAX;

    for (tri_idx, face) in mesh.faces.iter().enumerate() {
        let v0 = mesh.vertices[face[0]];
        let v1 = mesh.vertices[face[1]];
        let v2 = mesh.vertices[face[2]];

        if let Some(hit) = ray_triangle_intersection(ray, v0, v1, v2) {
            if hit.distance < closest_dist && hit.distance > 0.0 {
                closest_dist = hit.distance;
                closest_hit = Some(RayHit {
                    distance: hit.distance,
                    point: hit.point,
                    normal: hit.normal,
                    triangle_index: tri_idx,
                    barycentric: hit.barycentric,
                });
            }
        }
    }

    closest_hit
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
    group.run(|| {
        rays.par_iter()
            .map(|ray| cast_ray_mesh(ray, mesh))
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

    if a.abs() < epsilon {
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

struct RayHitInfo {
    distance: f32,
    point: Point3<f32>,
    normal: Vector3<f32>,
    barycentric: (f32, f32, f32),
}

/// Distance query: closest point on mesh to query point
pub fn closest_point_on_mesh(
    query: &Point3<f32>,
    mesh: &TriangleMesh,
) -> (Point3<f32>, f32, usize) {
    let mut closest_point = mesh.vertices[0];
    let mut closest_dist = (query.coords - closest_point.coords).norm();
    let mut closest_tri = 0;

    for (tri_idx, face) in mesh.faces.iter().enumerate() {
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

/// Compute mesh distance to another mesh (Hausdorff distance)
pub fn mesh_to_mesh_distance(source: &TriangleMesh, target: &TriangleMesh) -> (f32, f32) {
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

/// Check if point is inside mesh (ray casting method)
pub fn point_inside_mesh(query: &Point3<f32>, mesh: &TriangleMesh) -> bool {
    // Cast ray in +X direction and count intersections
    let ray = Ray::new(*query, Vector3::x());
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

    // Odd number of intersections = inside
    intersection_count % 2 == 1
}
